"""Vision capture, preprocessing, inference and evidence contracts for M4."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from .calibration import CalibrationRecord, CalibrationService
from .camera import CameraOwnership


VISION_PROFILE_VERSION = "m4.vision.v1"
SUPPORTED_MODEL_EXTENSIONS = {".pt", ".onnx"}
ProgressCallback = Callable[[str], None]


class VisionError(ValueError):
    """Raised for invalid vision settings, models or frame geometry."""


class ModelLoadError(VisionError):
    """Raised when a local model cannot be loaded."""


@dataclass(frozen=True)
class RawDetection:
    """Adapter output in the model input coordinate space."""

    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    model_size: tuple[int, int] | None = None
    letterbox_scale: float | None = None
    letterbox_pad_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    frame_id: int
    captured_at: str
    coordinate_space: str = "full_frame"

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "center_px": list(self.center_px),
            "frame_id": self.frame_id,
            "captured_at": self.captured_at,
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True)
class FrameResult:
    frame_id: int
    captured_at: str
    raw_bgr: np.ndarray
    processed_bgr: np.ndarray
    annotated_bgr: np.ndarray
    roi: tuple[int, int, int, int]
    full_frame_size: tuple[int, int]
    detections: tuple[Detection, ...] = ()
    device: str = "preview only"


@dataclass
class VisionProfile:
    camera_index: int = 0
    backend: str | None = None
    requested_camera: dict[str, Any] = field(default_factory=dict)
    readback_camera: dict[str, Any] = field(default_factory=dict)
    frame_size: tuple[int, int] | None = None
    roi: dict[str, int] | None = None
    confidence: float = 0.5
    brightness: float = 0.0
    contrast: float = 1.0
    model_path: str | None = None
    model_sha256: str | None = None
    model_device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VISION_PROFILE_VERSION,
            "camera_index": self.camera_index,
            "backend": self.backend,
            "requested_camera": self.requested_camera,
            "readback_camera": self.readback_camera,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "roi": self.roi,
            "confidence": self.confidence,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "model_device": self.model_device,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisionProfile":
        version = data.get("schema_version")
        if version not in (None, VISION_PROFILE_VERSION):
            raise VisionError(f"Unsupported vision profile version: {version!r}")
        frame_size = data.get("frame_size")
        return cls(
            camera_index=int(data.get("camera_index", 0)),
            backend=data.get("backend"),
            requested_camera=dict(data.get("requested_camera", {})),
            readback_camera=dict(data.get("readback_camera", {})),
            frame_size=(int(frame_size[0]), int(frame_size[1])) if frame_size else None,
            roi=dict(data["roi"]) if data.get("roi") is not None else None,
            confidence=float(data.get("confidence", 0.5)),
            brightness=float(data.get("brightness", 0.0)),
            contrast=float(data.get("contrast", 1.0)),
            model_path=data.get("model_path"),
            model_sha256=data.get("model_sha256"),
            model_device=data.get("model_device"),
        )


@dataclass(frozen=True)
class ModelLoadRequest:
    path: Path
    expected_sha256: str | None = None
    label: str = "model"


class InferenceAdapter(Protocol):
    device: str
    class_names: Mapping[int, str]

    def infer(self, image_bgr: np.ndarray, confidence: float) -> Sequence[RawDetection]:
        ...


class UltralyticsInferenceAdapter:
    """Small adapter around a locally installed Ultralytics model."""

    def __init__(self, model: Any, model_path: Path) -> None:
        self.model = model
        self.model_path = model_path
        raw_names = getattr(model, "names", {}) or {}
        self.class_names = _normalise_class_names(raw_names)
        self.device = _model_device(model)

    @classmethod
    def load(cls, model_path: Path) -> "UltralyticsInferenceAdapter":
        validate_model_path(model_path)
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
        except Exception as error:  # Ultralytics wraps torch/ONNX errors inconsistently.
            raise ModelLoadError(f"Could not load model {model_path}: {error}") from error
        adapter = cls(model, model_path)
        if not adapter.class_names:
            adapter.class_names = {}
        return adapter

    def infer(self, image_bgr: np.ndarray, confidence: float) -> Sequence[RawDetection]:
        try:
            results = self.model.predict(
                source=image_bgr,
                conf=float(confidence),
                verbose=False,
            )
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                return ()
            xyxy = _tensor_array(boxes.xyxy)
            confidences = _tensor_array(boxes.conf).reshape(-1)
            classes = _tensor_array(boxes.cls).reshape(-1)
            height, width = image_bgr.shape[:2]
            names = _normalise_class_names(getattr(result, "names", self.class_names))
            if names:
                self.class_names = names
            return tuple(
                RawDetection(
                    class_id=int(classes[index]),
                    confidence=float(confidences[index]),
                    bbox_xyxy=tuple(float(value) for value in xyxy[index]),
                    model_size=(width, height),
                )
                for index in range(min(len(xyxy), len(confidences), len(classes)))
            )
        except Exception as error:
            raise VisionError(f"Inference failed: {error}") from error


def _tensor_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalise_class_names(value: Any) -> dict[int, str]:
    if isinstance(value, Mapping):
        return {int(key): str(name) for key, name in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {index: str(name) for index, name in enumerate(value)}
    return {}


def _model_device(model: Any) -> str:
    for candidate in (
        getattr(model, "device", None),
        getattr(getattr(model, "model", None), "device", None),
    ):
        if candidate is not None:
            return str(candidate)
    return "unknown"


def validate_model_path(path: Path) -> Path:
    model_path = path.expanduser().resolve()
    if model_path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
        raise ModelLoadError("Choose a local .pt or .onnx model file.")
    if not model_path.is_file():
        raise ModelLoadError(f"Model file does not exist: {model_path}")
    return model_path


def model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_roi(
    roi: Sequence[int] | None, frame_size: Sequence[int]
) -> tuple[int, int, int, int]:
    width, height = int(frame_size[0]), int(frame_size[1])
    if width <= 0 or height <= 0:
        raise VisionError("Frame size must be positive.")
    if roi is None:
        return 0, 0, width, height
    if len(roi) != 4:
        raise VisionError("ROI must contain x, y, width and height.")
    x, y, roi_width, roi_height = (int(value) for value in roi)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    roi_width = min(max(1, roi_width), width - x)
    roi_height = min(max(1, roi_height), height - y)
    return x, y, roi_width, roi_height


def apply_software_correction(
    frame_bgr: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    """Apply contrast first, then brightness, matching the recorded pipeline."""

    if contrast <= 0:
        raise VisionError("Contrast must be greater than zero.")
    corrected = frame_bgr.astype(np.float32) * float(contrast) + float(brightness)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def map_bbox_to_full_frame(
    bbox_xyxy: Sequence[float],
    roi: Sequence[int],
    source_size: Sequence[int],
    model_size: Sequence[int] | None = None,
    *,
    letterbox_scale: float | None = None,
    letterbox_pad_xy: Sequence[float] | None = None,
) -> tuple[float, float, float, float]:
    """Undo model resize/padding and ROI offset for a full-frame bbox."""

    if len(bbox_xyxy) != 4:
        raise VisionError("A detection box must contain four coordinates.")
    source_width, source_height = (float(source_size[0]), float(source_size[1]))
    if source_width <= 0 or source_height <= 0:
        raise VisionError("Source image size must be positive.")
    model_width, model_height = (
        (float(model_size[0]), float(model_size[1]))
        if model_size is not None
        else (source_width, source_height)
    )
    if model_width <= 0 or model_height <= 0:
        raise VisionError("Model image size must be positive.")
    scale = float(letterbox_scale) if letterbox_scale is not None else min(
        model_width / source_width, model_height / source_height
    )
    if scale <= 0:
        raise VisionError("Model resize scale must be positive.")
    if letterbox_pad_xy is None:
        pad_x = (model_width - source_width * scale) / 2.0
        pad_y = (model_height - source_height * scale) / 2.0
    else:
        pad_x, pad_y = (float(letterbox_pad_xy[0]), float(letterbox_pad_xy[1]))
    local = [
        (float(bbox_xyxy[0]) - pad_x) / scale,
        (float(bbox_xyxy[1]) - pad_y) / scale,
        (float(bbox_xyxy[2]) - pad_x) / scale,
        (float(bbox_xyxy[3]) - pad_y) / scale,
    ]
    local[0] = max(0.0, min(local[0], source_width))
    local[1] = max(0.0, min(local[1], source_height))
    local[2] = max(0.0, min(local[2], source_width))
    local[3] = max(0.0, min(local[3], source_height))
    roi_x, roi_y = float(roi[0]), float(roi[1])
    return (
        local[0] + roi_x,
        local[1] + roi_y,
        local[2] + roi_x,
        local[3] + roi_y,
    )


def process_frame(
    frame_bgr: np.ndarray,
    *,
    frame_id: int,
    roi: Sequence[int] | None,
    brightness: float,
    contrast: float,
    confidence: float,
    adapter: InferenceAdapter | None,
    captured_at: str | None = None,
) -> FrameResult:
    """Create a corrected ROI, run optional inference, and return full-frame detections."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise VisionError("Camera frames must be three-channel BGR images.")
    height, width = frame_bgr.shape[:2]
    full_size = (width, height)
    full_roi = normalise_roi(roi, full_size)
    x, y, roi_width, roi_height = full_roi
    crop = frame_bgr[y : y + roi_height, x : x + roi_width].copy()
    processed = apply_software_correction(crop, brightness, contrast)
    names = dict(getattr(adapter, "class_names", {}) or {}) if adapter else {}
    raw_detections = adapter.infer(processed, confidence) if adapter else ()
    timestamp = captured_at or _utc_now_iso()
    detections: list[Detection] = []
    for raw in raw_detections:
        full_bbox = map_bbox_to_full_frame(
            raw.bbox_xyxy,
            full_roi,
            (roi_width, roi_height),
            raw.model_size or (roi_width, roi_height),
            letterbox_scale=raw.letterbox_scale,
            letterbox_pad_xy=raw.letterbox_pad_xy,
        )
        center = ((full_bbox[0] + full_bbox[2]) / 2.0, (full_bbox[1] + full_bbox[3]) / 2.0)
        detections.append(
            Detection(
                class_id=int(raw.class_id),
                class_name=names.get(int(raw.class_id), f"class_{int(raw.class_id)}"),
                confidence=float(raw.confidence),
                bbox_xyxy=full_bbox,
                center_px=center,
                frame_id=frame_id,
                captured_at=timestamp,
            )
        )
    annotated = processed.copy()
    for detection in detections:
        local_box = (
            int(round(detection.bbox_xyxy[0] - x)),
            int(round(detection.bbox_xyxy[1] - y)),
            int(round(detection.bbox_xyxy[2] - x)),
            int(round(detection.bbox_xyxy[3] - y)),
        )
        cv2.rectangle(
            annotated,
            (local_box[0], local_box[1]),
            (local_box[2], local_box[3]),
            (40, 170, 90),
            2,
        )
        cv2.putText(
            annotated,
            f"{detection.class_name} {detection.confidence:.0%}",
            (max(0, local_box[0]), max(18, local_box[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 170, 90),
            2,
            cv2.LINE_AA,
        )
    device = getattr(adapter, "device", "preview only") if adapter else "preview only"
    return FrameResult(
        frame_id=frame_id,
        captured_at=timestamp,
        raw_bgr=frame_bgr.copy(),
        processed_bgr=processed,
        annotated_bgr=annotated,
        roi=full_roi,
        full_frame_size=full_size,
        detections=tuple(detections),
        device=str(device),
    )


class LatestFrameBuffer:
    """A one-slot result buffer. Publishing replaces stale work instead of queuing it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: FrameResult | None = None

    def publish(self, result: FrameResult) -> None:
        with self._lock:
            self._latest = result

    def take_latest(self) -> FrameResult | None:
        with self._lock:
            result, self._latest = self._latest, None
            return result

    def clear(self) -> None:
        with self._lock:
            self._latest = None


CAMERA_PROPERTY_NAMES = {
    "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
    "exposure": "CAP_PROP_EXPOSURE",
    "autofocus": "CAP_PROP_AUTOFOCUS",
    "focus": "CAP_PROP_FOCUS",
    "auto_white_balance": "CAP_PROP_AUTO_WB",
    "white_balance": "CAP_PROP_WB_TEMPERATURE",
}

# Each automatic switch and the manual value it controls. The switch is written first; the
# manual value is only written while the switch is off, because every OpenCV camera backend
# (DirectShow, Media Foundation, V4L2) forces manual mode when a manual value is written.
CAMERA_AUTO_PAIRS = (
    ("auto_exposure", "exposure"),
    ("autofocus", "focus"),
    ("auto_white_balance", "white_balance"),
)


def _write_camera_property(capture: Any, name: str, raw_requested_value: Any) -> dict[str, Any]:
    prop = getattr(cv2, CAMERA_PROPERTY_NAMES[name], None)
    if prop is None:
        return {"requested": raw_requested_value, "actual": None, "accepted": False, "supported": False}
    try:
        requested_value = float(raw_requested_value)
        accepted = bool(capture.set(prop, requested_value))
        actual = float(capture.get(prop))
        return {
            "requested": requested_value,
            "actual": actual if np.isfinite(actual) else None,
            "accepted": accepted,
            # Backends that do not implement a property (e.g. AVFoundation on macOS) return
            # False from set() and 0 or -1 from get(), so only an accepted write counts.
            "supported": accepted,
        }
    except Exception as error:
        return {
            "requested": raw_requested_value,
            "actual": None,
            "accepted": False,
            "supported": False,
            "error": str(error),
        }


def _read_camera_property(capture: Any, name: str) -> float | None:
    prop = getattr(cv2, CAMERA_PROPERTY_NAMES[name], None)
    if prop is None:
        return None
    try:
        actual = float(capture.get(prop))
    except Exception:
        return None
    return actual if np.isfinite(actual) else None


def apply_camera_properties(capture: Any, requested: Mapping[str, Any]) -> dict[str, Any]:
    """Request driver properties and return values read back from the driver.

    The automatic switch of each pair is applied before its manual value, and the manual
    value is skipped (reported with ``mode == "auto"``) while the switch is on, so enabling
    auto exposure/focus/white balance is not immediately undone by the manual slider.
    """

    readback: dict[str, Any] = {}
    for auto_name, manual_name in CAMERA_AUTO_PAIRS:
        auto_requested = requested.get(auto_name)
        auto_enabled = auto_requested is not None and float(auto_requested) != 0.0
        if auto_requested is not None:
            readback[auto_name] = _write_camera_property(capture, auto_name, auto_requested)
            readback[auto_name]["mode"] = "auto" if auto_enabled else "manual"
        manual_requested = requested.get(manual_name)
        if manual_requested is None:
            continue
        if auto_enabled:
            readback[manual_name] = {
                "requested": float(manual_requested),
                "actual": _read_camera_property(capture, manual_name),
                "accepted": None,
                "supported": bool(readback[auto_name]["supported"]),
                "mode": "auto",
            }
        else:
            readback[manual_name] = _write_camera_property(capture, manual_name, manual_requested)
            readback[manual_name]["mode"] = "manual"
    return readback


def _open_capture(camera_index: int) -> Any:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(camera_index)
    if not capture.isOpened() and sys.platform == "win32":
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    return capture


def _capture_metadata(
    capture: Any,
    *,
    camera_index: int,
    backend: str | None,
    requested_camera: Mapping[str, Any],
    readback_camera: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_type": "camera",
        "camera_index": camera_index,
        "backend": backend,
        "requested_camera": dict(requested_camera),
        "readback_camera": dict(readback_camera),
        "readback_width": float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "readback_height": float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "readback_fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }


class VisionWorker(QObject):
    """Capture and inference worker. UI receives only the latest buffered result."""

    metadata_ready = Signal(object)
    status = Signal(str)
    model_status = Signal(str, object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        camera_index: int,
        settings: Mapping[str, Any],
        buffer: LatestFrameBuffer,
        *,
        adapter_loader: Callable[[Path], InferenceAdapter] = UltralyticsInferenceAdapter.load,
        capture_factory: Callable[[int], Any] = _open_capture,
    ) -> None:
        super().__init__()
        self.camera_index = int(camera_index)
        self.settings = dict(settings)
        self.buffer = buffer
        self.adapter_loader = adapter_loader
        self.capture_factory = capture_factory
        self._stop_requested = threading.Event()
        self._model_lock = threading.Lock()
        self._settings_lock = threading.Lock()
        self._requested_model: ModelLoadRequest | None = None
        self._model_generation = 0
        initial_model = self.settings.get("model_path")
        if initial_model:
            self._requested_model = ModelLoadRequest(
                Path(str(initial_model)),
                str(self.settings.get("model_sha256")) if self.settings.get("model_sha256") else None,
                "saved camera profile model",
            )
            self._model_generation = 1
        self._adapter: InferenceAdapter | None = None

    def request_model_load(
        self,
        path: Path | None,
        *,
        expected_sha256: str | None = None,
        label: str = "model",
    ) -> None:
        with self._model_lock:
            self._requested_model = ModelLoadRequest(path, expected_sha256, label) if path is not None else None
            self._model_generation += 1

    def update_settings(self, settings: Mapping[str, Any]) -> None:
        with self._settings_lock:
            self.settings.update(dict(settings))

    def _settings_snapshot(self) -> dict[str, Any]:
        with self._settings_lock:
            return dict(self.settings)

    def stop(self) -> None:
        self._stop_requested.set()

    def _take_model_request(self) -> tuple[ModelLoadRequest | None, int]:
        with self._model_lock:
            return self._requested_model, self._model_generation

    @Slot()
    def run(self) -> None:
        capture = None
        owned = False
        try:
            if not CameraOwnership.acquire("vision"):
                owner = CameraOwnership.current_owner() or "another module"
                self.error.emit(f"Camera is already in use by {owner}.")
                return
            owned = True
            capture = self.capture_factory(self.camera_index)
            if capture is None or not capture.isOpened():
                self.error.emit(f"Could not open camera {self.camera_index}.")
                return
            initial_settings = self._settings_snapshot()
            requested_camera = dict(initial_settings.get("requested_camera", {}))
            readback_camera = apply_camera_properties(capture, requested_camera)
            backend = _capture_backend(capture)
            applied_camera = dict(requested_camera)
            self.metadata_ready.emit(
                _capture_metadata(
                    capture,
                    camera_index=self.camera_index,
                    backend=backend,
                    requested_camera=requested_camera,
                    readback_camera=readback_camera,
                )
            )
            self.status.emit("Camera live. Preview is active; inference is optional.")
            frame_id = 0
            loaded_generation = 0
            while not self._stop_requested.is_set():
                requested_model, generation = self._take_model_request()
                if generation != loaded_generation:
                    loaded_generation = generation
                    self._adapter = None
                    if requested_model is not None:
                        self._load_model(requested_model)
                ok, frame = capture.read()
                if not ok:
                    self.error.emit("Camera frame read failed.")
                    break
                frame_id += 1
                settings = self._settings_snapshot()
                requested_camera = dict(settings.get("requested_camera", {}))
                if requested_camera != applied_camera:
                    readback_camera = apply_camera_properties(capture, requested_camera)
                    applied_camera = dict(requested_camera)
                    self.metadata_ready.emit(
                        _capture_metadata(
                            capture,
                            camera_index=self.camera_index,
                            backend=backend,
                            requested_camera=requested_camera,
                            readback_camera=readback_camera,
                        )
                    )
                try:
                    result = process_frame(
                        frame,
                        frame_id=frame_id,
                        roi=settings.get("roi"),
                        brightness=float(settings.get("brightness", 0.0)),
                        contrast=float(settings.get("contrast", 1.0)),
                        confidence=float(settings.get("confidence", 0.5)),
                        adapter=self._adapter,
                    )
                except Exception as error:
                    self.error.emit(str(error))
                    self._adapter = None
                    result = process_frame(
                        frame,
                        frame_id=frame_id,
                        roi=settings.get("roi"),
                        brightness=float(settings.get("brightness", 0.0)),
                        contrast=float(settings.get("contrast", 1.0)),
                        confidence=float(settings.get("confidence", 0.5)),
                        adapter=None,
                    )
                self.buffer.publish(result)
                QThread.msleep(15)
        except Exception as error:
            self.error.emit(str(error))
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            if owned:
                CameraOwnership.release("vision")
            self.finished.emit()

    def _load_model(self, request: ModelLoadRequest) -> None:
        model_path = request.path.expanduser().resolve()
        self.status.emit(f"Loading local model: {model_path.name}")
        try:
            if model_path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
                raise ModelLoadError("Choose a local .pt or .onnx model file.")
            if not model_path.is_file():
                if request.expected_sha256:
                    raise ModelLoadError(f"{request.label} was moved or deleted: {model_path}")
                raise ModelLoadError(f"Model file does not exist: {model_path}")
            current_sha = model_sha256(model_path)
            if request.expected_sha256 and current_sha.lower() != request.expected_sha256.lower():
                raise ModelLoadError(
                    f"{request.label} changed since it was recorded. "
                    f"Expected SHA-256 {request.expected_sha256}; found {current_sha}."
                )
            adapter = self.adapter_loader(validate_model_path(model_path))
        except Exception as error:
            self._adapter = None
            self.model_status.emit("error", str(error))
            self.error.emit(str(error))
            return
        self._adapter = adapter
        self.model_status.emit(
            "loaded",
            {
                "device": str(getattr(adapter, "device", "unknown")),
                "path": str(model_path),
                "sha256": current_sha,
            },
        )
        self.status.emit(f"Model loaded. Inference device: {getattr(adapter, 'device', 'unknown')}")


def _capture_backend(capture: Any) -> str | None:
    try:
        return str(capture.getBackendName())
    except Exception:
        return None


def save_profile(workspace_root: Path, profile: VisionProfile) -> Path:
    root = workspace_root.resolve()
    path = (root / "camera" / "vision_profile.json").resolve()
    if root not in path.parents:
        raise VisionError("Vision profile path escapes the candidate workspace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_profile(workspace_root: Path) -> VisionProfile:
    root = workspace_root.resolve()
    path = (root / "camera" / "vision_profile.json").resolve()
    if root not in path.parents:
        raise VisionError("Vision profile path escapes the candidate workspace.")
    if not path.is_file():
        return VisionProfile()
    try:
        return VisionProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VisionError(f"Could not read vision profile: {error}") from error


def robot_coordinates_for_detection(
    record: CalibrationRecord,
    detection: Detection,
    full_frame_size: Sequence[int],
) -> tuple[float, float]:
    """Apply M2 only after geometry validation; class IDs remain model labels."""

    return CalibrationService.apply(record, detection.center_px, full_frame_size)


def save_snapshot(
    workspace_root: Path,
    result: FrameResult,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    root = workspace_root.resolve()
    directory = (root / "exports" / "vision").resolve()
    if root not in directory.parents:
        raise VisionError("Vision snapshot path escapes the candidate workspace.")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"snapshot-{stamp}-frame-{result.frame_id}"
    raw_path = directory / f"{stem}-raw.png"
    annotated_path = directory / f"{stem}-annotated.png"
    metadata_path = directory / f"{stem}.json"
    if not cv2.imwrite(str(raw_path), result.raw_bgr) or not cv2.imwrite(str(annotated_path), result.annotated_bgr):
        for path in (raw_path, annotated_path):
            path.unlink(missing_ok=True)
        raise VisionError("Could not write the vision snapshot images.")
    payload = dict(metadata)
    payload.update(
        {
            "schema_version": "m4.vision.snapshot.v1",
            "frame_id": result.frame_id,
            "captured_at": result.captured_at,
            "full_frame_size": list(result.full_frame_size),
            "roi": list(result.roi),
            "detections": [detection.to_dict() for detection in result.detections],
            "raw_image": raw_path.name,
            "annotated_image": annotated_path.name,
        }
    )
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw_path, annotated_path, metadata_path


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
