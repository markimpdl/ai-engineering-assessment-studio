"""Threaded OpenCV capture and image loading for the calibration page."""

from __future__ import annotations

import threading
import sys
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

import cv2
from PySide6.QtCore import QObject, Signal, Slot, QThread


class CameraOwnership:
    """Process-local ownership guard for the single shared webcam."""

    _lock = threading.Lock()
    _owner: str | None = None

    @classmethod
    def acquire(cls, owner: str) -> bool:
        with cls._lock:
            if cls._owner is not None and cls._owner != owner:
                return False
            cls._owner = owner
            return True

    @classmethod
    def release(cls, owner: str) -> None:
        with cls._lock:
            if cls._owner == owner:
                cls._owner = None

    @classmethod
    def current_owner(cls) -> str | None:
        with cls._lock:
            return cls._owner


def validate_rtsp_url(value: str) -> str:
    """Validate a stream address without connecting to it or exposing credentials."""
    url = value.strip()
    if not url or any(ord(character) < 32 for character in url):
        raise ValueError("Enter an RTSP URL such as rtsp://host:8554/camera.")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port
    except ValueError as error:
        raise ValueError("Enter a valid RTSP URL with a host and optional port.") from error
    if parts.scheme.lower() not in ("rtsp", "rtsps") or not host or parts.fragment:
        raise ValueError("Enter a valid rtsp:// or rtsps:// URL with a host.")
    return url


def _open_rtsp_capture(url: str):
    """Bound network open/read waits so stopping a stream can finish promptly."""
    return cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
    )


class CameraWorker(QObject):
    frame_ready = Signal(object)
    metadata_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self, camera_index: int, *, source_type: str = "camera", rtsp_url: str | None = None
    ) -> None:
        super().__init__()
        if source_type not in ("camera", "rtsp"):
            raise ValueError(f"Unsupported calibration source: {source_type!r}")
        self.camera_index = camera_index
        self.source_type = source_type
        self.rtsp_url = validate_rtsp_url(rtsp_url or "") if source_type == "rtsp" else None
        self.stop_requested = threading.Event()

    @Slot()
    def run(self) -> None:
        capture = None
        try:
            if self.source_type == "rtsp":
                self._run_rtsp()
                return
            if sys.platform == "win32":
                capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened() and sys.platform == "win32":
                capture.release()
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                self.error.emit(f"Could not open camera {self.camera_index}.")
                return
            self.metadata_ready.emit(
                {
                    "source_type": "camera",
                    "camera_index": self.camera_index,
                    "backend": capture.getBackendName(),
                    "readback_width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    "readback_height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    "readback_fps": capture.get(cv2.CAP_PROP_FPS),
                }
            )
            while not self.stop_requested.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.error.emit("Camera frame read failed.")
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb)
                QThread.msleep(33)
        except Exception as error:  # OpenCV backends can raise platform-specific errors.
            self.error.emit(
                "RTSP stream failed. Check the connection and try again."
                if self.source_type == "rtsp" else str(error)
            )
        finally:
            if capture is not None:
                capture.release()
            self.finished.emit()

    def _run_rtsp(self) -> None:
        assert self.rtsp_url is not None
        source_id = sha256(self.rtsp_url.encode("utf-8")).hexdigest()
        while not self.stop_requested.is_set():
            capture = None
            try:
                capture = _open_rtsp_capture(self.rtsp_url)
                if not capture.isOpened():
                    self.error.emit("Could not open RTSP stream. Retrying...")
                else:
                    self.metadata_ready.emit(
                        {
                            "source_type": "rtsp",
                            "source_id": source_id,
                            "backend": capture.getBackendName(),
                            "readback_width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                            "readback_height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            "readback_fps": capture.get(cv2.CAP_PROP_FPS),
                        }
                    )
                    while not self.stop_requested.is_set():
                        ok, frame = capture.read()
                        if not ok:
                            if not self.stop_requested.is_set():
                                self.error.emit("RTSP frame read failed. Reconnecting...")
                            break
                        self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        QThread.msleep(33)
            except Exception:  # Backend errors can include the URL or credentials.
                if not self.stop_requested.is_set():
                    self.error.emit("RTSP stream failed. Reconnecting...")
            finally:
                if capture is not None:
                    capture.release()
            self.stop_requested.wait(1.0)

    def stop(self) -> None:
        self.stop_requested.set()


class ImageLoadWorker(QObject):
    image_ready = Signal(object, object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    @Slot()
    def run(self) -> None:
        try:
            image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if image is None:
                self.error.emit(f"Could not read image: {self.path.name}")
                return
            self.image_ready.emit(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                {"source_type": "saved_image", "path": str(self.path)},
            )
        except Exception as error:
            self.error.emit(str(error))
        finally:
            self.finished.emit()
