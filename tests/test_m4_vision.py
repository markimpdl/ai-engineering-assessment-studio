from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ai_assessment.core.workspace import CandidateWorkspace
from ai_assessment.services.calibration import CalibrationGeometryMismatchError, CalibrationService
from ai_assessment.services.camera import CameraOwnership
from ai_assessment.services.results import ResultsService
from ai_assessment.services.training import TrainingError, sha256_file
from ai_assessment.services.vision import (
    Detection,
    LatestFrameBuffer,
    ModelLoadError,
    RawDetection,
    VisionProfile,
    VisionWorker,
    apply_camera_properties,
    apply_software_correction,
    map_bbox_to_full_frame,
    process_frame,
    robot_coordinates_for_detection,
    save_profile,
    load_profile,
    validate_model_path,
)


class FakeAdapter:
    class_names = {2: "Ajwa"}
    device = "cpu"

    def __init__(self, bbox: tuple[float, float, float, float] = (5.0, 6.0, 15.0, 16.0)) -> None:
        self.bbox = bbox
        self.seen: np.ndarray | None = None

    def infer(self, image_bgr: np.ndarray, confidence: float) -> tuple[RawDetection, ...]:
        del confidence
        self.seen = image_bgr.copy()
        return (RawDetection(2, 0.91, self.bbox, model_size=(image_bgr.shape[1], image_bgr.shape[0])),)


class FakeCapture:
    def __init__(self, opened: bool, frames: list[np.ndarray] | None = None) -> None:
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def get(self, prop: int) -> float:
        del prop
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        del prop, value
        return False

    def getBackendName(self) -> str:
        return "fake"

    def release(self) -> None:
        self.released = True


class RecordingCapture(FakeCapture):
    def __init__(self, frames: list[np.ndarray], on_read: object = None) -> None:
        super().__init__(True, frames)
        self.on_read = on_read
        self.read_count = 0
        self.set_values: list[tuple[int, float]] = []
        self.properties: dict[int, float] = {}

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if self.read_count == 1 and callable(self.on_read):
            self.on_read()
        return super().read()

    def get(self, prop: int) -> float:
        return self.properties.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = float(value)
        self.set_values.append((prop, float(value)))
        return True


class VisionServiceTests(unittest.TestCase):
    def test_camera_ownership_prevents_calibration_and_vision_competing(self) -> None:
        CameraOwnership.release("calibration")
        CameraOwnership.release("vision")
        self.assertTrue(CameraOwnership.acquire("calibration"))
        self.assertFalse(CameraOwnership.acquire("vision"))
        self.assertEqual(CameraOwnership.current_owner(), "calibration")
        CameraOwnership.release("calibration")
        self.assertTrue(CameraOwnership.acquire("vision"))
        CameraOwnership.release("vision")

    def test_resize_and_padding_are_reversed_before_roi_offset(self) -> None:
        # A 200x100 ROI letterboxed into 400x400 has scale 2 and 100px vertical padding.
        mapped = map_bbox_to_full_frame(
            (40.0, 160.0, 120.0, 240.0),
            (100, 50, 200, 100),
            (200, 100),
            (400, 400),
        )
        self.assertEqual(mapped, (120.0, 80.0, 160.0, 120.0))

    def test_correction_order_is_contrast_then_brightness(self) -> None:
        frame = np.full((1, 1, 3), 100, dtype=np.uint8)
        corrected = apply_software_correction(frame, brightness=10, contrast=2)
        self.assertEqual(int(corrected[0, 0, 0]), 210)

    def test_detection_structure_reports_full_frame_coordinates(self) -> None:
        adapter = FakeAdapter()
        result = process_frame(
            np.zeros((80, 100, 3), dtype=np.uint8),
            frame_id=7,
            roi=(10, 20, 50, 40),
            brightness=0,
            contrast=1,
            confidence=0.5,
            adapter=adapter,
            captured_at="2026-09-27T00:00:00Z",
        )
        detection = result.detections[0]
        self.assertEqual(detection.class_id, 2)
        self.assertEqual(detection.class_name, "Ajwa")
        self.assertEqual(detection.frame_id, 7)
        self.assertEqual(detection.coordinate_space, "full_frame")
        self.assertEqual(detection.bbox_xyxy, (15.0, 26.0, 25.0, 36.0))
        self.assertEqual(detection.center_px, (20.0, 31.0))
        self.assertEqual(adapter.seen.shape[:2], (40, 50))

    def test_latest_frame_buffer_drops_stale_results(self) -> None:
        buffer = LatestFrameBuffer()
        for frame_id in (1, 2, 3):
            result = process_frame(
                np.zeros((4, 4, 3), dtype=np.uint8),
                frame_id=frame_id,
                roi=None,
                brightness=0,
                contrast=1,
                confidence=0.5,
                adapter=None,
            )
            buffer.publish(result)
        self.assertEqual(buffer.take_latest().frame_id, 3)
        self.assertIsNone(buffer.take_latest())

    def test_geometry_mismatch_is_not_converted_to_robot_coordinates(self) -> None:
        camera_points = ((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0))
        robot_points = tuple((u + 1.0, v + 2.0) for u, v in camera_points)
        record = CalibrationService.fit("affine", camera_points, robot_points, (640, 480))
        detection = Detection(0, "Ajwa", 0.9, (10.0, 20.0, 30.0, 40.0), (20.0, 30.0), 1, "now")
        with self.assertRaises(CalibrationGeometryMismatchError):
            robot_coordinates_for_detection(record, detection, (1280, 720))

    def test_profile_roundtrip_preserves_model_and_processing_settings(self) -> None:
        profile = VisionProfile(
            camera_index=2,
            backend="fake",
            requested_camera={"exposure": -6.0},
            readback_camera={"exposure": {"actual": -5.0}},
            frame_size=(1280, 720),
            roi={"x": 10, "y": 20, "width": 300, "height": 200},
            confidence=0.73,
            brightness=12,
            contrast=1.4,
            model_path="C:/models/date.pt",
            model_sha256="abc",
            model_device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_profile(Path(directory), profile)
            loaded = load_profile(Path(directory))
        self.assertEqual(path.name, "vision_profile.json")
        self.assertEqual(loaded.to_dict(), profile.to_dict())

    def test_camera_property_readback_is_json_serializable(self) -> None:
        readback = apply_camera_properties(FakeCapture(True), {"exposure": -6.0})
        json.dumps(readback)
        self.assertIs(type(readback["exposure"]["supported"]), bool)

    def test_unsupported_camera_properties_are_reported_as_unsupported(self) -> None:
        # AVFoundation-style backend: set() returns False and get() returns 0.
        readback = apply_camera_properties(FakeCapture(True), {"auto_exposure": 1.0, "exposure": -6.0})
        self.assertFalse(readback["auto_exposure"]["supported"])
        self.assertFalse(readback["exposure"]["supported"])
        self.assertEqual(readback["exposure"]["mode"], "auto")

    def test_auto_switch_is_written_first_and_manual_value_is_skipped_while_auto(self) -> None:
        import cv2

        capture = RecordingCapture([])
        readback = apply_camera_properties(
            capture,
            {"exposure": -6.0, "auto_exposure": 1.0, "focus": 25.0, "autofocus": 0.0},
        )
        written = [prop for prop, _value in capture.set_values]
        self.assertEqual(written, [cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_AUTOFOCUS, cv2.CAP_PROP_FOCUS])
        self.assertNotIn(cv2.CAP_PROP_EXPOSURE, written)
        self.assertEqual(readback["auto_exposure"]["mode"], "auto")
        self.assertEqual(readback["exposure"]["mode"], "auto")
        self.assertTrue(readback["exposure"]["supported"])
        self.assertEqual(readback["focus"]["mode"], "manual")
        self.assertEqual(readback["focus"]["actual"], 25.0)

        # Switching auto off writes the manual value again, after the switch.
        capture.set_values.clear()
        apply_camera_properties(capture, {"auto_exposure": 0.0, "exposure": -6.0})
        self.assertEqual(
            [prop for prop, _value in capture.set_values],
            [cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_EXPOSURE],
        )
        self.assertEqual(capture.properties[cv2.CAP_PROP_EXPOSURE], -6.0)

    def test_camera_property_changes_apply_while_worker_is_running(self) -> None:
        CameraOwnership.release("vision")
        capture = RecordingCapture(
            [
                np.zeros((20, 20, 3), dtype=np.uint8),
                np.zeros((20, 20, 3), dtype=np.uint8),
            ]
        )
        worker = VisionWorker(
            0,
            {"requested_camera": {"exposure": -6.0}},
            LatestFrameBuffer(),
            capture_factory=lambda index: capture,
        )

        def update_exposure() -> None:
            worker.update_settings({"requested_camera": {"exposure": -7.0}})
            worker.stop()

        capture.on_read = update_exposure
        worker.run()

        applied_values = [value for _prop, value in capture.set_values]
        self.assertIn(-6.0, applied_values)
        self.assertIn(-7.0, applied_values)
        self.assertTrue(capture.released)

    def test_camera_and_model_failures_are_reported_and_camera_released(self) -> None:
        camera_errors: list[str] = []
        unopened = FakeCapture(False)
        camera_worker = VisionWorker(0, {}, LatestFrameBuffer(), capture_factory=lambda index: unopened)
        camera_worker.error.connect(camera_errors.append)
        camera_worker.run()
        self.assertTrue(any("Could not open camera" in message for message in camera_errors))
        self.assertTrue(unopened.released)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "date.pt"
            model_path.write_bytes(b"fake")
            model_errors: list[str] = []
            capture = FakeCapture(True, [np.zeros((20, 20, 3), dtype=np.uint8)])

            def failing_loader(path: Path) -> FakeAdapter:
                raise ModelLoadError(f"bad model: {path.name}")

            worker = VisionWorker(
                0,
                {"model_path": str(model_path)},
                LatestFrameBuffer(),
                adapter_loader=failing_loader,
                capture_factory=lambda index: capture,
            )
            worker.error.connect(model_errors.append)
            worker.run()
            self.assertTrue(any("bad model" in message for message in model_errors))
            self.assertTrue(capture.released)

    def test_camera_preview_runs_without_a_model(self) -> None:
        capture = FakeCapture(True, [np.zeros((20, 30, 3), dtype=np.uint8)])
        buffer = LatestFrameBuffer()
        errors: list[str] = []
        worker = VisionWorker(0, {}, buffer, capture_factory=lambda index: capture)
        worker.error.connect(errors.append)
        worker.run()
        result = buffer.take_latest()
        self.assertIsNotNone(result)
        self.assertEqual(result.device, "preview only")
        self.assertEqual(result.detections, ())
        self.assertTrue(capture.released)

    def test_model_reference_rejects_missing_or_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ModelLoadError):
                validate_model_path(root / "missing.pt")
            bad = root / "weights.txt"
            bad.write_text("x", encoding="utf-8")
            with self.assertRaises(ModelLoadError):
                validate_model_path(bad)

    def _workspace_with_completed_run(self, root: Path, *, run_id: str = "run_01_done") -> tuple[CandidateWorkspace, Path]:
        workspace = CandidateWorkspace.create(root / "C014", candidate_id="C014")
        checkpoint = workspace.root / "models" / run_id / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"trained checkpoint")
        result_dir = workspace.root / "results" / "training" / run_id
        result_dir.mkdir(parents=True)
        record = {
            "run_id": run_id,
            "status": "completed",
            "started_at": "2026-10-02T10:00:00Z",
            "paths": {
                "model_dir": f"models/{run_id}",
                "result_dir": f"results/training/{run_id}",
            },
            "checkpoints": [
                {
                    "path": f"models/{run_id}/weights/best.pt",
                    "sha256": sha256_file(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                }
            ],
        }
        (result_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
        return workspace, checkpoint

    def test_completed_run_discovery_finds_verified_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace, checkpoint = self._workspace_with_completed_run(Path(directory))
            references = ResultsService(workspace).completed_best_checkpoints()
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].run_id, "run_01_done")
        self.assertEqual(references[0].path, checkpoint)
        self.assertEqual(references[0].status, "available")

    def test_selected_model_resolution_verifies_recorded_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace, checkpoint = self._workspace_with_completed_run(Path(directory))
            service = ResultsService(workspace)
            service.select_model("run_01_done")
            selected = service.selected_checkpoint()
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.path, checkpoint)
        self.assertTrue(selected.is_available)

    def test_recorded_checkpoint_missing_and_hash_mismatch_are_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace, checkpoint = self._workspace_with_completed_run(Path(directory))
            service = ResultsService(workspace)
            checkpoint.unlink()
            missing = service.completed_best_checkpoints()[0]
            self.assertEqual(missing.status, "missing")
            self.assertIn("moved or deleted", missing.message)
            with self.assertRaises(TrainingError):
                service.verify_checkpoint(missing)

        with tempfile.TemporaryDirectory() as directory:
            workspace, checkpoint = self._workspace_with_completed_run(Path(directory))
            checkpoint.write_bytes(b"tampered checkpoint")
            changed = ResultsService(workspace).completed_best_checkpoints()[0]
            self.assertEqual(changed.status, "changed")
            self.assertIn("SHA-256", changed.message)

    def test_base_weights_are_not_discovered_as_trained_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = CandidateWorkspace.create(Path(directory) / "C014", candidate_id="C014")
            base = workspace.root / "models" / "base_weights" / "yolo26n.pt"
            base.parent.mkdir(parents=True)
            base.write_bytes(b"base weights")
            result_dir = workspace.root / "results" / "training" / "run_bad"
            result_dir.mkdir(parents=True)
            (result_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_bad",
                        "status": "completed",
                        "started_at": "2026-10-02T10:00:00Z",
                        "checkpoints": [
                            {
                                "path": "models/base_weights/yolo26n.pt",
                                "sha256": sha256_file(base),
                                "bytes": base.stat().st_size,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(ResultsService(workspace).completed_best_checkpoints(), [])

    def test_vision_keeps_stale_profile_until_user_uses_results_model(self) -> None:
        from PySide6.QtWidgets import QApplication

        from ai_assessment.modules.vision import VisionPage

        app = QApplication.instance() or QApplication([])
        del app
        with tempfile.TemporaryDirectory() as directory:
            workspace, checkpoint = self._workspace_with_completed_run(Path(directory))
            old = workspace.root / "models" / "old_manual.pt"
            old.write_bytes(b"old manual model")
            save_profile(
                workspace.root,
                VisionProfile(model_path=str(old), model_sha256=sha256_file(old)),
            )
            ResultsService(workspace).select_model("run_01_done")
            page = VisionPage(workspace)
            try:
                self.assertEqual(Path(page.model_path.text()), old)
                self.assertIn("run_01_done", page.selected_model_label.text())
                page._use_selected_model()
                self.assertEqual(Path(page.model_path.text()), checkpoint)
                self.assertEqual(page._current_model_expected_sha, sha256_file(checkpoint))
            finally:
                page.shutdown()

    def test_camera_readback_text_names_auto_and_unsupported_properties(self) -> None:
        from ai_assessment.modules.vision import VisionPage

        requested = {"exposure": -6.0, "focus": 25.0, "white_balance": 4500.0}
        readback = {
            "auto_exposure": {"supported": True, "mode": "auto"},
            "exposure": {"actual": -4.0, "supported": True, "mode": "auto"},
            "autofocus": {"supported": False, "mode": "auto"},
            "focus": {"actual": None, "supported": False, "mode": "auto"},
            "auto_white_balance": {"supported": False, "mode": "manual"},
            "white_balance": {"actual": 0.0, "supported": False, "mode": "manual"},
        }
        text = VisionPage._describe_camera_readback(requested, readback)
        self.assertEqual(
            text,
            "exposure: auto (driver -4) | focus: auto (not supported) | white_balance: 4500 -> 0 (not supported)",
        )

    def test_camera_controls_default_to_manual_and_focus_slider_disables_auto(self) -> None:
        from PySide6.QtWidgets import QApplication

        from ai_assessment.modules.vision import VisionPage

        app = QApplication.instance() or QApplication([])
        del app
        with tempfile.TemporaryDirectory() as directory:
            workspace = CandidateWorkspace.create(Path(directory) / "C014", candidate_id="C014")
            page = VisionPage(workspace)
            try:
                self.assertFalse(page.auto_exposure.isChecked())
                self.assertFalse(page.auto_focus.isChecked())
                self.assertFalse(page.auto_white_balance.isChecked())

                page.auto_focus.setChecked(True)
                page.focus.setValue(25)

                self.assertFalse(page.auto_focus.isChecked())
                self.assertEqual(page._settings()["requested_camera"]["autofocus"], 0.0)
                self.assertEqual(page._settings()["requested_camera"]["focus"], 25.0)
            finally:
                page.shutdown()


if __name__ == "__main__":
    unittest.main()
