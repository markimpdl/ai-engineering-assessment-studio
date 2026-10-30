from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ai_assessment.services.calibration import (
    CalibrationError,
    CalibrationGeometryMismatchError,
    CalibrationService,
    fit_affine,
    fit_homography,
    transform_point,
)
from ai_assessment.core.workspace import CandidateWorkspace


class CalibrationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_points = ((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0))

    def test_known_affine_mapping(self) -> None:
        robot_points = tuple(
            (2.0 * u + 3.0 * v + 10.0, -u + 4.0 * v + 5.0)
            for u, v in self.camera_points
        )

        matrix = fit_affine(self.camera_points, robot_points)

        np.testing.assert_allclose(
            matrix,
            ((2.0, 3.0, 10.0), (-1.0, 4.0, 5.0)),
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            transform_point(matrix, 25.0, 30.0),
            (150.0, 100.0),
            rtol=0.0,
            atol=1e-10,
        )

    def test_affine_accepts_small_point_selection_error(self) -> None:
        camera = ((154, 250), (154, 105), (418, 251), (417, 105))
        robot = ((6, -256), (66, -256), (6, -146), (66, -146))

        matrix = fit_affine(camera, robot)
        errors = [
            np.linalg.norm(np.asarray(transform_point(matrix, *uv)) - xy)
            for uv, xy in zip(camera, robot)
        ]
        self.assertGreater(max(errors), 0.1)
        self.assertLess(max(errors), 0.2)

        inconsistent_robot = (*robot[:3], (200, -146))
        with self.assertRaisesRegex(CalibrationError, "Affine point fit error"):
            fit_affine(camera, inconsistent_robot)

    def test_known_homography_mapping(self) -> None:
        expected = np.array(
            [[1.2, 0.1, 4.0], [0.05, 0.9, -2.0], [0.0005, -0.0008, 1.0]]
        )
        robot_points = tuple(
            transform_point(expected, u, v) for u, v in self.camera_points
        )

        matrix = fit_homography(self.camera_points, robot_points)

        np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(
            transform_point(matrix, 40.0, 30.0),
            transform_point(expected, 40.0, 30.0),
            rtol=0.0,
            atol=1e-8,
        )

    def test_degenerate_points_are_rejected(self) -> None:
        collinear = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0))

        with self.assertRaises(CalibrationError):
            fit_affine(collinear, self.camera_points)
        with self.assertRaises(CalibrationError):
            fit_homography(self.camera_points, collinear)

    def test_save_load_preserves_full_precision(self) -> None:
        robot_points = tuple(
            (2.0 * u + 3.0 * v + 10.0, -u + 4.0 * v + 5.0)
            for u, v in self.camera_points
        )
        record = CalibrationService.fit(
            "affine",
            self.camera_points,
            robot_points,
            (1920, 1080),
            camera_metadata={"source_type": "saved_image", "camera_index": None},
            roi=None,
            snapshot_png=b"png-placeholder",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            CalibrationService.save(path, record)
            loaded = CalibrationService.load(path)
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded.schema_version, 3)
        self.assertEqual(loaded.source_schema_version, 3)
        self.assertEqual(loaded.snapshot_png, b"png-placeholder")
        self.assertEqual(loaded.camera_metadata["source_type"], "saved_image")
        np.testing.assert_array_equal(np.asarray(loaded.matrix), np.asarray(record.matrix))
        self.assertEqual(raw["matrix"], [list(row) for row in record.matrix])

    def test_schema_one_and_two_legacy_files_load(self) -> None:
        homography = fit_homography(
            self.camera_points,
            tuple((u + 5.0, v + 7.0) for u, v in self.camera_points),
        )
        affine = fit_affine(
            self.camera_points,
            tuple((2.0 * u + 1.0, 3.0 * v + 2.0) for u, v in self.camera_points),
        )
        snapshot = base64.b64encode(b"png-placeholder").decode("ascii")
        common = {
            "coordinate_convention": "pixel origin top-left",
            "image_width": 640,
            "image_height": 480,
            "camera_points_uv": [list(point) for point in self.camera_points],
            "snapshot_png_base64": snapshot,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            schema_one = root / "legacy-v1.json"
            schema_one.write_text(
                json.dumps(
                    {
                        **common,
                        "schema_version": 1,
                        "robot_points_xy": [
                            [u + 5.0, v + 7.0] for u, v in self.camera_points
                        ],
                        "matrix_3x3": homography.tolist(),
                    }
                ),
                encoding="utf-8",
            )
            schema_two = root / "legacy-v2.json"
            schema_two.write_text(
                json.dumps(
                    {
                        **common,
                        "schema_version": 2,
                        "transform": "affine",
                        "robot_points_xy": [
                            [2.0 * u + 1.0, 3.0 * v + 2.0]
                            for u, v in self.camera_points
                        ],
                        "matrix": affine.tolist(),
                    }
                ),
                encoding="utf-8",
            )

            loaded_one = CalibrationService.load(schema_one)
            loaded_two = CalibrationService.load(schema_two)

        self.assertEqual(loaded_one.source_schema_version, 1)
        self.assertEqual(loaded_one.transform, "homography")
        self.assertEqual(loaded_two.source_schema_version, 2)
        self.assertEqual(loaded_two.transform, "affine")

    def test_geometry_mismatch_blocks_application(self) -> None:
        robot_points = tuple((u + 1.0, v + 2.0) for u, v in self.camera_points)
        record = CalibrationService.fit(
            "affine", self.camera_points, robot_points, (640, 480)
        )

        with self.assertRaises(CalibrationGeometryMismatchError):
            CalibrationService.apply(record, (20.0, 30.0), (1280, 720))
        np.testing.assert_allclose(
            CalibrationService.apply(record, (20.0, 30.0), (640, 480)),
            (21.0, 32.0),
            rtol=0.0,
            atol=1e-10,
        )

    def test_calibration_page_preserves_fit_across_live_resume_and_frame_change(self) -> None:
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtWidgets import QApplication

        from ai_assessment.modules.calibration import CalibrationPage

        app = QApplication.instance() or QApplication([])
        del app
        with tempfile.TemporaryDirectory() as directory:
            workspace = CandidateWorkspace.create(Path(directory) / "C014", candidate_id="C014")
            page = CalibrationPage(workspace)
            try:
                image = QImage(100, 80, QImage.Format.Format_RGB32)
                image.fill(QColor("white"))
                page._source_mode = "camera"
                page._frozen = True
                page._camera_metadata = {"source_type": "camera", "camera_index": 0}
                page._set_current_image(image)
                for index, (u, v) in enumerate(((10, 10), (70, 10), (70, 60), (10, 60))):
                    page._select_row(index)
                    page._image_clicked(u, v)
                    page._robot_fields[index][0].setText(str(u + 5))
                    page._robot_fields[index][1].setText(str(v + 7))
                page._calculate()
                record = page._record
                points = page._points[:]
                self.assertIsNotNone(record)
                page._toggle_freeze()
                self.assertEqual(page._record, record)
                self.assertEqual(page._points, points)
                page._camera_metadata_ready({"source_type": "camera", "camera_index": 0})
                page._camera_frame_ready(np.zeros((80, 100, 3), dtype=np.uint8))
                self.assertEqual(page._record, record)
                page._toggle_freeze()
                page._image_clicked(30, 30)
                self.assertIn("ROBOT X: 35", page.test_result.text())
                self.assertEqual(page._points, points)
                page._toggle_freeze()
                page._camera_frame_ready(np.zeros((100, 120, 3), dtype=np.uint8))
                self.assertEqual(page._record, record)
                self.assertEqual(page._points, points)
                self.assertFalse(page.geometry_warning.isHidden())
                page._camera_metadata_ready({"source_type": "camera", "camera_index": 1})
                page._save_calibration()
                saved = CalibrationService.load(page._calibration_path())
                self.assertEqual(saved.camera_metadata["camera_index"], 0)
                self.assertEqual(saved.snapshot_png, record.snapshot_png)
                page._clear_points()
                self.assertIsNone(page._record)
                self.assertEqual(page._points, [None] * 4)
                self.assertEqual(page.matrix_text.toPlainText(), "")
            finally:
                page.shutdown()

    def test_robot_coordinates_are_visible_editable_numeric_text(self) -> None:
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication

        from ai_assessment.modules.calibration import CalibrationPage

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            workspace = CandidateWorkspace.create(Path(directory) / "C014", candidate_id="C014")
            page = CalibrationPage(workspace)
            try:
                field = page._robot_fields[0][0]
                self.assertEqual(field.text(), "0.0")
                self.assertTrue(field.isEnabled())
                self.assertFalse(field.isReadOnly())
                field.selectAll()
                QTest.keyClicks(field, "-195.5")
                self.assertEqual(field.text(), "-195.5")
                field.selectAll()
                QTest.keyClicks(field, "38")
                self.assertEqual(field.text(), "38")
                page._clear_points()
                self.assertEqual(field.text(), "0.0")
            finally:
                page.shutdown()

    def test_rtsp_worker_reconnects_without_saving_or_reporting_credentials(self) -> None:
        from ai_assessment.services.camera import CameraWorker, _open_rtsp_capture, validate_rtsp_url
        import cv2

        url = "rtsp://viewer:private@127.0.0.1:8554/camera"
        self.assertEqual(validate_rtsp_url(url), url)
        for invalid in ("", "http://127.0.0.1/stream", "rtsp://", "rtsp://host:bad/camera"):
            with self.assertRaises(ValueError):
                validate_rtsp_url(invalid)
        with patch("ai_assessment.services.camera.cv2.VideoCapture") as open_cv:
            _open_rtsp_capture(url)
        open_cv.assert_called_once_with(
            url,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )

        failed = Mock()
        failed.isOpened.return_value = False
        live = Mock()
        live.isOpened.return_value = True
        live.getBackendName.return_value = "FFMPEG"
        live.get.return_value = 30.0
        live.read.return_value = (True, np.zeros((80, 100, 3), dtype=np.uint8))
        worker = CameraWorker(0, source_type="rtsp", rtsp_url=url)
        errors = []
        metadata = []
        frames = []
        worker.error.connect(errors.append)
        worker.metadata_ready.connect(metadata.append)
        worker.frame_ready.connect(lambda frame: (frames.append(frame), worker.stop()))
        with patch("ai_assessment.services.camera._open_rtsp_capture", side_effect=[failed, live]) as open_capture:
            with patch.object(worker.stop_requested, "wait", return_value=False):
                worker.run()

        self.assertEqual(open_capture.call_count, 2)
        self.assertEqual(len(frames), 1)
        self.assertEqual(metadata[0]["source_type"], "rtsp")
        self.assertEqual(len(metadata[0]["source_id"]), 64)
        self.assertNotIn("private", str(metadata))
        self.assertNotIn("private", str(errors))
        failed.release.assert_called_once()
        live.release.assert_called_once()

    def test_rtsp_calibration_freezes_and_blocks_another_stream(self) -> None:
        from PySide6.QtWidgets import QApplication

        from ai_assessment.modules.calibration import CalibrationPage

        app = QApplication.instance() or QApplication([])
        del app
        with tempfile.TemporaryDirectory() as directory:
            workspace = CandidateWorkspace.create(Path(directory) / "C014", candidate_id="C014")
            page = CalibrationPage(workspace)
            try:
                page.source_choice.setCurrentIndex(1)
                self.assertFalse(page.rtsp_row.isHidden())
                page._source_mode = "rtsp"
                page._camera_metadata_ready({"source_type": "rtsp", "source_id": "first"})
                page._camera_frame_ready(np.zeros((80, 100, 3), dtype=np.uint8))
                self.assertTrue(page.freeze_button.isEnabled())
                page._toggle_freeze()
                self.assertTrue(page._frozen)
                points = ((10, 10), (70, 10), (70, 60), (10, 60))
                robot = tuple((u + 5, v + 7) for u, v in points)
                page._record = CalibrationService.fit(
                    "affine", points, robot, (100, 80), camera_metadata=page._camera_metadata
                )
                page._edit_mode = False
                page.send_robot_on_click.setChecked(True)
                page._send_robot_command = Mock()
                page._image_clicked(30, 30)
                page._send_robot_command.assert_called_once_with(35, 37)
                page._send_robot_command.reset_mock()
                page._camera_metadata_ready({"source_type": "rtsp", "source_id": "second"})
                page._image_clicked(30, 30)
                page._send_robot_command.assert_not_called()
                self.assertIn("RTSP stream differs", page.status.text())
                record = page._record
                page.source_choice.setCurrentIndex(0)
                self.assertEqual(page._record, record)
                self.assertTrue(page._current_image.isNull())
                self.assertFalse(page.freeze_button.isEnabled())
            finally:
                page.shutdown()

    def test_robot_click_requires_selector_matching_source_and_geometry(self) -> None:
        from PySide6.QtGui import QColor, QImage
        from PySide6.QtWidgets import QApplication

        from ai_assessment.modules.calibration import CalibrationPage, _robot_base_url

        self.assertEqual(_robot_base_url("http://127.0.0.1:8080/command"), "http://127.0.0.1:8080")
        with self.assertRaises(ValueError):
            _robot_base_url("http://alice:secret@127.0.0.1:8080")
        app = QApplication.instance() or QApplication([])
        del app
        with tempfile.TemporaryDirectory() as directory:
            workspace = CandidateWorkspace.create(Path(directory) / "C014", candidate_id="C014")
            page = CalibrationPage(workspace)
            try:
                image = QImage(100, 80, QImage.Format.Format_RGB32)
                image.fill(QColor("white"))
                page._source_mode = "camera"
                page._frozen = True
                page._camera_metadata = {"source_type": "camera", "camera_index": 0}
                page._set_current_image(image)
                points = ((10, 10), (70, 10), (70, 60), (10, 60))
                robot = tuple((u + 5, v + 7) for u, v in points)
                page._record = CalibrationService.fit("affine", points, robot, (100, 80), camera_metadata=page._camera_metadata)
                page._edit_mode = False
                page._send_robot_command = Mock()
                page._image_clicked(30, 30)
                page._send_robot_command.assert_not_called()
                page.send_robot_on_click.setChecked(True)
                page._image_clicked(30, 30)
                page._send_robot_command.assert_called_once_with(35, 37)
                page._send_robot_command.reset_mock()
                page._camera_metadata = {"source_type": "camera", "camera_index": 1}
                page._image_clicked(30, 30)
                page._send_robot_command.assert_not_called()
                page._camera_metadata = {"source_type": "camera", "camera_index": 0}
                page._set_current_image(QImage(120, 100, QImage.Format.Format_RGB32))
                page._image_clicked(30, 30)
                page._send_robot_command.assert_not_called()
            finally:
                page.shutdown()

    def test_robot_worker_posts_calibrated_xy_and_fixed_z(self) -> None:
        from ai_assessment.modules.calibration import CalibrationMoveWorker

        worker = CalibrationMoveWorker("http://127.0.0.1:8080", 35, 37)
        adapter = Mock()
        adapter.move_xyz.return_value.status_code = 200
        with patch("ai_assessment.modules.calibration.RequestsRobotAdapter", return_value=adapter) as create:
            worker.run()
        self.assertEqual(create.call_args.kwargs["capabilities"].cartesian_move, True)
        adapter.move_xyz.assert_called_once_with(35, 37, 100, 2000)


if __name__ == "__main__":
    unittest.main()

