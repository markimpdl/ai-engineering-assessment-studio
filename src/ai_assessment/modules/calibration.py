"""M2 calibration workspace embedded in the main PySide6 shell."""

from __future__ import annotations

import binascii
from math import hypot
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PySide6.QtCore import QPoint, QRectF, Qt, QBuffer, QIODevice, QObject, QThread, Signal, Slot, QLocale
from PySide6.QtGui import QColor, QDoubleValidator, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.workspace import CandidateWorkspace
from ..services.calibration import (
    CalibrationError,
    CalibrationGeometryMismatchError,
    CalibrationRecord,
    CalibrationService,
    TransformType,
)
from ..services.camera import CameraOwnership, CameraWorker, ImageLoadWorker, validate_rtsp_url
from ..services.robot import RequestsRobotAdapter, RobotCapabilities, RobotError, RobotTimeoutError


def _qimage_from_rgb(rgb: Any) -> QImage:
    image = QImage(
        rgb.data,
        int(rgb.shape[1]),
        int(rgb.shape[0]),
        int(rgb.strides[0]),
        QImage.Format.Format_RGB888,
    )
    return image.copy()


def _format_matrix(matrix: tuple[tuple[float, ...], ...]) -> str:
    values = [
        ["0.00" if abs(value) < 0.005 else f"{value:.2f}" for value in row]
        for row in matrix
    ]
    widths = [max(len(row[column]) for row in values) for column in range(3)]
    if len(values) == 2:
        left, right = ("[", "["), ("]", "]")
    else:
        left, right = ("[", "|", "["), ("]", "|", "]")
    return "\n".join(
        f"{left[index]}  {'   '.join(value.rjust(widths[column]) for column, value in enumerate(row))}  {right[index]}"
        for index, row in enumerate(values)
    )


class CalibrationImageView(QWidget):
    clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("calibrationImageView")
        self.setMinimumSize(440, 320)
        self.image = QImage()
        self.points: list[tuple[float, float] | None] = [None] * 4
        self.test_point: tuple[float, float] | None = None
        self.hover_pixel: tuple[int, int] | None = None
        self.hover_widget_x = 0.0
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image: QImage) -> None:
        if image.isNull() or (
            not self.image.isNull() and self.image.size() != image.size()
        ):
            self.hover_pixel = None
        self.image = image
        self.update()

    def set_points(self, points: list[tuple[float, float] | None]) -> None:
        self.points = points[:]
        self.update()

    def image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        ratio = min(
            self.width() / self.image.width(), self.height() / self.image.height()
        )
        width = self.image.width() * ratio
        height = self.image.height() * ratio
        return QRectF(
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
        )

    def pixel_at(self, position: Any) -> tuple[float, float] | None:
        rectangle = self.image_rect()
        if rectangle.isNull() or not rectangle.contains(position):
            return None
        u = max(
            0.0,
            min(
                self.image.width() - 1.0,
                (position.x() - rectangle.x())
                * self.image.width()
                / rectangle.width(),
            ),
        )
        v = max(
            0.0,
            min(
                self.image.height() - 1.0,
                (position.y() - rectangle.y())
                * self.image.height()
                / rectangle.height(),
            ),
        )
        return u, v

    def mouseMoveEvent(self, event: Any) -> None:
        pixel = self.pixel_at(event.position())
        self.hover_pixel = (
            (round(pixel[0]), round(pixel[1])) if pixel is not None else None
        )
        self.hover_widget_x = event.position().x()
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self.hover_pixel = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        pixel = self.pixel_at(event.position())
        if pixel is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(*pixel)
        super().mousePressEvent(event)

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eaf1f7"))
        if self.image.isNull():
            painter.setPen(QColor("#526b83"))
            painter.setFont(QFont("Segoe UI", 15))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Start a camera or open an image\nThen freeze the frame and select four points",
            )
            return
        rectangle = self.image_rect()
        painter.drawImage(rectangle, self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, point in enumerate(self.points):
            if point is not None:
                self._draw_marker(painter, rectangle, point, str(index + 1), QColor("#ffad33"))
        if self.test_point is not None:
            self._draw_marker(painter, rectangle, self.test_point, "T", QColor("#ffd447"))
        if self.hover_pixel is not None:
            self._draw_magnifier(painter)

    def _draw_marker(
        self,
        painter: QPainter,
        rectangle: QRectF,
        point: tuple[float, float],
        label: str,
        color: QColor,
    ) -> None:
        px = rectangle.x() + point[0] * rectangle.width() / self.image.width()
        py = rectangle.y() + point[1] * rectangle.height() / self.image.height()
        painter.setPen(QPen(QColor("#111111"), 4))
        painter.drawEllipse(int(px - 6), int(py - 6), 12, 12)
        painter.setPen(QPen(color, 3))
        painter.drawEllipse(int(px - 6), int(py - 6), 12, 12)
        painter.drawLine(int(px - 12), int(py), int(px + 12), int(py))
        painter.drawLine(int(px), int(py - 12), int(px), int(py + 12))
        font = QFont("Segoe UI", 12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(int(px + 13), int(py - 10), label)

    def _draw_magnifier(self, painter: QPainter) -> None:
        if self.hover_pixel is None:
            return
        u, v = self.hover_pixel
        span, scale = 35, 5
        tile = QImage(span, span, QImage.Format.Format_RGB32)
        tile.fill(QColor("#eaf1f7"))
        tile_painter = QPainter(tile)
        tile_painter.drawImage(QPoint(span // 2 - u, span // 2 - v), self.image)
        tile_painter.end()

        side = span * scale + 20
        left = 12 if self.hover_widget_x > self.width() / 2 else self.width() - side - 12
        top = 12
        card = QRectF(left, top, side, side + 42)
        image_target = QRectF(left + 10, top + 34, span * scale, span * scale)
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#087f75"), 2))
        painter.drawRoundedRect(card, 9, 9)
        painter.setPen(QColor("#17304d"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(left + 10, top + 4, side - 20, 25),
            Qt.AlignmentFlag.AlignVCenter,
            f"5x   u={u}   v={v}",
        )
        painter.save()
        painter.setClipRect(image_target)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(image_target, tile)
        painter.restore()
        painter.setPen(QPen(QColor("#cbd9e6"), 1))
        painter.drawRect(image_target)
        center_x, center_y = image_target.center().x(), image_target.center().y()
        for color, thickness in ((QColor("#ffffff"), 4), (QColor("#f16b35"), 2)):
            painter.setPen(QPen(color, thickness))
            painter.drawLine(int(center_x - 12), int(center_y), int(center_x + 12), int(center_y))
            painter.drawLine(int(center_x), int(center_y - 12), int(center_x), int(center_y + 12))


def _numeric_box() -> QLineEdit:
    box = QLineEdit("0.0")
    validator = QDoubleValidator(-1_000_000_000, 1_000_000_000, 6, box)
    validator.setNotation(QDoubleValidator.Notation.StandardNotation)
    numeric_locale = QLocale.c()
    numeric_locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
    validator.setLocale(numeric_locale)
    box.setValidator(validator)
    box.setAlignment(Qt.AlignmentFlag.AlignRight)
    box.setMinimumWidth(96)
    box.setToolTip("Robot coordinate. Type a number, using a period for decimals.")
    return box


def _robot_base_url(text: str) -> str:
    """Accept a base address or a full /command URL without duplicating the endpoint."""
    value = text.strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        # Accessing port also catches malformed values such as :abc.
        _ = parsed.port
    except ValueError as error:
        raise ValueError("Enter an http:// or https:// robot URL without credentials or query parameters.") from error
    if parsed.path.endswith("/command"):
        value = value[: -len("/command")]
    return value


class CalibrationMoveWorker(QObject):
    completed = Signal(int)
    failed = Signal(str, bool)
    finished = Signal()

    def __init__(self, base_url: str, x: int, y: int) -> None:
        super().__init__()
        self.base_url = base_url
        self.x = x
        self.y = y

    @Slot()
    def run(self) -> None:
        adapter: RequestsRobotAdapter | None = None
        try:
            adapter = RequestsRobotAdapter(
                self.base_url,
                capabilities=RobotCapabilities(cartesian_move=True),
                timeout_s=3.0,
            )
            response = adapter.move_xyz(self.x, self.y, 100, 2000)
            self.completed.emit(response.status_code)
        except RobotTimeoutError:
            self.failed.emit(
                "Robot response timed out. The move may have started; check the robot before another click.",
                True,
            )
        except RobotError as error:
            self.failed.emit(f"Robot move failed: {error}", False)
        finally:
            try:
                if adapter is not None:
                    adapter.session.close()
            finally:
                self.finished.emit()


class CalibrationPage(QWidget):
    """Four-point calibration page with non-blocking camera, RTSP and image workers."""

    status_changed = Signal(str)

    def __init__(self, workspace: CandidateWorkspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.setObjectName("calibrationPage")
        self._current_image = QImage()
        self._source_mode: str | None = None
        self._selected_source_type = "camera"
        self._frozen = False
        self._points: list[tuple[float, float] | None] = [None] * 4
        self._selected_row = 0
        self._edit_mode = True
        self._record: CalibrationRecord | None = None
        self._camera_metadata: dict[str, Any] = {}
        self._roi_metadata: dict[str, Any] | None = None
        self._camera_thread: QThread | None = None
        self._camera_worker: CameraWorker | None = None
        self._image_thread: QThread | None = None
        self._image_worker: ImageLoadWorker | None = None
        self._robot_thread: QThread | None = None
        self._robot_worker: CalibrationMoveWorker | None = None
        self._pending_robot_target: dict[str, int] | None = None

        self._build_ui()
        self._select_row(0)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(10)

        eyebrow = QLabel("VISION  /  ROBOTICS")
        eyebrow.setObjectName("calibrationEyebrow")
        title = QLabel("Camera to robot calibration")
        title.setObjectName("calibrationTitle")
        subtitle = QLabel(
            "Freeze a full-frame image, select four corresponding points, fit a transform, and test clicks. Robot moves are optional."
        )
        subtitle.setObjectName("calibrationSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(eyebrow)
        root.addWidget(title)
        root.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setObjectName("calibrationSplitter")
        splitter.addWidget(self._build_source_panel())
        splitter.addWidget(self._build_configuration_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.setStyleSheet(
            """
            #calibrationPage { background: #f3f7fb; color: #172b4d; }
            QLabel#calibrationEyebrow { color: #078b80; font-size: 12px; font-weight: 800; }
            QLabel#calibrationTitle { color: #17304d; font-size: 28px; font-weight: 800; }
            QLabel#calibrationSubtitle { color: #61748c; font-size: 13px; }
            QFrame#calibrationCard { background: #ffffff; border: 1px solid #dce6ef; border-radius: 10px; }
            QLabel#calibrationSection { color: #17304d; font-size: 17px; font-weight: 750; }
            QLabel#calibrationHint { background: #fff6e5; color: #795214; border-radius: 8px; padding: 10px; }
            QLabel#calibrationStatus { background: #e9f8f5; color: #14675f; border-radius: 8px; padding: 10px; }
            QLabel#calibrationWarning { background: #fff0ed; color: #9f2d20; border-radius: 8px; padding: 10px; }
            QLabel#calibrationResult { background: #087f75; color: white; border-radius: 10px; padding: 14px; font-size: 25px; font-weight: 800; }
            QLabel#calibrationMatrixNote { color: #61748c; font-size: 12px; }
            QCheckBox#calibrationRobotToggle { background: #fff6e5; color: #795214; border: 1px solid #e7cf9d; border-radius: 8px; padding: 9px; font-weight: 750; }
            QCheckBox#calibrationRobotToggle:checked { background: #dff6ea; color: #14675f; border-color: #8ac7ad; }
            QCheckBox#calibrationRobotToggle::indicator { width: 19px; height: 19px; }
            QPushButton { background: #ffffff; color: #17304d; border: 1px solid #8fa5b5; border-radius: 5px; padding: 7px 13px; min-height: 32px; font-weight: 600; }
            QPushButton:hover { background: #e8f3f2; border-color: #087f75; }
            QPushButton#calibrationPrimary { background: #087f75; border-color: #087f75; color: white; font-weight: 800; }
            QPushButton#pointButton:checked { background: #ddf6f1; border: 2px solid #0b9d91; color: #086c63; font-weight: 800; }
            QRadioButton { background: #ffffff; color: #17304d; border: 1px solid #8fa5b5; border-radius: 5px; padding: 7px 13px; min-height: 32px; }
            QRadioButton:checked { background: #ddf6f1; border: 2px solid #0b9d91; color: #086c63; }
            QComboBox, QSpinBox, QLineEdit { background: #ffffff; color: #17304d; border: 1px solid #cbd9e6; border-radius: 7px; padding: 5px 8px; min-height: 27px; }
            QComboBox QAbstractItemView { background: #ffffff; color: #17304d; selection-background-color: #ddf6f1; selection-color: #086c63; }
            QPlainTextEdit#calibrationMatrix { background: #f2f6fb; color: #17304d; border: 1px solid #d4e2ed; border-radius: 8px; padding: 9px; }
            """
        )

    def _build_source_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("calibrationCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(9)

        heading = QLabel("01  Image source")
        heading.setObjectName("calibrationSection")
        layout.addWidget(heading)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Source"))
        self.source_choice = QComboBox()
        self.source_choice.addItem("Webcam", "camera")
        self.source_choice.addItem("RTSP stream", "rtsp")
        self.source_choice.currentIndexChanged.connect(self._source_choice_changed)
        camera_row.addWidget(self.source_choice)
        self.camera_index_label = QLabel("Camera index")
        camera_row.addWidget(self.camera_index_label)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 9)
        self.camera_index.setValue(0)
        camera_row.addWidget(self.camera_index)
        self.start_button = QPushButton("Start camera")
        self.start_button.setObjectName("calibrationPrimary")
        self.start_button.clicked.connect(self._toggle_camera)
        camera_row.addWidget(self.start_button)
        camera_row.addStretch(1)
        layout.addLayout(camera_row)

        self.rtsp_row = QWidget()
        rtsp_row_layout = QHBoxLayout(self.rtsp_row)
        rtsp_row_layout.setContentsMargins(0, 0, 0, 0)
        rtsp_row_layout.addWidget(QLabel("RTSP URL"))
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText("rtsp://10.16.224.100:8554/camera")
        self.rtsp_url.setToolTip("Enter a stream URL. It stays in this session only.")
        rtsp_row_layout.addWidget(self.rtsp_url, 1)
        layout.addWidget(self.rtsp_row)
        self.rtsp_row.setVisible(False)

        actions = QHBoxLayout()
        self.freeze_button = QPushButton("Freeze frame")
        self.freeze_button.clicked.connect(self._toggle_freeze)
        self.freeze_button.setEnabled(False)
        actions.addWidget(self.freeze_button)
        open_button = QPushButton("Open saved image...")
        open_button.clicked.connect(self._open_image)
        actions.addWidget(open_button)
        layout.addLayout(actions)

        self.view = CalibrationImageView()
        self.view.clicked.connect(self._image_clicked)
        layout.addWidget(self.view, 1)

        self.test_result = QLabel("ROBOT X: --     Y: --")
        self.test_result.setObjectName("calibrationResult")
        self.test_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.test_result.setMinimumHeight(70)
        layout.addWidget(self.test_result)

        self.status = QLabel(
            "Choose a webcam, RTSP stream or saved image. Freeze before selecting points."
        )
        self.status.setObjectName("calibrationStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        return panel

    def _build_configuration_panel(self) -> QScrollArea:
        panel = QFrame()
        panel.setObjectName("calibrationCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(8)

        heading = QLabel("02  Calibrate the plane")
        heading.setObjectName("calibrationSection")
        layout.addWidget(heading)
        hint = QLabel(
            "Select a numbered row, click its marker in the image, and enter the matching robot X/Y. Hover the image for the 5x cursor magnifier."
        )
        hint.setObjectName("calibrationHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        robot_heading = QLabel("Optional robot test move")
        robot_heading.setObjectName("calibrationSection")
        layout.addWidget(robot_heading)
        self.send_robot_on_click = QCheckBox("Enable robot movement on test clicks (Z = 100)")
        self.send_robot_on_click.setChecked(False)
        self.send_robot_on_click.setObjectName("calibrationRobotToggle")
        self.send_robot_on_click.setToolTip(
            "After calculating a matrix, click the frozen image to send one move to the robot."
        )
        layout.addWidget(self.send_robot_on_click)
        robot_url_row = QHBoxLayout()
        robot_url_row.addWidget(QLabel("Robot URL"))
        self.robot_url = QLineEdit()
        self.robot_url.setPlaceholderText("http://192.168.1.10:8080")
        self.robot_url.setToolTip("Base URL or full /command URL. The address stays in this session only.")
        robot_url_row.addWidget(self.robot_url, 1)
        layout.addLayout(robot_url_row)
        robot_note = QLabel(
            "Off by default. With a valid matrix, clicking a frozen image sends POST /command "
            "with the displayed X/Y, Z = 100 and duration = 2000 ms."
        )
        robot_note.setObjectName("calibrationMatrixNote")
        robot_note.setWordWrap(True)
        layout.addWidget(robot_note)

        table = QGridLayout()
        table.setHorizontalSpacing(6)
        table.setVerticalSpacing(5)
        for column, label in enumerate(("POINT", "PIXEL U", "PIXEL V", "ROBOT X", "ROBOT Y")):
            field = QLabel(label)
            field.setStyleSheet("color: #61748c; font-size: 11px; font-weight: 750;")
            table.addWidget(field, 0, column)
        self._point_buttons: list[QPushButton] = []
        self._pixel_labels: list[tuple[QLabel, QLabel]] = []
        self._robot_fields: list[tuple[QLineEdit, QLineEdit]] = []
        for row in range(4):
            button = QPushButton(f"{row + 1:02d}")
            button.setObjectName("pointButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, index=row: self._select_row(index))
            self._point_buttons.append(button)
            u_label, v_label = QLabel("--"), QLabel("--")
            u_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._pixel_labels.append((u_label, v_label))
            x_box, y_box = _numeric_box(), _numeric_box()
            x_box.textChanged.connect(self._invalidate_fit)
            y_box.textChanged.connect(self._invalidate_fit)
            self._robot_fields.append((x_box, y_box))
            for column, widget in enumerate((button, u_label, v_label, x_box, y_box)):
                table.addWidget(widget, row + 1, column)
        layout.addLayout(table)

        model_label = QLabel("MATRIX TYPE")
        model_label.setStyleSheet("color: #61748c; font-size: 11px; font-weight: 750;")
        layout.addWidget(model_label)
        model_row = QHBoxLayout()
        self.homography_choice = QRadioButton("3 x 3  Perspective")
        self.affine_choice = QRadioButton("2 x 3  Affine")
        self.model_group = QButtonGroup(self)
        self.model_group.setExclusive(True)
        self.model_group.addButton(self.homography_choice)
        self.model_group.addButton(self.affine_choice)
        self.homography_choice.setChecked(True)
        self.homography_choice.toggled.connect(self._invalidate_fit)
        self.affine_choice.toggled.connect(self._invalidate_fit)
        model_row.addWidget(self.homography_choice)
        model_row.addWidget(self.affine_choice)
        layout.addLayout(model_row)

        controls = QHBoxLayout()
        calculate_button = QPushButton("Calculate matrix")
        calculate_button.setObjectName("calibrationPrimary")
        calculate_button.clicked.connect(self._calculate)
        controls.addWidget(calculate_button)
        self.clear_button = QPushButton("Erase points and matrix")
        self.clear_button.clicked.connect(self._clear_points)
        controls.addWidget(self.clear_button)
        layout.addLayout(controls)

        matrix_label = QLabel("CAMERA PIXEL -> ROBOT PLANE")
        matrix_label.setStyleSheet("color: #61748c; font-size: 11px; font-weight: 750;")
        layout.addWidget(matrix_label)
        self.matrix_text = QPlainTextEdit()
        self.matrix_text.setObjectName("calibrationMatrix")
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setPlaceholderText("The matrix will appear here.")
        self.matrix_text.setFont(QFont("Consolas", 15))
        self.matrix_text.setMinimumHeight(105)
        self.matrix_text.setMaximumHeight(120)
        self.matrix_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.matrix_text)
        matrix_note = QLabel(
            "Displayed to 2 decimals; calculations and saved JSON retain full precision."
        )
        matrix_note.setObjectName("calibrationMatrixNote")
        matrix_note.setWordWrap(True)
        layout.addWidget(matrix_note)

        actions = QHBoxLayout()
        self.save_button = QPushButton("Save to workspace")
        self.save_button.clicked.connect(self._save_calibration)
        self.save_button.setEnabled(False)
        actions.addWidget(self.save_button)
        load_button = QPushButton("Load workspace calibration")
        load_button.clicked.connect(self._load_calibration)
        actions.addWidget(load_button)
        layout.addLayout(actions)

        self.geometry_warning = QLabel()
        self.geometry_warning.setObjectName("calibrationWarning")
        self.geometry_warning.setWordWrap(True)
        self.geometry_warning.setVisible(False)
        layout.addWidget(self.geometry_warning)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        return scroll

    def _set_status(self, message: str) -> None:
        self.status.setText(message)
        self.status_changed.emit(message)

    def _source_choice_changed(self, _index: int) -> None:
        if self._camera_thread is not None:
            return
        source_type = str(self.source_choice.currentData())
        if source_type != self._selected_source_type:
            self._selected_source_type = source_type
            self._frozen = False
            self._camera_metadata = {}
            self._set_current_image(QImage())
            self.freeze_button.setEnabled(False)
            self.freeze_button.setText("Freeze frame")
            self.test_result.setText("ROBOT X: --     Y: --")
            self._set_status("Source changed. Start the new source; points and matrix are retained.")
        rtsp = source_type == "rtsp"
        self.camera_index_label.setVisible(not rtsp)
        self.camera_index.setVisible(not rtsp)
        self.rtsp_row.setVisible(rtsp)
        self.start_button.setText("Start stream" if rtsp else "Start camera")

    def _toggle_camera(self) -> None:
        if self._camera_thread is not None:
            self._stop_camera()
            return
        source_type = str(self.source_choice.currentData())
        rtsp_url = None
        if source_type == "rtsp":
            try:
                rtsp_url = validate_rtsp_url(self.rtsp_url.text())
            except ValueError as error:
                self._set_status(str(error))
                return
        elif not CameraOwnership.acquire("calibration"):
            owner = CameraOwnership.current_owner() or "another module"
            self._set_status(f"Camera is already in use by {owner}. Stop it before starting calibration.")
            return
        self._source_mode = source_type
        self._frozen = False
        self._camera_metadata = {}
        self._set_current_image(QImage())
        self.view.test_point = None
        self.view.update()
        self.test_result.setText("ROBOT X: --     Y: --")
        self.freeze_button.setEnabled(False)
        self.freeze_button.setText("Freeze frame")
        self.start_button.setText("Stop stream" if source_type == "rtsp" else "Stop camera")
        self.source_choice.setEnabled(False)
        self.camera_index.setEnabled(False)
        self.rtsp_url.setEnabled(False)
        self._camera_thread = QThread(self)
        self._camera_worker = CameraWorker(
            self.camera_index.value(), source_type=source_type, rtsp_url=rtsp_url
        )
        self._camera_worker.moveToThread(self._camera_thread)
        self._camera_thread.started.connect(self._camera_worker.run)
        self._camera_worker.frame_ready.connect(self._camera_frame_ready)
        self._camera_worker.metadata_ready.connect(self._camera_metadata_ready)
        self._camera_worker.error.connect(self._camera_error)
        self._camera_worker.finished.connect(self._camera_thread.quit)
        self._camera_worker.finished.connect(self._camera_worker.deleteLater)
        self._camera_thread.finished.connect(self._camera_thread_finished)
        self._camera_thread.finished.connect(self._camera_thread.deleteLater)
        self._camera_thread.start()
        self._set_status(
            "Opening RTSP stream in a worker thread..."
            if source_type == "rtsp" else "Opening camera in a worker thread..."
        )

    def _stop_camera(self) -> None:
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self.start_button.setEnabled(False)
            self._set_status("Stopping image source...")

    @Slot(object)
    def _camera_frame_ready(self, rgb: Any) -> None:
        if self._source_mode not in ("camera", "rtsp") or self._frozen:
            return
        self._set_current_image(_qimage_from_rgb(rgb))
        self.freeze_button.setEnabled(True)

    @Slot(object)
    def _camera_metadata_ready(self, metadata: dict[str, Any]) -> None:
        if metadata.get("source_type") != self._source_mode:
            return
        self._camera_metadata = dict(metadata)
        self._update_geometry_warning()
        self._set_status(
            "RTSP live. Freeze the image before selecting or testing points. Existing calibration is retained."
            if self._source_mode == "rtsp"
            else "Camera live. Freeze the image before selecting or testing points. Existing calibration is retained."
        )

    @Slot(str)
    def _camera_error(self, message: str) -> None:
        if self._source_mode in ("camera", "rtsp"):
            if self._source_mode == "rtsp" and not self._frozen:
                self.freeze_button.setEnabled(False)
            self._set_status(message)

    @Slot()
    def _camera_thread_finished(self) -> None:
        CameraOwnership.release("calibration")
        self._camera_thread = None
        self._camera_worker = None
        self.source_choice.setEnabled(True)
        self.camera_index.setEnabled(True)
        self.rtsp_url.setEnabled(True)
        self.start_button.setEnabled(True)
        self._source_choice_changed(self.source_choice.currentIndex())
        self.freeze_button.setEnabled(not self._current_image.isNull())

    def _toggle_freeze(self) -> None:
        if self._current_image.isNull():
            self._set_status("Waiting for a camera frame or saved image.")
            return
        if self._source_mode not in ("camera", "rtsp"):
            self._frozen = True
            return
        self._frozen = not self._frozen
        if self._frozen:
            self._set_status("Frame frozen; select four corresponding points.")
            self.freeze_button.setText("Resume live")
        else:
            self.freeze_button.setText("Freeze frame")
            self.view.test_point = None
            self.view.update()
            self.test_result.setText("ROBOT X: --     Y: --")
            self._set_status("Source live; points and matrix retained. Freeze a frame to test again.")

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open calibration image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self._source_mode = "saved_image"
        self._frozen = True
        self._stop_camera()
        self._image_thread = QThread(self)
        self._image_worker = ImageLoadWorker(Path(path))
        self._image_worker.moveToThread(self._image_thread)
        self._image_thread.started.connect(self._image_worker.run)
        self._image_worker.image_ready.connect(self._image_ready)
        self._image_worker.error.connect(self._image_error)
        self._image_worker.finished.connect(self._image_thread.quit)
        self._image_worker.finished.connect(self._image_worker.deleteLater)
        self._image_thread.finished.connect(self._image_thread_finished)
        self._image_thread.finished.connect(self._image_thread.deleteLater)
        self._image_thread.start()
        self._set_status("Loading saved image in a worker thread...")

    @Slot(object, object)
    def _image_ready(self, rgb: Any, metadata: dict[str, Any]) -> None:
        self._source_mode = "saved_image"
        self._frozen = True
        self._camera_metadata = dict(metadata)
        self._set_current_image(_qimage_from_rgb(rgb))
        self.freeze_button.setEnabled(False)
        self.freeze_button.setText("Frame frozen")
        self._set_status("Saved image loaded; existing points and matrix retained. Check geometry before testing.")

    @Slot(str)
    def _image_error(self, message: str) -> None:
        self._set_status(message)

    @Slot()
    def _image_thread_finished(self) -> None:
        self._image_thread = None
        self._image_worker = None

    def _set_current_image(self, image: QImage) -> None:
        self._current_image = image
        self.view.set_image(image)
        self._update_geometry_warning()

    def _select_row(self, index: int) -> None:
        self._selected_row = index
        self._edit_mode = True
        for row, button in enumerate(self._point_buttons):
            button.setChecked(row == index)
        self._set_status(f"Editing point {index + 1}; click its position in the image.")

    def _image_clicked(self, u: float, v: float) -> None:
        if self._current_image.isNull() or not self._frozen:
            self._set_status("Freeze a camera frame before selecting or testing points.")
            return
        if self._record is not None and not self._edit_mode:
            try:
                result = CalibrationService.apply(
                    self._record,
                    (u, v),
                    (self._current_image.width(), self._current_image.height()),
                )
            except CalibrationError as error:
                self._show_geometry_warning(str(error))
                self._set_status(str(error))
                return
            self.view.test_point = (u, v)
            self.view.update()
            self.test_result.setText(
                f"ROBOT X: {round(result[0])}     Y: {round(result[1])}"
            )
            if self.send_robot_on_click.isChecked():
                mismatch = self._robot_source_mismatch()
                if mismatch is not None:
                    self._show_geometry_warning(mismatch)
                    self._set_status(f"Robot command blocked: {mismatch}")
                else:
                    self._send_robot_command(round(result[0]), round(result[1]))
            else:
                self._set_status("Test prediction shown. Robot sending is off.")
            return

        self._points[self._selected_row] = (float(round(u)), float(round(v)))
        self.view.set_points(self._points)
        self._pixel_labels[self._selected_row][0].setText(str(round(u)))
        self._pixel_labels[self._selected_row][1].setText(str(round(v)))
        self._invalidate_fit()
        next_row = next(
            (index for index, point in enumerate(self._points) if point is None),
            None,
        )
        if next_row is not None:
            self._select_row(next_row)
        else:
            self._set_status("Four image points selected; enter robot X/Y and calculate.")

    def _robot_source_mismatch(self) -> str | None:
        if self._record is None:
            return None
        expected = self._record.camera_metadata or {}
        actual = self._camera_metadata
        if expected.get("source_type") and actual.get("source_type") != expected["source_type"]:
            return "The current image source differs from the calibrated source."
        if (
            expected.get("source_type") == "camera"
            and expected.get("camera_index") is not None
            and actual.get("camera_index") != expected["camera_index"]
        ):
            return "The camera index differs from the calibrated camera."
        if (
            expected.get("source_type") == "saved_image"
            and expected.get("path")
            and actual.get("path") != expected["path"]
        ):
            return "The saved image differs from the calibrated image."
        if (
            expected.get("source_type") == "rtsp"
            and expected.get("source_id") != actual.get("source_id")
        ):
            return "The RTSP stream differs from the calibrated stream."
        return None

    def _send_robot_command(self, x: int, y: int) -> None:
        if self._robot_thread is not None:
            self._set_status("A robot command is still in progress; no new move was queued.")
            return
        try:
            base_url = _robot_base_url(self.robot_url.text())
        except ValueError as error:
            self._set_status(str(error))
            return
        target = {"x": x, "y": y, "z": 100, "duration_ms": 2000}
        self._pending_robot_target = target
        self._robot_thread = QThread(self)
        self._robot_worker = CalibrationMoveWorker(base_url, x, y)
        self._robot_worker.moveToThread(self._robot_thread)
        self._robot_thread.started.connect(self._robot_worker.run)
        self._robot_worker.completed.connect(self._robot_move_completed)
        self._robot_worker.failed.connect(self._robot_move_failed)
        self._robot_worker.finished.connect(self._robot_thread.quit)
        self._robot_worker.finished.connect(self._robot_worker.deleteLater)
        self._robot_thread.finished.connect(self._robot_thread_finished)
        self._robot_thread.finished.connect(self._robot_thread.deleteLater)
        self.workspace.record_event("calibration.robot_test.started", target)
        self._robot_thread.start()
        self._set_status(f"Sending move to ({x}, {y}, 100); waiting for the robot response.")

    @Slot(int)
    def _robot_move_completed(self, status_code: int) -> None:
        self.workspace.record_event(
            "calibration.robot_test.completed",
            {**(self._pending_robot_target or {}), "status_code": status_code},
        )
        self._set_status(f"Robot accepted the move (HTTP {status_code}).")

    @Slot(str, bool)
    def _robot_move_failed(self, message: str, uncertain: bool) -> None:
        self.workspace.record_event(
            "calibration.robot_test.uncertain" if uncertain else "calibration.robot_test.failed",
            self._pending_robot_target or {},
            outcome="uncertain" if uncertain else "error",
        )
        self._set_status(message)

    @Slot()
    def _robot_thread_finished(self) -> None:
        self._robot_thread = None
        self._robot_worker = None
        self._pending_robot_target = None

    def _invalidate_fit(self, *_args: Any) -> None:
        self._record = None
        self._edit_mode = True
        self.save_button.setEnabled(False)
        self.matrix_text.clear()
        self.test_result.setText("ROBOT X: --     Y: --")
        self.view.test_point = None
        self.view.update()

    def _clear_points(self) -> None:
        self._points = [None] * 4
        self.view.set_points(self._points)
        self.view.test_point = None
        self._record = None
        self._edit_mode = True
        for row, (u_label, v_label) in enumerate(self._pixel_labels):
            u_label.setText("--")
            v_label.setText("--")
            for field in self._robot_fields[row]:
                field.blockSignals(True)
                field.setText("0.0")
                field.blockSignals(False)
        self.matrix_text.clear()
        self.test_result.setText("ROBOT X: --     Y: --")
        self.save_button.setEnabled(False)
        self.geometry_warning.setVisible(False)
        self._selected_row = 0
        for index, button in enumerate(self._point_buttons):
            button.setChecked(index == 0)
        self._set_status("Current points and matrix erased. A previously saved workspace calibration remains available.")

    def _calculate(self) -> None:
        if self._current_image.isNull() or not self._frozen:
            self._set_status("Freeze a camera frame or open a saved image first.")
            return
        if any(point is None for point in self._points):
            self._set_status("Select all four image points first.")
            return
        if any(not field.hasAcceptableInput() for pair in self._robot_fields for field in pair):
            self._set_status("Enter a valid number for every robot X/Y coordinate.")
            return
        robot_points = [(float(x.text()), float(y.text())) for x, y in self._robot_fields]
        transform: TransformType = (
            "homography" if self.homography_choice.isChecked() else "affine"
        )
        try:
            self._record = CalibrationService.fit(
                transform,
                [point for point in self._points if point is not None],
                robot_points,
                (self._current_image.width(), self._current_image.height()),
                camera_metadata=self._camera_metadata,
                roi=self._roi_metadata,
                snapshot_png=self._snapshot_png(),
            )
        except CalibrationError as error:
            self._set_status(str(error))
            return
        self.matrix_text.setPlainText(_format_matrix(self._record.matrix))
        self._edit_mode = False
        for button in self._point_buttons:
            button.setChecked(False)
        self.save_button.setEnabled(True)
        self._update_geometry_warning()
        point_errors = []
        for (u, v), (x, y) in zip(
            self._record.camera_points_uv, self._record.robot_points_xy
        ):
            fitted_x, fitted_y = CalibrationService.apply(
                self._record, (u, v), self._record.image_size
            )
            point_errors.append(hypot(fitted_x - x, fitted_y - y))
        self._set_status(
            f"Matrix ready. Maximum point fit error: {max(point_errors):.2f} "
            "robot units. Click the image to test a robot X/Y prediction."
        )

    def _snapshot_png(self) -> bytes | None:
        if self._current_image.isNull():
            return None
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not self._current_image.save(buffer, "PNG"):
            return None
        return bytes(buffer.data())

    def _calibration_path(self) -> Path:
        return self.workspace.resolve_inside("calibration", "calibration.json")

    def _save_calibration(self) -> None:
        if self._record is None:
            self._set_status("Calculate a matrix before saving.")
            return
        try:
            path = self._calibration_path()
            CalibrationService.save(path, self._record)
            self.workspace.record_event(
                "calibration.saved",
                {
                    "path": str(path.relative_to(self.workspace.root)),
                    "transform": self._record.transform,
                    "image_size": list(self._record.image_size),
                },
            )
            self._set_status(f"Saved active calibration to {path}.")
        except (OSError, CalibrationError, ValueError) as error:
            self._set_status(f"Could not save calibration: {error}")

    def _load_calibration(self) -> None:
        path = self._calibration_path()
        if not path.exists():
            self._set_status(f"No workspace calibration found at {path}.")
            return
        try:
            record = CalibrationService.load(path)
        except (OSError, KeyError, TypeError, ValueError, binascii.Error) as error:
            self._set_status(f"Could not load calibration: {error}")
            return

        restored_snapshot = False
        if self._current_image.isNull() and record.snapshot_png is not None:
            snapshot = QImage.fromData(record.snapshot_png, "PNG")
            if not snapshot.isNull():
                self._source_mode = "saved_calibration"
                self._frozen = True
                self._set_current_image(snapshot)
                self.freeze_button.setEnabled(False)
                self.freeze_button.setText("Frame frozen")
                restored_snapshot = True
        self.homography_choice.blockSignals(True)
        self.affine_choice.blockSignals(True)
        self.homography_choice.setChecked(record.transform == "homography")
        self.affine_choice.setChecked(record.transform == "affine")
        self.homography_choice.blockSignals(False)
        self.affine_choice.blockSignals(False)
        self._record = record
        self._points = [tuple(point) for point in record.camera_points_uv]
        self.view.set_points(self._points)
        for row, ((u, v), (x, y)) in enumerate(
            zip(record.camera_points_uv, record.robot_points_xy)
        ):
            self._pixel_labels[row][0].setText(str(round(u)))
            self._pixel_labels[row][1].setText(str(round(v)))
            for field, value in zip(self._robot_fields[row], (x, y)):
                field.blockSignals(True)
                field.setText(f"{value:.6f}".rstrip("0").rstrip(".") or "0")
                field.blockSignals(False)
        self.matrix_text.setPlainText(_format_matrix(record.matrix))
        if restored_snapshot:
            self._camera_metadata = dict(record.camera_metadata or {})
        self._roi_metadata = dict(record.roi) if record.roi is not None else None
        self._edit_mode = False
        self.save_button.setEnabled(True)
        self._update_geometry_warning()
        self._set_status(
            f"Loaded schema {record.source_schema_version} calibration. "
            "Click the image to test only when geometry matches."
        )

    def _update_geometry_warning(self) -> None:
        if self._record is None:
            self.geometry_warning.setVisible(False)
            return
        if self._current_image.isNull():
            self._show_geometry_warning(
                "Calibration loaded without a current image. Load a matching image before testing."
            )
            return
        try:
            CalibrationService.validate_geometry(
                self._record,
                (self._current_image.width(), self._current_image.height()),
            )
        except CalibrationGeometryMismatchError as error:
            self._show_geometry_warning(str(error))
        else:
            mismatch = self._robot_source_mismatch()
            if mismatch is not None:
                self._show_geometry_warning(mismatch)
            else:
                self.geometry_warning.clear()
                self.geometry_warning.setVisible(False)

    def _show_geometry_warning(self, message: str) -> None:
        self.geometry_warning.setText(message)
        self.geometry_warning.setVisible(True)

    def shutdown(self) -> None:
        self._stop_camera()
        if self._camera_thread is not None:
            self._camera_thread.quit()
            self._camera_thread.wait(7000)
        if self._image_thread is not None:
            self._image_thread.quit()
            self._image_thread.wait(1500)
        if self._robot_thread is not None:
            self._robot_thread.quit()
            self._robot_thread.wait(6000)

