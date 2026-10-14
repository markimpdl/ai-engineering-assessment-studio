"""M5 Robot Studio: embedded simulator and supervised shared robot controls."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.workspace import CandidateWorkspace
from ..services.robot import (
    POSITIONS,
    PhysicalMaxArmAdapter,
    RobotError,
    RobotSafetyError,
    RobotService,
    RobotUncertainError,
    SimulatorRobotAdapter,
)
from ..services.simulator import EmbeddedSimulator


class TaskSignals(QObject):
    result = Signal(object)
    failed = Signal(str)
    finished = Signal()


class RobotTask(QRunnable):
    def __init__(self, function: Callable[[], object]) -> None:
        super().__init__()
        self.function = function
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.result.emit(self.function())
        except Exception as error:
            self.signals.failed.emit(str(error))
        finally:
            self.signals.finished.emit()


class RobotPage(QWidget):
    trial_progress_signal = Signal(int, int, str)

    def __init__(self, workspace: CandidateWorkspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.thread_pool = QThreadPool.globalInstance()
        self.simulator = EmbeddedSimulator()
        self.simulator_ready = False
        self._state_request_active = False
        self._last_state_signature: tuple[Any, ...] | None = None
        self._trial_cancel = threading.Event()
        self._manual_buttons: list[QPushButton] = []
        self._build_ui()
        self.trial_progress_signal.connect(self._update_trial_progress)
        self._start_embedded_simulator()
        self._load_subset_classes()
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(250)
        self._state_timer.timeout.connect(self._refresh_simulator_state)
        self._state_timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(9)
        eyebrow = QLabel("ROBOT")
        eyebrow.setObjectName("robotEyebrow")
        title = QLabel("Robot Studio")
        title.setObjectName("robotTitle")
        subtitle = QLabel(
            "Use the verified MaxArm HTTP contract for manual moves and supervised simulator trials."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("robotSubtitle")
        root.addWidget(eyebrow)
        root.addWidget(title)
        root.addWidget(subtitle)

        self.status = QLabel("Starting local simulator...")
        self.status.setObjectName("robotStatus")
        self.target_badge = QLabel("TARGET: LOCAL SIMULATOR")
        self.target_badge.setObjectName("robotTarget")
        header = QHBoxLayout()
        header.addWidget(self.status, 1)
        header.addWidget(self.target_badge)
        root.addLayout(header)

        connection = QGroupBox("Connection and safety")
        connection_layout = QGridLayout(connection)
        self.target_combo = QComboBox()
        self.target_combo.addItems(["Local simulator", "Physical MaxArm"])
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.base_url = QLineEdit(self.simulator.base_url)
        self.base_url.setMinimumWidth(260)
        self.connect_button = QPushButton("Check health")
        self.connect_button.clicked.connect(lambda: self._run_task(self._check_health, self._show_health))
        self.deep_button = QPushButton("Deep health")
        self.deep_button.clicked.connect(lambda: self._run_task(lambda: self.service.health(True), self._show_health))
        self.positions_button = QPushButton("Read positions")
        self.positions_button.clicked.connect(lambda: self._run_task(self.service.positions, self._show_positions))
        self.expose_lan = QCheckBox("Expose simulator on LAN")
        self.expose_lan.setToolTip("Restart the simulator on 0.0.0.0. Use only on a trusted network.")
        self.expose_lan.toggled.connect(self._toggle_lan)
        self.arm_button = QPushButton("Enable physical motion")
        self.arm_button.setEnabled(False)
        self.arm_button.clicked.connect(self._arm_physical)
        connection_layout.addWidget(QLabel("Target"), 0, 0)
        connection_layout.addWidget(self.target_combo, 0, 1)
        connection_layout.addWidget(QLabel("Base URL"), 0, 2)
        connection_layout.addWidget(self.base_url, 0, 3)
        connection_layout.addWidget(self.connect_button, 0, 4)
        connection_layout.addWidget(self.deep_button, 0, 5)
        connection_layout.addWidget(self.positions_button, 1, 4)
        connection_layout.addWidget(self.expose_lan, 1, 0, 1, 2)
        connection_layout.addWidget(self.arm_button, 1, 3)
        root.addWidget(connection)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self._build_manual_group())
        left.addWidget(self._build_trial_group())
        left.addWidget(self._build_xyz_group())
        left.addStretch(1)
        right = QVBoxLayout()
        right.addWidget(self._build_state_group())
        right.addWidget(self._build_positions_group())
        right.addWidget(self._build_history_group(), 1)
        body.addLayout(left, 1)
        body.addLayout(right, 1)
        root.addLayout(body, 1)

        self.setStyleSheet(
            """
            #robotPage { background: #f5f8f7; color: #17211d; }
            QLabel#robotEyebrow { color: #17745d; font-size: 12px; font-weight: 800; }
            QLabel#robotTitle { color: #153c2d; font-size: 28px; font-weight: 800; }
            QLabel#robotSubtitle { color: #566b62; font-size: 13px; }
            QLabel#robotStatus { background: #e7f4ed; color: #18583e; padding: 8px; border-radius: 6px; }
            QLabel#robotTarget { background: #174c36; color: white; padding: 8px; border-radius: 6px; font-weight: 800; }
            QGroupBox { background: #ffffff; border: 1px solid #d6e2dc; border-radius: 7px; margin-top: 8px; padding: 10px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton, QComboBox, QLineEdit, QSpinBox { min-height: 27px; padding: 4px 7px; }
            QPushButton { background: #ffffff; color: #17304d; border: 1px solid #8fa5b5; border-radius: 5px; min-height: 32px; padding: 7px 13px; font-weight: 600; }
            QPushButton:hover { background: #e8f3f2; border-color: #087f75; }
            QPushButton#robotPrimary { background: #17745d; color: white; font-weight: 800; }
            QTableWidget, QPlainTextEdit { background: white; border: 1px solid #d5e2dc; }
            QProgressBar { min-height: 17px; }
            """
        )

    def _build_manual_group(self) -> QGroupBox:
        group = QGroupBox("Manual verified commands")
        layout = QGridLayout(group)
        self.location_combo = QComboBox()
        self.location_combo.addItems(sorted(POSITIONS))
        self.height_combo = QComboBox()
        self.height_combo.addItems(["high", "low"])
        self.duration = QSpinBox()
        self.duration.setRange(100, 10_000)
        self.duration.setValue(1000)
        self.duration.setSuffix(" ms")
        self.move_button = QPushButton("Move")
        self.move_button.setObjectName("robotPrimary")
        self.move_button.clicked.connect(self._manual_move)
        self.suction_on = QPushButton("Suction on")
        self.suction_off = QPushButton("Suction off")
        self.suction_on.clicked.connect(lambda: self._manual_suction("on"))
        self.suction_off.clicked.connect(lambda: self._manual_suction("off"))
        self._manual_buttons.extend([self.move_button, self.suction_on, self.suction_off])
        for row, (label, widget) in enumerate(
            [("Location", self.location_combo), ("Height", self.height_combo), ("Duration", self.duration)]
        ):
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)
        layout.addWidget(self.move_button, 0, 2, 3, 1)
        layout.addWidget(self.suction_on, 0, 3)
        layout.addWidget(self.suction_off, 1, 3)
        return group

    def _build_trial_group(self) -> QGroupBox:
        group = QGroupBox("Supervised simulator pick and place")
        layout = QGridLayout(group)
        self.trial_class = QComboBox()
        self.mapping_table = QTableWidget(0, 2)
        self.mapping_table.setHorizontalHeaderLabels(["Class", "Destination"])
        self.mapping_table.setMaximumHeight(150)
        self.mapping_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.trial_source = QComboBox()
        self.trial_source.addItems(sorted(POSITIONS))
        self.trial_destination = QComboBox()
        self.trial_destination.addItems(sorted(POSITIONS))
        self.trial_run = QPushButton("Run supervised trial")
        self.trial_run.setObjectName("robotPrimary")
        self.trial_run.clicked.connect(self._run_trial)
        self.trial_cancel = QPushButton("Cancel next step")
        self.trial_cancel.setEnabled(False)
        self.trial_cancel.clicked.connect(lambda: self._trial_cancel.set())
        self.trial_progress = QProgressBar()
        self.trial_progress.setRange(0, 8)
        self.trial_progress.setValue(0)
        self.trial_class.currentTextChanged.connect(self._sync_trial_destination)
        self.trial_destination.currentTextChanged.connect(self._set_selected_mapping)
        layout.addWidget(self.mapping_table, 0, 0, 1, 3)
        layout.addWidget(QLabel("Run class"), 1, 0)
        layout.addWidget(self.trial_class, 1, 1)
        layout.addWidget(QLabel("Source"), 2, 0)
        layout.addWidget(self.trial_source, 2, 1)
        layout.addWidget(QLabel("Destination"), 3, 0)
        layout.addWidget(self.trial_destination, 3, 1)
        layout.addWidget(self.trial_run, 1, 2, 2, 1)
        layout.addWidget(self.trial_cancel, 3, 2)
        layout.addWidget(self.trial_progress, 4, 0, 1, 3)
        return group

    def _build_xyz_group(self) -> QGroupBox:
        group = QGroupBox("Simulator-only XYZ")
        layout = QFormLayout(group)
        self.xyz_x, self.xyz_y, self.xyz_z = (QSpinBox() for _ in range(3))
        for widget, value, minimum, maximum in (
            (self.xyz_x, -3, -300, 100),
            (self.xyz_y, -130, -300, 100),
            (self.xyz_z, 89, 0, 300),
        ):
            widget.setRange(minimum, maximum)
            widget.setValue(value)
            widget.setSuffix(" mm")
        self.xyz_button = QPushButton("Move XYZ")
        self.xyz_button.clicked.connect(self._manual_xyz)
        layout.addRow("X", self.xyz_x)
        layout.addRow("Y", self.xyz_y)
        layout.addRow("Z", self.xyz_z)
        layout.addRow(self.xyz_button)
        return group

    def _build_state_group(self) -> QGroupBox:
        group = QGroupBox("Live simulator state")
        layout = QFormLayout(group)
        self.current_label = QLabel("-")
        self.target_label = QLabel("-")
        self.progress_label = QLabel("-")
        self.suction_label = QLabel("-")
        self.objects_label = QLabel("-")
        for label in (self.current_label, self.target_label, self.progress_label, self.suction_label, self.objects_label):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addRow("Current XYZ", self.current_label)
        layout.addRow("Target XYZ", self.target_label)
        layout.addRow("Motion progress", self.progress_label)
        layout.addRow("Suction", self.suction_label)
        layout.addRow("Objects", self.objects_label)
        return group

    def _build_positions_group(self) -> QGroupBox:
        group = QGroupBox("Verified positions")
        layout = QVBoxLayout(group)
        self.positions_table = QTableWidget(0, 7)
        self.positions_table.setHorizontalHeaderLabels(["Position", "Low X", "Low Y", "Low Z", "High X", "High Y", "High Z"])
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        self.positions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.positions_table)
        self._show_positions_from_contract(POSITIONS)
        return group

    def _build_history_group(self) -> QGroupBox:
        group = QGroupBox("Simulator command history")
        layout = QVBoxLayout(group)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setMaximumBlockCount(120)
        layout.addWidget(self.history)
        return group

    def _start_embedded_simulator(self) -> None:
        def start() -> None:
            try:
                self.simulator.start()
                self.simulator_ready = True
            except Exception as error:
                self.simulator_ready = False
                self.workspace.record_event("robot.simulator.error", {"error": str(error)}, outcome="error")

        threading.Thread(target=start, name="robot-simulator-start", daemon=True).start()

    @property
    def service(self) -> RobotService:
        if self.target_combo.currentIndex() == 0:
            adapter = SimulatorRobotAdapter(self.base_url.text().strip() or self.simulator.base_url)
        else:
            adapter = PhysicalMaxArmAdapter(self.base_url.text().strip())
        if not hasattr(self, "_service") or self._service.adapter.base_url != adapter.base_url or self._service.capabilities != adapter.capabilities:
            self._service = RobotService(self.workspace, adapter)
        return self._service

    def _run_task(self, function: Callable[[], object], on_result: Callable[[object], None]) -> None:
        task = RobotTask(function)
        task.signals.result.connect(on_result)
        task.signals.failed.connect(self._show_error)
        self.thread_pool.start(task)

    def _check_health(self) -> object:
        if self.target_combo.currentIndex() == 1 and not self.service.physical_armed:
            return self.service.health()
        return self.service.health()

    def _show_health(self, response: object) -> None:
        data = getattr(response, "data", {})
        status = data.get("status", data.get("ok", "unknown"))
        self.status.setText(f"Health: {status} at {self.service.adapter.base_url}")

    def _show_positions(self, response: object) -> None:
        data = getattr(response, "data", {})
        self._show_positions_from_contract(data.get("positions", {}))
        self.status.setText("Positions read successfully.")

    def _show_positions_from_contract(self, positions: dict[str, Any]) -> None:
        self.positions_table.setRowCount(0)
        for name in sorted(positions):
            row = self.positions_table.rowCount()
            self.positions_table.insertRow(row)
            values = [name]
            for height in ("low", "high"):
                values.extend(str(value) for value in positions[name].get(height, ["-", "-", "-"]))
            for column, value in enumerate(values):
                self.positions_table.setItem(row, column, QTableWidgetItem(value))

    def _show_error(self, message: str) -> None:
        self.status.setText(f"Robot error: {message}")

    def _target_changed(self, index: int) -> None:
        physical = index == 1
        self.target_badge.setText("TARGET: PHYSICAL MAXARM" if physical else "TARGET: LOCAL SIMULATOR")
        self.arm_button.setEnabled(physical)
        self.expose_lan.setEnabled(not physical)
        self.xyz_button.setEnabled(not physical)
        if physical:
            self.status.setText("Physical target is disarmed. Check health and enable the session explicitly.")
        else:
            self.base_url.setText(self.simulator.base_url)
            self.status.setText("Local simulator selected; physical motion remains unavailable.")

    def _toggle_lan(self, enabled: bool) -> None:
        if self.target_combo.currentIndex() != 0:
            return
        self.simulator.stop()
        self.simulator = EmbeddedSimulator(host="0.0.0.0") if enabled else EmbeddedSimulator()
        self.base_url.setText(self.simulator.base_url)
        self.simulator_ready = False
        self._start_embedded_simulator()
        self.status.setText("Simulator restarting with LAN exposure." if enabled else "Simulator restarting on localhost.")

    def _arm_physical(self) -> None:
        self.arm_button.setEnabled(False)
        self._run_task(self.service.arm_physical, lambda _: (self.status.setText("Physical motion enabled for this session."), self.arm_button.setText("Physical motion enabled")))

    def _set_manual_enabled(self, enabled: bool) -> None:
        for button in self._manual_buttons:
            button.setEnabled(enabled)
        self.trial_run.setEnabled(enabled)
        self.target_combo.setEnabled(enabled)
        self.connect_button.setEnabled(enabled)
        self.positions_button.setEnabled(enabled)

    def _manual_move(self) -> None:
        location, height, duration = self.location_combo.currentText(), self.height_combo.currentText(), self.duration.value()
        self._run_task(lambda: self.service.move(location, height, duration), lambda _: self.status.setText(f"Move accepted: {location}/{height}."))

    def _manual_suction(self, state: str) -> None:
        self._run_task(lambda: self.service.suction(state), lambda _: self.status.setText(f"Suction {state} command accepted."))

    def _manual_xyz(self) -> None:
        values = (self.xyz_x.value(), self.xyz_y.value(), self.xyz_z.value(), self.duration.value())
        self._run_task(lambda: self.service.move_xyz(*values), lambda _: self.status.setText("Simulator XYZ move accepted."))

    def _run_trial(self) -> None:
        if self.trial_source.currentText() == self.trial_destination.currentText():
            self.status.setText("Choose different source and destination positions.")
            return
        self._trial_cancel.clear()
        self._set_manual_enabled(False)
        self.trial_cancel.setEnabled(True)
        source, destination, duration = self.trial_source.currentText(), self.trial_destination.currentText(), self.duration.value()
        self.status.setText(f"Supervised trial running: {source} to {destination}.")
        task = RobotTask(lambda: self.service.run_pick_place(source, destination, duration, cancel_event=self._trial_cancel, progress=self._trial_progress))
        task.signals.result.connect(self._trial_finished)
        task.signals.failed.connect(self._trial_failed)
        self.thread_pool.start(task)

    def _trial_progress(self, completed: int, total: int, label: str) -> None:
        self.trial_progress_signal.emit(completed, total, label)

    @Slot(int, int, str)
    def _update_trial_progress(self, completed: int, total: int, label: str) -> None:
        self.trial_progress.setValue(completed)
        self.trial_progress.setFormat(f"{completed}/{total}: {label}")

    def _trial_finished(self, result: object) -> None:
        self._set_manual_enabled(True)
        self.trial_cancel.setEnabled(False)
        self.trial_progress.setValue(len(getattr(result, "completed_steps", ())))
        self.status.setText(f"Trial {getattr(result, 'status', 'finished')}: {getattr(result, 'message', '')}".strip())

    def _trial_failed(self, message: str) -> None:
        self._set_manual_enabled(True)
        self.trial_cancel.setEnabled(False)
        self.status.setText(f"Trial failed: {message}")

    def _load_subset_classes(self) -> None:
        manifests = sorted(self.workspace.root.joinpath("dataset").glob("**/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        names: list[str] = []
        if manifests:
            try:
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                mapping = manifest.get("selected_class_mapping", {})
                names = [str(mapping[str(original)]["name"]) for original in manifest.get("selected_order", []) if str(original) in mapping]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                names = []
        self.trial_class.clear()
        self.trial_class.addItems(names or ["No exported classes found"])
        self.trial_class.setEnabled(bool(names))
        self.mapping_table.setRowCount(0)
        for index, name in enumerate(names):
            self.mapping_table.insertRow(index)
            self.mapping_table.setItem(index, 0, QTableWidgetItem(name))
            destination = QComboBox()
            destination.addItems(sorted(POSITIONS))
            destination.setCurrentText(sorted(POSITIONS)[index % len(POSITIONS)])
            destination.currentTextChanged.connect(
                lambda _text, class_name=name: self._sync_trial_destination(class_name)
            )
            self.mapping_table.setCellWidget(index, 1, destination)
        if names:
            self._sync_trial_destination(names[0])

    def _sync_trial_destination(self, class_name: str) -> None:
        """Show a mapping row's destination in the trial selector when it is the selected class."""

        if class_name != self.trial_class.currentText():
            return
        for row in range(self.mapping_table.rowCount()):
            item = self.mapping_table.item(row, 0)
            if item is not None and item.text() == class_name:
                combo = self.mapping_table.cellWidget(row, 1)
                if isinstance(combo, QComboBox):
                    self.trial_destination.blockSignals(True)
                    self.trial_destination.setCurrentText(combo.currentText())
                    self.trial_destination.blockSignals(False)
                return

    def _set_selected_mapping(self, destination: str) -> None:
        class_name = self.trial_class.currentText()
        for row in range(self.mapping_table.rowCount()):
            item = self.mapping_table.item(row, 0)
            combo = self.mapping_table.cellWidget(row, 1)
            if item is not None and item.text() == class_name and isinstance(combo, QComboBox):
                combo.blockSignals(True)
                combo.setCurrentText(destination)
                combo.blockSignals(False)
                return

    def _refresh_simulator_state(self) -> None:
        if self._state_request_active or not self.simulator_ready or self.target_combo.currentIndex() != 0:
            return
        self._state_request_active = True
        adapter = SimulatorRobotAdapter(self.base_url.text().strip() or self.simulator.base_url, timeout_s=0.5)
        task = RobotTask(adapter.simulator_state)
        task.signals.result.connect(self._show_simulator_state)
        task.signals.failed.connect(lambda _message: None)
        task.signals.finished.connect(lambda: setattr(self, "_state_request_active", False))
        self.thread_pool.start(task)

    def _show_simulator_state(self, response: object) -> None:
        data = getattr(response, "data", {})
        self.current_label.setText(str(data.get("current_xyz", "-")))
        self.target_label.setText(str(data.get("target_xyz", "-")))
        self.progress_label.setText(f"{float(data.get('motion_progress', 0.0)) * 100:.0f}%" + (" moving" if data.get("moving") else " settled"))
        self.suction_label.setText("ON" if data.get("suction_on") else "OFF")
        objects = data.get("objects", [])
        self.objects_label.setText(", ".join(f"{item.get('object_id')}: {item.get('placed_at') or 'held'}" for item in objects) or "none")
        signature = (
            tuple(data.get("target_xyz", ())),
            bool(data.get("moving")),
            bool(data.get("suction_on")),
            data.get("attached_object_id"),
        )
        if signature != self._last_state_signature:
            self._last_state_signature = signature
            self.workspace.record_event(
                "robot.simulator.state.changed",
                {
                    "target_xyz": data.get("target_xyz"),
                    "moving": data.get("moving"),
                    "suction_on": data.get("suction_on"),
                    "attached_object_id": data.get("attached_object_id"),
                },
            )
        events = data.get("events", [])
        self.history.setPlainText("\n".join(f"{item.get('time', '')}  {item.get('level', '')}  {item.get('message', '')}" for item in events[-40:]))

    def shutdown(self) -> None:
        if hasattr(self, "_state_timer"):
            self._state_timer.stop()
        self._trial_cancel.set()
        self.service.disarm()
        self.simulator.stop()

