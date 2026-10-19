"""M6 Training Studio page backed by the workspace training service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
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
from ..services.training import (
    TrainingConfig,
    TrainingError,
    TrainingService,
    validate_subset,
)


class TrainingTaskSignals(QObject):
    started = Signal(object)
    failed = Signal(str)
    finished = Signal()


class TrainingStartTask(QRunnable):
    def __init__(self, function: Any) -> None:
        super().__init__()
        self.function = function
        self.signals = TrainingTaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit(self.function())
        except Exception as error:
            self.signals.failed.emit(str(error))
        finally:
            self.signals.finished.emit()


class TrainingPage(QWidget):
    output_signal = Signal(str)
    progress_signal = Signal(int, int)
    complete_signal = Signal(object)

    def __init__(self, workspace: CandidateWorkspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.service = TrainingService(workspace)
        self.thread_pool = QThreadPool.globalInstance()
        self.current_run_id: str | None = None
        self._build_ui()
        self.output_signal.connect(self._append_log)
        self.progress_signal.connect(self._set_progress)
        self.complete_signal.connect(self._run_complete)
        self._refresh_subsets()
        self._refresh_runs()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(9)
        eyebrow = QLabel("TRAINING")
        eyebrow.setObjectName("trainingEyebrow")
        title = QLabel("Training Studio")
        title.setObjectName("trainingTitle")
        subtitle = QLabel("Record YOLO runs against the selected four-class subset. Each run keeps its own results and checkpoints for comparison.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("trainingSubtitle")
        root.addWidget(eyebrow)
        root.addWidget(title)
        root.addWidget(subtitle)
        self.status = QLabel("Choose a four-class subset and an existing .pt file or an official model name.")
        self.status.setObjectName("trainingStatus")
        root.addWidget(self.status)

        source = QGroupBox("Inputs")
        source_layout = QGridLayout(source)
        self.subset_combo = QComboBox()
        self.subset_combo.setMinimumWidth(280)
        self.subset_combo.currentIndexChanged.connect(self._subset_changed)
        self.refresh_subset_button = QPushButton("Refresh subsets")
        self.refresh_subset_button.clicked.connect(self._refresh_subsets)
        self.weight_edit = QLineEdit()
        self.weight_edit.setPlaceholderText("Local .pt path or official name, e.g. yolo26n.pt")
        self.weight_edit.setToolTip("Type yolo26n.pt, yolo11s.pt, or yolov8n.pt to download once if missing; or browse a local .pt file.")
        self.weight_browse = QPushButton("Browse weights")
        self.weight_browse.clicked.connect(self._browse_weights)
        source_layout.addWidget(QLabel("Subset"), 0, 0)
        source_layout.addWidget(self.subset_combo, 0, 1, 1, 2)
        source_layout.addWidget(self.refresh_subset_button, 0, 3)
        source_layout.addWidget(QLabel("Base weights"), 1, 0)
        source_layout.addWidget(self.weight_edit, 1, 1, 1, 2)
        source_layout.addWidget(self.weight_browse, 1, 3)
        self.server_edit = QLineEdit(self.service.server_address())
        self.server_edit.setPlaceholderText("Training server IP, e.g. 192.168.1.228 (blank = train on this PC)")
        self.server_edit.setToolTip("With a valid server address, training runs on that server and its results are copied back here. Blank or invalid: training runs on this PC.")
        self.server_edit.editingFinished.connect(self._apply_server_address)
        self.server_test_button = QPushButton("Test server")
        self.server_test_button.clicked.connect(self._test_server)
        self._applied_server_text = self.server_edit.text().strip()
        source_layout.addWidget(QLabel("Server IP"), 2, 0)
        source_layout.addWidget(self.server_edit, 2, 1, 1, 2)
        source_layout.addWidget(self.server_test_button, 2, 3)
        root.addWidget(source)

        config = QGroupBox("Experiment configuration")
        form = QFormLayout(config)
        self.run_number = self._spin(1, 2_147_483_647, 1)
        self.run_number.setToolTip("Defaults to the next recorded run number; choose another positive number if needed.")
        self.epochs = self._spin(1, 1000, 10)
        self.image_size = self._spin(32, 2048, 640)
        self.batch = self._spin(1, 512, 8)
        self.seed = self._spin(0, 2_147_483_647, 0)
        self.workers = self._spin(0, 32, 0)
        self.patience = self._spin(0, 1000, 20)
        self.device = QLineEdit()
        self.device.setPlaceholderText("blank = Ultralytics default, e.g. 0 or cpu")
        self.amp = QCheckBox("Enable AMP (may download yolo26n.pt for checks)")
        self.amp.setChecked(False)
        self.amp.setToolTip("Off: FP32 still uses the selected GPU. On: Ultralytics may download yolo26n.pt for its AMP check.")
        form.addRow("Run number", self.run_number)
        form.addRow("Epochs", self.epochs)
        form.addRow("Image size", self.image_size)
        form.addRow("Batch", self.batch)
        form.addRow("Seed", self.seed)
        form.addRow("Workers", self.workers)
        form.addRow("Patience", self.patience)
        form.addRow("Device", self.device)
        form.addRow("Mixed precision", self.amp)
        root.addWidget(config)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Start training")
        self.start_button.setObjectName("trainingPrimary")
        self.start_button.clicked.connect(self._start_training)
        self.cancel_button = QPushButton("Cancel training")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_training)
        self.refresh_runs_button = QPushButton("Refresh past runs")
        self.refresh_runs_button.clicked.connect(self._refresh_runs)
        actions.addWidget(self.start_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.refresh_runs_button)
        actions.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        actions.addWidget(self.progress, 1)
        root.addLayout(actions)

        split = QHBoxLayout()
        left = QGroupBox("Recorded runs")
        left_layout = QVBoxLayout(left)
        self.runs_table = QTableWidget(0, 6)
        self.runs_table.setHorizontalHeaderLabels(["Run", "Number", "Status", "mAP50", "mAP50-95", "Started"])
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.itemSelectionChanged.connect(self._show_selected_run)
        left_layout.addWidget(self.runs_table)
        right = QGroupBox("Child-process log")
        right_layout = QVBoxLayout(right)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        right_layout.addWidget(self.log_view)
        split.addWidget(left, 1)
        split.addWidget(right, 1)
        root.addLayout(split, 1)
        self._apply_style()

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            #trainingPage { background: #f5f8f7; color: #17211d; }
            QLabel#trainingEyebrow { color: #17745d; font-size: 12px; font-weight: 800; }
            QLabel#trainingTitle { color: #153c2d; font-size: 28px; font-weight: 800; }
            QLabel#trainingSubtitle { color: #566b62; font-size: 13px; }
            QLabel#trainingStatus { background: #e7f4ed; color: #18583e; padding: 8px; border-radius: 6px; }
            QGroupBox { background: #ffffff; border: 1px solid #d6e2dc; border-radius: 7px; margin-top: 8px; padding: 10px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton, QComboBox, QLineEdit, QSpinBox { min-height: 27px; padding: 4px 7px; }
            QPushButton { background: #ffffff; color: #17304d; border: 1px solid #8fa5b5; border-radius: 5px; min-height: 32px; padding: 7px 13px; font-weight: 600; }
            QPushButton:hover { background: #e8f3f2; border-color: #087f75; }
            QPushButton#trainingPrimary { background: #17745d; color: white; font-weight: 800; }
            QTableWidget, QPlainTextEdit { background: white; border: 1px solid #d5e2dc; }
            """
        )

    def _subset_paths(self) -> list[Path]:
        return sorted(
            (path for path in self.workspace.resolve_inside("dataset").glob("subset-*") if (path / "manifest.json").is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def _refresh_subsets(self) -> None:
        current = self.subset_combo.currentData()
        self.subset_combo.clear()
        for path in self._subset_paths():
            try:
                validation = validate_subset(self.workspace, path)
                label = f"{path.name} | {', '.join(validation.class_names)}"
            except TrainingError as error:
                label = f"{path.name} | invalid: {error}"
            self.subset_combo.addItem(label, str(path))
        if current:
            index = self.subset_combo.findData(current)
            if index >= 0:
                self.subset_combo.setCurrentIndex(index)
        self._subset_changed()

    def _subset_changed(self, *_args: Any) -> None:
        raw = self.subset_combo.currentData()
        if not raw:
            self.status.setText("No complete four-class subset found under the candidate workspace.")
            return
        try:
            value = validate_subset(self.workspace, Path(str(raw)))
            self.status.setText(f"Subset ready: four classes, train/val/test preserved. Manifest {value.manifest_sha256[:12]}...")
        except TrainingError as error:
            self.status.setText(f"Subset unavailable: {error}")

    def _browse_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose local YOLO base weights", "", "PyTorch weights (*.pt)")
        if path:
            self.weight_edit.setText(path)

    def _apply_server_address(self, *, force: bool = False) -> str | None:
        text = self.server_edit.text().strip()
        changed = text != self._applied_server_text
        if not changed and not force:
            return self.service.remote_client.base_url if self.service.remote_client else None
        url, error = self.service.set_server_address(text)
        self._applied_server_text = text
        if error:
            self.status.setText(f"{error}. Training will run on this PC.")
        elif changed:
            self.status.setText(f"Training server set to {url}. Press Test server to check it." if url else "No training server set. Training will run on this PC.")
        return url

    def _test_server(self) -> None:
        url = self._apply_server_address(force=True)
        client = self.service.remote_client
        if url is None or client is None:
            if not self.server_edit.text().strip():
                self.status.setText("No training server set. Training will run on this PC.")
            return
        self.server_test_button.setEnabled(False)
        self.status.setText(f"Testing training server {url} ...")
        task = TrainingStartTask(lambda: (url, client.health()))
        task.signals.started.connect(self._server_test_passed)
        task.signals.failed.connect(self._server_test_failed)
        task.signals.finished.connect(self._server_test_finished)
        self.thread_pool.start(task)

    def _server_test_finished(self) -> None:
        self.server_test_button.setEnabled(self.start_button.isEnabled())

    def _server_test_passed(self, result: object) -> None:
        url, health = result if isinstance(result, tuple) else ("", {})
        value = health if isinstance(health, dict) else {}
        device = value.get("accelerator") or "server default"
        self.status.setText(
            f"Training server {url} is reachable ({device}; {value.get('running', 0)}/{value.get('max_concurrent', '?')} jobs running, "
            f"{value.get('queued', 0)} queued). Training will run on the server."
        )

    def _server_test_failed(self, message: str) -> None:
        self.status.setText(f"Server test failed: {message} Training will run on this PC until the server is reachable.")

    def _start_training(self) -> None:
        self._apply_server_address()
        raw_subset = self.subset_combo.currentData()
        raw_weight = self.weight_edit.text().strip()
        if not raw_subset or not raw_weight:
            self.status.setText("Choose a complete four-class subset and a local .pt file or official model name first.")
            return
        config = TrainingConfig(
            experiment_index=self.run_number.value(),
            epochs=self.epochs.value(),
            image_size=self.image_size.value(),
            batch=self.batch.value(),
            seed=self.seed.value(),
            device=self.device.text().strip(),
            amp=self.amp.isChecked(),
            workers=self.workers.value(),
            patience=self.patience.value(),
        )
        self._set_running(True)
        self.log_view.clear()
        self.progress.setValue(0)
        self.status.setText("Checking base weights (downloading an official model if needed), then starting training...")
        task = TrainingStartTask(
            lambda: self.service.start_training(
                Path(str(raw_subset)),
                Path(raw_weight),
                config,
                on_output=self.output_signal.emit,
                on_progress=lambda current, total: self.progress_signal.emit(current, total),
                on_complete=self.complete_signal.emit,
            )
        )
        task.signals.started.connect(self._run_started)
        task.signals.failed.connect(self._start_failed)
        self.thread_pool.start(task)

    def _run_started(self, run: object) -> None:
        self.current_run_id = getattr(run, "run_id", None)
        executor: dict[str, Any] = {}
        try:
            executor = self.service.load_run(str(self.current_run_id)).get("executor") or {}
        except TrainingError:
            pass
        if executor.get("kind") == "remote":
            self.status.setText(f"Training started on server {executor.get('server', '')}: {self.current_run_id}")
        elif executor.get("remote_fallback_reason"):
            self.status.setText(f"Training server unavailable; training started on this PC: {self.current_run_id}")
        else:
            self.status.setText(f"Training started on this PC: {self.current_run_id}")

    def _start_failed(self, message: str) -> None:
        self._set_running(False)
        self.status.setText(f"Training could not start: {message}")

    def _cancel_training(self) -> None:
        if self.current_run_id and self.service.cancel_active(self.current_run_id):
            self.status.setText("Cancellation requested; the child process tree is being released.")

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.refresh_subset_button.setEnabled(not running)
        self.refresh_runs_button.setEnabled(not running)
        self.server_edit.setEnabled(not running)
        self.server_test_button.setEnabled(not running)

    @Slot(str)
    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    @Slot(int, int)
    def _set_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(min(current, total))

    @Slot(object)
    def _run_complete(self, record: object) -> None:
        value = record if isinstance(record, dict) else {}
        self._set_running(False)
        self.current_run_id = None
        self._refresh_runs()
        self.status.setText(f"Training {value.get('status', 'finished')}: {value.get('run_id', '')} {value.get('error') or ''}".strip())

    def _refresh_runs(self) -> None:
        records = self.service.list_runs()
        used_numbers = [
            number for record in records
            if isinstance(number := record.get("experiment_index"), int) and not isinstance(number, bool) and number > 0
        ]
        self.run_number.setValue(min(max(used_numbers, default=0) + 1, self.run_number.maximum()))
        self.runs_table.setRowCount(0)
        for record in records:
            row = self.runs_table.rowCount()
            self.runs_table.insertRow(row)
            metrics = record.get("metrics", {}) or {}
            values = [
                str(record.get("run_id", "")),
                str(record.get("experiment_index", "")),
                str(record.get("status", "")),
                self._display_metric(metrics.get("map50")),
                self._display_metric(metrics.get("map50_95")),
                str(record.get("started_at", "")),
            ]
            for column, value in enumerate(values):
                self.runs_table.setItem(row, column, QTableWidgetItem(value))

    @staticmethod
    def _display_metric(value: Any) -> str:
        return "unavailable" if value is None else f"{float(value):.4f}"

    def _show_selected_run(self) -> None:
        selected = self.runs_table.selectedItems()
        if not selected:
            return
        run_id = selected[0].text()
        try:
            record = self.service.load_run(run_id)
            log_rel = record.get("paths", {}).get("log")
            log_path = self.workspace.resolve_inside(log_rel) if log_rel else None
            self.log_view.setPlainText(log_path.read_text(encoding="utf-8") if log_path and log_path.is_file() else "No log file recorded.")
            self.status.setText(f"Loaded recorded run {run_id}; status is {record.get('status', 'unknown')}.")
        except (OSError, TrainingError, ValueError) as error:
            self.status.setText(f"Could not reopen run: {error}")

    def shutdown(self) -> None:
        if self.current_run_id:
            self.service.cancel_active(self.current_run_id)
