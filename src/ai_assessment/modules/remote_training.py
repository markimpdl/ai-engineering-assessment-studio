"""Remote cluster controls. All network calls run off the Qt GUI thread."""
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QSpinBox, QComboBox, QMessageBox
)
from ..services.remote_cluster import RemoteClusterClient
from ..services.training import validate_subset

class Signals(QObject):
    result = Signal(object)
    error = Signal(str)

class Action(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = Signals()
    @Slot()
    def run(self):
        try:
            self.signals.result.emit(self.fn())
        except Exception as exc:
            self.signals.error.emit(str(exc))

class RemoteTrainingPanel(QWidget):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self.pool = QThreadPool.globalInstance()
        self.busy = False
        self._download_path = None
        self._jobs_by_id = {}
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(5000)
        self.status_timer.timeout.connect(self._auto_refresh)
        self.url = QLineEdit(os.getenv("AI_GPU_MANAGER_URL", ""))
        self.url.setPlaceholderText("http://MANAGER-LAN-IP:8000")
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Cluster API key (trusted operator only)")
        self.dataset = QLineEdit(os.getenv("AI_GPU_DATASET", "dates-v16"))
        self.dataset.setToolTip("Filled after uploading an M3 four-class subset; pre-staged IDs also work.")
        self.source_run = QComboBox()
        self.source_run.addItem("Official YOLO base weights (new training)", None)
        self.source_run.currentIndexChanged.connect(self._source_changed)
        self.model = QComboBox()
        self.model.addItems(["yolo26n.pt", "yolo26s.pt"])
        self.epochs = QSpinBox(); self.epochs.setRange(1, 200); self.epochs.setValue(5)
        self.imgsz = QSpinBox(); self.imgsz.setRange(160, 640); self.imgsz.setSingleStep(32); self.imgsz.setValue(512)
        self.imgsz.setToolTip("Enter any size from 160 to 640. It rounds to the nearest multiple of 32 after editing.")
        self.imgsz.editingFinished.connect(self.normalize_image_size)
        self.batch = QSpinBox(); self.batch.setRange(1, 8); self.batch.setValue(4)
        self.subset = QComboBox()
        self.refresh_subsets()
        self.refresh_subset_button = QPushButton("Refresh subsets")
        self.refresh_subset_button.clicked.connect(self.refresh_subsets)
        self.upload_button = QPushButton("Upload selected subset")
        self.upload_button.clicked.connect(self.upload)
        layout = QVBoxLayout(self)
        note = QLabel("Direct GPU cluster training (trusted operator pilot). Upload a four-class Dataset Studio subset or use a pre-staged dataset ID.")
        note.setWordWrap(True); layout.addWidget(note)
        for caption, widget in (("Manager URL",self.url),("Cluster API key",self.token),("Dataset ID",self.dataset),
                                ("Model",self.model),("Starting weights",self.source_run),("Epochs",self.epochs),("Image size",self.imgsz),("Batch",self.batch)):
            line=QHBoxLayout(); line.addWidget(QLabel(caption)); line.addWidget(widget, 1); layout.addLayout(line)
        selection = QHBoxLayout()
        selection.addWidget(self.subset, 1)
        selection.addWidget(self.refresh_subset_button)
        selection.addWidget(self.upload_button)
        layout.addLayout(selection)
        buttons=QHBoxLayout()
        for caption, fn in (("Submit remote job", self.submit), ("Refresh jobs", self.refresh),
                            ("Cancel selected", self.cancel)):
            button=QPushButton(caption)
            button.clicked.connect(fn)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        results_row = QHBoxLayout()
        self.download_button = QPushButton("Download results ZIP")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download)
        self.open_folder_button = QPushButton("Open results folder")
        self.open_folder_button.clicked.connect(self.open_results_folder)
        results_row.addWidget(self.download_button)
        results_row.addWidget(self.open_folder_button)
        results_row.addStretch()
        layout.addLayout(results_row)
        self.download_hint = QLabel("Select a completed job with an artifact to download its results.")
        self.download_hint.setWordWrap(True)
        layout.addWidget(self.download_hint)
        self.status=QLabel("Ready");layout.addWidget(self.status)
        self.table=QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Job ID", "Status", "GPU node", "Model", "Start", "End", "Download"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        self.table.itemSelectionChanged.connect(self.update_download_state)
        self.table.setColumnWidth(6, 86)
        self.table.setToolTip("Start shows submission time on cluster v0.8; End shows terminal update time.")

    def normalize_image_size(self):
        """Round to nearest 32; half-way cases round upward."""
        number = self.imgsz.value()
        normalized = max(160, min(640, ((number + 16) // 32) * 32))
        if number != normalized:
            self.imgsz.setValue(normalized)

    def _auto_refresh(self):
        if self.isVisible() and not self.busy and self.url.text().strip() and self.token.text().strip():
            self.refresh()

    def refresh_subsets(self):
        """Re-read M3 exports; retain selection if it still exists."""
        previous = self.subset.currentData()
        self.subset.blockSignals(True)
        try:
            self.subset.clear()
            self.subset.addItem("Select exported Dataset Studio subset", None)
            for candidate in sorted(
                self.workspace.resolve_inside("dataset").glob("subset-*"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            ):
                try:
                    validated = validate_subset(self.workspace, candidate)
                except Exception:
                    continue
                self.subset.addItem(
                    f"{candidate.name} | {', '.join(validated.class_names)}",
                    str(candidate),
                )
            selected = self.subset.findData(previous) if previous else -1
            self.subset.setCurrentIndex(selected if selected >= 0 else (1 if self.subset.count() > 1 else 0))
        finally:
            self.subset.blockSignals(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_subsets()
        self.status_timer.start()
        self._auto_refresh()

    def hideEvent(self, event):
        self.status_timer.stop()
        super().hideEvent(event)

    def client(self):
        return RemoteClusterClient(self.url.text().strip(),self.token.text().strip(),
                                   self.workspace.session.candidate_id)

    def run_action(self, fn, callback):
        if self.busy: return
        self.busy=True
        task=Action(fn)
        task.signals.result.connect(self._finish)
        task.signals.error.connect(self._finish)
        task.signals.result.connect(callback)
        task.signals.error.connect(lambda err: self.status.setText("Error: "+err))
        self.pool.start(task)

    def _finish(self, *_):
        self.busy=False

    def upload(self):
        self.refresh_subsets()
        path = self.subset.currentData()
        if not path:
            self.status.setText("Choose an exported Dataset Studio subset")
            return
        try:
            validate_subset(self.workspace, Path(path))
            client = self.client()
        except Exception as exc:
            self.status.setText("Upload unavailable: " + str(exc))
            return
        self.status.setText("Uploading dataset...")
        self.run_action(lambda: client.upload_subset(Path(path)), self.upload_finished)

    def upload_finished(self, result):
        dataset_id = result.get("dataset_id")
        self.dataset.setText(dataset_id)
        self.status.setText("Dataset uploaded: " + dataset_id + ". Ready to train.")

    def submit(self):
        self.normalize_image_size()
        client=self.client()
        args=(self.model.currentText(),self.dataset.text().strip(),
              self.epochs.value(),self.imgsz.value(),self.batch.value(),
              (self.source_run.currentData() or {}).get("id"))
        self.run_action(lambda:client.submit(*args),lambda job:(self.status.setText("Submitted: "+str(job.get("id", job.get("job_id", "?")))),self.refresh()))

    def refresh(self):
        client=self.client()
        self.run_action(client.jobs,self.display)

    def _source_changed(self, *_):
        model = self.source_run.currentData()
        if isinstance(model, dict):
            index = self.model.findText(str(model.get("model", "")))
            if index >= 0:
                self.model.setCurrentIndex(index)
            self.model.setEnabled(False)
        else:
            self.model.setEnabled(True)

    @staticmethod
    def format_job_time(value):
        if value is None:
            return "—"
        try:
            return datetime.fromtimestamp(float(value)).strftime("%d/%m %H:%M")
        except (ValueError, TypeError, OverflowError, OSError):
            return "—"

    def display(self, jobs):
        previous_selected = self.selected_id()
        self._jobs_by_id = {str(job.get("id")): job for job in jobs if job.get("id")}
        previous = self.source_run.currentData()
        selected_id = previous.get("id") if isinstance(previous, dict) else None
        self.source_run.blockSignals(True)
        try:
            self.source_run.clear()
            self.source_run.addItem("Official YOLO base weights (new training)", None)
            for job in jobs:
                if job.get("status") == "completed" and job.get("mode") == "train" and job.get("artifact"):
                    self.source_run.addItem(
                        f"{str(job.get('id', ''))[:10]} | {job.get('model', '')} | {job.get('dataset', '')}",
                        {"id": job.get("id"), "model": job.get("model")},
                    )
            if selected_id:
                for idx in range(1, self.source_run.count()):
                    item = self.source_run.itemData(idx)
                    if isinstance(item, dict) and item.get("id") == selected_id:
                        self.source_run.setCurrentIndex(idx)
                        break
        finally:
            self.source_run.blockSignals(False)
        self._source_changed()
        self.table.setRowCount(0)
        selected_row = -1
        for job in jobs:
            row=self.table.rowCount();self.table.insertRow(row)
            for col,key in enumerate(("id","status","node","model")):
                self.table.setItem(row,col,QTableWidgetItem(str(job.get(key) or "")))
            ready = job.get("status") == "completed" and bool(job.get("artifact"))
            # v0.8 uses created_at (submission); future API may return started_at.
            start_time = job.get("started_at") or job.get("created_at")
            terminal = job.get("status") in ("completed", "failed", "cancelled")
            end_time = (job.get("finished_at") or job.get("updated_at")) if terminal else None
            self.table.setItem(row, 4, QTableWidgetItem(self.format_job_time(start_time)))
            self.table.setItem(row, 5, QTableWidgetItem(self.format_job_time(end_time)))
            download = QPushButton("↓")
            download.setToolTip("Download this job's ZIP results" if ready else "Results not yet available")
            download.setAccessibleName("Download job " + str(job.get("id", "")))
            download.setEnabled(ready)
            download.clicked.connect(lambda _checked=False, jid=str(job.get("id")): self._download_for_job(jid))
            self.table.setCellWidget(row, 6, download)
            if str(job.get("id")) == previous_selected:
                selected_row = row
        if selected_row >= 0:
            self.table.selectRow(selected_row)
        self.update_download_state()
        self.status.setText(f"{len(jobs)} remote jobs")

    def selected_id(self):
        row=self.table.currentRow()
        return self.table.item(row,0).text() if row>=0 and self.table.item(row,0) else None

    def cancel(self):
        job=self.selected_id()
        if job:
            client=self.client()
            self.run_action(lambda:client.cancel(job),lambda _:(self.status.setText("Cancellation requested"),self.refresh()))

    def update_download_state(self):
        job_id = self.selected_id()
        job = self._jobs_by_id.get(job_id) if job_id else None
        available = bool(job and job.get("status") == "completed" and job.get("artifact"))
        self.download_button.setEnabled(available and not self.busy)
        if not job:
            self.download_hint.setText("Select a completed job with an artifact to download its results.")
        elif available:
            self.download_hint.setText(
                f"Results available for {job_id[:12]}... Click Download results ZIP. "
                "The archive contains the files provided by the cluster; download does not automatically import them into Results Studio."
            )
        else:
            self.download_hint.setText(
                f"Selected job: {job.get('status', 'unknown')}. "
                "Results can be downloaded after the cluster has completed the job and uploaded its artifact."
            )

    def open_results_folder(self):
        folder = self.workspace.resolve_inside("exports", "remote_training")
        folder.mkdir(parents=True, exist_ok=True)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self.status.setText("Could not open folder: " + str(folder))

    def download(self):
        self._download_for_job(self.selected_id())

    def _download_for_job(self, job):
        record = self._jobs_by_id.get(job) if job else None
        if not record or record.get("status") != "completed" or not record.get("artifact"):
            self.update_download_state()
            return
        try:
            client = self.client()
        except Exception as exc:
            self.status.setText("Download unavailable: " + str(exc))
            return
        target = self.workspace.resolve_inside("exports", "remote_training", job + ".zip")
        self.status.setText("Downloading results for " + job[:12] + "...")
        self.download_button.setEnabled(False)
        self.run_action(lambda: client.download(job, target), self.download_finished)

    def download_finished(self, path):
        self._download_path = Path(path)
        self.status.setText("Results downloaded: " + str(path))
        self.update_download_state()
        self.download_hint.setText(
            "Saved results ZIP: " + str(path) + ". Click Open results folder to locate the archive."
        )
