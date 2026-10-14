"""M6 training validation, process supervision, run records, and metrics."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import yaml

from ..core.workspace import CandidateWorkspace
from .remote_training import SERVER_TOKEN_ENV, SERVER_URL_ENV, RemoteTrainingClient, parse_server_address
from .vision import model_sha256, validate_model_path


TRAINING_SCHEMA_VERSION = "m6.training.v1"
RUN_RECORD_NAME = "run.json"
SERVER_SETTINGS_NAME = "training_server.json"
_PROGRESS_PATTERN = re.compile(r"(?:Epoch\s+)?(\d+)\s*/\s*(\d+)", re.IGNORECASE)

# Official Ultralytics detection checkpoints accepted by name. Keep this list
# narrow: arbitrary paths and URLs must never trigger a network request.
OFFICIAL_DETECTION_MODELS = frozenset(
    f"{family}{size}.pt"
    for family in ("yolo26", "yolo11", "yolov8")
    for size in "nsmlx"
)
OFFICIAL_WEIGHTS_REPO = "ultralytics/assets"
_OFFICIAL_DOWNLOAD_LOCK = threading.Lock()


class TrainingError(ValueError):
    """Raised when a run cannot be validated or completed safely."""


class TrainingBusyError(TrainingError):
    pass


class TrainingCancelled(TrainingError):
    pass


@dataclass(frozen=True)
class DatasetValidation:
    subset_root: Path
    data_yaml: Path
    manifest_path: Path
    manifest_sha256: str
    class_names: tuple[str, ...]
    split_paths: dict[str, str]


@dataclass(frozen=True)
class TrainingConfig:
    experiment_index: int
    epochs: int = 10
    image_size: int = 640
    batch: int = 8
    seed: int = 0
    device: str = ""
    workers: int = 0
    patience: int = 20
    run_label: str = ""
    resume: bool = False
    amp: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_index": self.experiment_index,
            "epochs": self.epochs,
            "image_size": self.image_size,
            "batch": self.batch,
            "seed": self.seed,
            "device": self.device,
            "workers": self.workers,
            "patience": self.patience,
            "run_label": self.run_label,
            "resume": self.resume,
            "amp": self.amp,
        }


@dataclass(frozen=True)
class MetricsSummary:
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    latency_ms: float | None = None
    per_class: tuple[dict[str, Any], ...] = ()
    loss_curve: tuple[dict[str, Any], ...] = ()
    confusion_matrix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "map50": self.map50,
            "map50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "latency_ms": self.latency_ms,
            "per_class": [dict(item) for item in self.per_class],
            "loss_curve": [dict(item) for item in self.loss_curve],
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True)
class TrainingRun:
    run_id: str
    record_path: Path
    log_path: Path
    model_dir: Path
    result_dir: Path
    status: str


class ProcessLike(Protocol):
    pid: int
    stdout: Any

    def poll(self) -> int | None: ...
    def wait(self) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _metric(metrics: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            value = _float_or_none(metrics[key])
            if value is not None:
                return value
    return None


def _path_inside(root: Path, candidate: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.expanduser().resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise TrainingError(f"{label} escapes the candidate workspace: {resolved}")
    return resolved


def validate_subset(workspace: CandidateWorkspace, subset_root: Path) -> DatasetValidation:
    """Validate an M3 export without changing the source or subset."""

    dataset_root = workspace.resolve_inside("dataset")
    root = _path_inside(dataset_root, subset_root, "Dataset subset")
    if not root.is_dir():
        raise TrainingError(f"Dataset subset does not exist: {root}")
    data_yaml = root / "data.yaml"
    manifest_path = root / "manifest.json"
    if not data_yaml.is_file() or not manifest_path.is_file():
        raise TrainingError("An M3 subset must contain data.yaml and manifest.json.")
    try:
        yaml_data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise TrainingError(f"Could not read the subset metadata: {error}") from error
    if manifest.get("status") != "complete":
        raise TrainingError("The dataset manifest is not marked complete.")
    selected_order = manifest.get("selected_order")
    mapping = manifest.get("selected_class_mapping")
    if not isinstance(selected_order, list) or len(selected_order) != 4 or len(set(selected_order)) != 4:
        raise TrainingError("The M3 manifest must contain exactly four selected classes in order.")
    if not isinstance(mapping, dict) or set(mapping) != {str(value) for value in selected_order}:
        raise TrainingError("The M3 class mapping is incomplete or inconsistent.")
    names_raw = yaml_data.get("names")
    names = tuple(str(value) for value in names_raw) if isinstance(names_raw, list) else ()
    if len(names) != 4:
        raise TrainingError("Training requires exactly four class names in data.yaml.")
    new_ids = [mapping[str(original)].get("new_id") for original in selected_order]
    if new_ids != [0, 1, 2, 3]:
        raise TrainingError("The M3 class mapping must remap selected classes to IDs 0 through 3.")
    mapped_names = [str(mapping[str(original)].get("name", "")) for original in selected_order]
    if tuple(mapped_names) != names:
        raise TrainingError("data.yaml names do not match the M3 manifest mapping.")

    split_paths: dict[str, str] = {}
    for split in ("train", "val", "test"):
        raw_value = yaml_data.get(split)
        if not isinstance(raw_value, str):
            raise TrainingError(f"data.yaml is missing the {split} split path.")
        image_dir = _path_inside(root, root / raw_value, f"{split} split")
        if not image_dir.is_dir():
            raise TrainingError(f"The {split} image directory does not exist: {image_dir}")
        label_dir = image_dir.parent / "labels"
        if not label_dir.is_dir():
            raise TrainingError(f"The {split} label directory does not exist: {label_dir}")
        split_paths[split] = raw_value
    return DatasetValidation(
        root,
        data_yaml,
        manifest_path,
        sha256_file(manifest_path),
        names,
        split_paths,
    )


def validate_base_weights(workspace: CandidateWorkspace, path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".pt":
        raise TrainingError("Training base weights must be a .pt file.")
    if not resolved.is_file():
        raise TrainingError(f"Base-weight file does not exist: {resolved}")
    if resolved.stat().st_size == 0:
        raise TrainingError(f"Base-weight file is empty: {resolved}")
    return resolved


def _download_official_weights(target: Path) -> Path:
    """Use Ultralytics' release asset downloader without loading a model."""
    from ultralytics.utils.downloads import attempt_download_asset

    return Path(attempt_download_asset(target, repo=OFFICIAL_WEIGHTS_REPO))


def resolve_base_weights(
    workspace: CandidateWorkspace,
    requested: Path,
    *,
    on_output: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a local .pt or cache an official detection checkpoint by name."""
    raw = str(requested).strip()
    path = Path(raw).expanduser()
    if path.is_file():
        return validate_base_weights(workspace, path), {"origin": "local_file", "requested": raw}
    if path.parent != Path(".") or path.name not in OFFICIAL_DETECTION_MODELS:
        raise TrainingError(
            f"Base weights not found: {raw}. Choose an existing local .pt file or "
            "enter an official detection model name such as yolo26n.pt."
        )
    target = workspace.resolve_inside("models", "base_weights", path.name)
    downloaded = False
    # Serialize cache writes across concurrent Training starts in this process.
    with _OFFICIAL_DOWNLOAD_LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            if on_output:
                on_output(f"Downloading official Ultralytics model {path.name} to {target} ...")
            try:
                actual = _download_official_weights(target).expanduser().resolve()
            except Exception as error:
                raise TrainingError(
                    f"Could not download {path.name} from official Ultralytics assets. "
                    "Allow GitHub release downloads on the router, or browse an existing local .pt file. "
                    f"Details: {error}"
                ) from error
            if actual != target or not target.is_file():
                raise TrainingError(f"Official model download did not produce the expected file: {target}")
            downloaded = True
        resolved = validate_base_weights(workspace, target)
    if on_output:
        on_output(f"{'Downloaded' if downloaded else 'Using cached'} model: {resolved}")
    return resolved, {
        "origin": "ultralytics_official_download" if downloaded else "workspace_cache",
        "requested": path.name,
        "source_repo": OFFICIAL_WEIGHTS_REPO,
        "downloaded_this_run": downloaded,
    }


def parse_metrics(result_dir: Path, runner_result: Path | None = None, *additional_dirs: Path) -> MetricsSummary:
    payload: dict[str, Any] = {}
    if runner_result and runner_result.is_file():
        try:
            loaded = json.loads(runner_result.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, ValueError):
            payload = {}
    raw_metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
    map50 = _metric(raw_metrics, "metrics/mAP50(B)", "metrics/mAP50", "map50")
    map50_95 = _metric(raw_metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95")
    precision = _metric(raw_metrics, "metrics/precision(B)", "metrics/precision", "precision")
    recall = _metric(raw_metrics, "metrics/recall(B)", "metrics/recall", "recall")
    latency = _metric(payload, "inference_latency_ms", "latency_ms")

    loss_curve: list[dict[str, Any]] = []
    search_dirs = (result_dir, *additional_dirs)
    csv_paths = sorted({path for directory in search_dirs if directory.is_dir() for path in directory.rglob("results.csv")})
    if csv_paths:
        try:
            with csv_paths[0].open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    cleaned = {key.strip(): value.strip() for key, value in row.items() if key}
                    loss_curve.append(cleaned)
        except (OSError, csv.Error):
            loss_curve = []
    payload_curve = payload.get("loss_curve")
    if not loss_curve and isinstance(payload_curve, list):
        loss_curve = [dict(row) for row in payload_curve if isinstance(row, dict)]
    per_class_raw = payload.get("per_class", [])
    per_class = tuple(dict(item) for item in per_class_raw if isinstance(item, dict)) if isinstance(per_class_raw, list) else ()
    confusion = payload.get("confusion_matrix")
    confusion_path = None
    if isinstance(confusion, str) and confusion:
        candidate = Path(confusion)
        if not candidate.is_absolute():
            candidate = result_dir / candidate
        if candidate.is_file():
            confusion_path = str(candidate)
    return MetricsSummary(map50, map50_95, precision, recall, latency, per_class, tuple(loss_curve), confusion_path)


def _package_versions() -> dict[str, str | None]:
    packages = ("ultralytics", "torch", "numpy", "opencv-python", "PySide6")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _checkpoint_hashes(workspace: CandidateWorkspace, model_dir: Path) -> list[dict[str, Any]]:
    hashes: list[dict[str, Any]] = []
    if not model_dir.is_dir():
        return hashes
    for path in sorted(model_dir.rglob("*.pt")):
        hashes.append({
            "path": str(path.relative_to(workspace.root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return hashes


def _safe_json_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class TrainingService:
    """Own one workspace's supervised child-process training lifecycle."""

    _workspace_jobs: dict[Path, "TrainingJob"] = {}
    _jobs_lock = threading.RLock()

    def __init__(
        self,
        workspace: CandidateWorkspace,
        *,
        python_executable: str | None = None,
        process_factory: Callable[..., ProcessLike] = subprocess.Popen,
        remote_client: RemoteTrainingClient | None = None,
    ) -> None:
        self.workspace = workspace
        self.python_executable = python_executable or sys.executable
        self.process_factory = process_factory
        # A configured LAN server runs the job when reachable; an injected
        # process factory (tests) keeps execution local.
        if remote_client is None and process_factory is subprocess.Popen:
            try:
                remote_client = self._client_for(parse_server_address(self.server_address()))
            except ValueError:
                remote_client = None
        self.remote_client = remote_client

    def _server_settings_path(self) -> Path:
        return self.workspace.resolve_inside("results", SERVER_SETTINGS_NAME)

    def server_address(self) -> str:
        """The saved training server address, else the operator environment default."""
        path = self._server_settings_path()
        if path.is_file():
            try:
                return str(json.loads(path.read_text(encoding="utf-8")).get("address", ""))
            except (OSError, ValueError, AttributeError):
                return ""
        return os.environ.get(SERVER_URL_ENV, "")

    def set_server_address(self, text: str) -> tuple[str | None, str | None]:
        """Save the address and return (server URL or None for this PC, validation error)."""
        error: str | None = None
        try:
            url = parse_server_address(text)
        except ValueError as exc:
            url, error = None, str(exc)
        path = self._server_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _safe_json_write(path, {"address": text.strip(), "server_url": url, "updated_at": utc_now_iso()})
        self.remote_client = self._client_for(url)
        self.workspace.record_event("training.server.configured", {"address": text.strip(), "server_url": url, "valid": error is None})
        return url, error

    def _client_for(self, url: str | None) -> RemoteTrainingClient | None:
        if url is None:
            return None
        return RemoteTrainingClient(
            url,
            os.environ.get(SERVER_TOKEN_ENV, "").strip(),
            candidate_id=self.workspace.session.candidate_id,
            client_id=self.workspace.session.session_id,
        )

    @property
    def active_run_id(self) -> str | None:
        with self._jobs_lock:
            job = self._workspace_jobs.get(self.workspace.root)
        return job.run_id if job is not None else None

    def _records_root(self) -> Path:
        root = self.workspace.resolve_inside("results", "training")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def list_runs(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        root = self._records_root()
        for path in sorted(root.glob("*/run.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict):
                    records.append(record)
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda record: str(record.get("started_at", "")))

    def load_run(self, run_id: str) -> dict[str, Any]:
        path = self.workspace.resolve_inside("results", "training", run_id, RUN_RECORD_NAME)
        if not path.is_file():
            raise TrainingError(f"Training run not found: {run_id}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise TrainingError(f"Training run record is unreadable: {run_id}") from error

    def _config_fingerprint(self, config: TrainingConfig, dataset: DatasetValidation, weight_hash: str) -> str:
        encoded = json.dumps({"config": config.to_dict(), "manifest": dataset.manifest_sha256, "weights": weight_hash}, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _build_args(self, config: TrainingConfig, dataset: DatasetValidation, weight_path: Path, model_dir: Path, runner_result: Path) -> list[str]:
        args = [
            self.python_executable,
            "-X",
            "utf8",
            "-u",
            "-m",
            "ai_assessment.services.training_runner",
            "--weights",
            str(weight_path),
            "--data",
            str(dataset.data_yaml),
            "--project",
            str(model_dir.parent),
            "--name",
            model_dir.name,
            "--result-json",
            str(runner_result),
            "--epochs",
            str(config.epochs),
            "--imgsz",
            str(config.image_size),
            "--batch",
            str(config.batch),
            "--seed",
            str(config.seed),
            "--workers",
            str(config.workers),
            "--patience",
            str(config.patience),
        ]
        if config.device.strip():
            args.extend(["--device", config.device.strip()])
        if config.amp:
            args.append("--amp")
        return args

    def start_training(
        self,
        subset_root: Path,
        weight_path: Path,
        config: TrainingConfig,
        *,
        on_output: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        on_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> TrainingRun:
        if config.resume:
            raise TrainingError("Resuming is not a new distinct experiment; reopen the existing run instead.")
        if config.experiment_index not in {1, 2, 3, 4}:
            raise TrainingError("Choose experiment 1, 2, 3, or 4.")
        dataset = validate_subset(self.workspace, subset_root)
        with self._jobs_lock:
            if self.workspace.root in self._workspace_jobs:
                raise TrainingBusyError("A training job is already active in this workspace.")
            existing_records = self.list_runs()
            if any(record.get("status") == "running" for record in existing_records):
                raise TrainingBusyError("A previous training job is still recorded as running; inspect or recover it before starting another.")
            if sum(record.get("status") == "completed" for record in existing_records) >= 4:
                raise TrainingError("This workspace already contains four completed experiment runs.")
        weight, weight_source = resolve_base_weights(self.workspace, weight_path, on_output=on_output)
        weight_hash = model_sha256(weight)
        if weight_source.get("downloaded_this_run"):
            self.workspace.record_event(
                "training.base_weights.downloaded",
                {"model_name": weight_source["requested"], "source_repo": OFFICIAL_WEIGHTS_REPO,
                 "path": str(weight.relative_to(self.workspace.root)).replace("\\", "/"),
                 "sha256": weight_hash, "bytes": weight.stat().st_size},
                artifact_hashes={"base_weights": weight_hash},
            )
        with self._jobs_lock:
            workspace_key = self.workspace.root
            if workspace_key in self._workspace_jobs:
                raise TrainingBusyError("A training job is already active in this workspace.")
            records = self.list_runs()
            if any(record.get("status") == "running" for record in records):
                raise TrainingBusyError("A previous training job is still recorded as running; inspect or recover it before starting another.")
            completed_runs = [record for record in records if record.get("status") == "completed"]
            if len(completed_runs) >= 4:
                raise TrainingError("This workspace already contains four completed experiment runs.")
            fingerprint = self._config_fingerprint(config, dataset, weight_hash)
            if any(record.get("config_fingerprint") == fingerprint for record in completed_runs):
                raise TrainingError("This experiment configuration was already completed; change a parameter before starting another run.")
            run_id = f"run_{config.experiment_index:02d}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
            model_dir = self.workspace.resolve_inside("models", run_id)
            result_dir = self.workspace.resolve_inside("results", "training", run_id)
            log_path = self.workspace.resolve_inside("logs", "training", f"{run_id}.log")
            runner_result = result_dir / "runner_result.json"
            # Let Ultralytics create the exact run directory with exist_ok=False;
            # pre-creating it would cause an automatic name increment.
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=False)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            args = self._build_args(config, dataset, weight, model_dir, runner_result)
            record = {
                "schema_version": TRAINING_SCHEMA_VERSION,
                "run_id": run_id,
                "experiment_index": config.experiment_index,
                "status": "running",
                "outcome": "running",
                "config": config.to_dict(),
                "config_fingerprint": fingerprint,
                "dataset": {
                    "subset_root": str(dataset.subset_root.relative_to(self.workspace.root)).replace("\\", "/"),
                    "data_yaml": str(dataset.data_yaml.relative_to(self.workspace.root)).replace("\\", "/"),
                    "manifest": str(dataset.manifest_path.relative_to(self.workspace.root)).replace("\\", "/"),
                    "manifest_sha256": dataset.manifest_sha256,
                    "class_names": list(dataset.class_names),
                    "split_paths": dataset.split_paths,
                },
                "base_weights": {"path": str(weight), "sha256": weight_hash, "bytes": weight.stat().st_size, **weight_source},
                "environment": {"python": sys.version, "platform": sys.platform, "packages": _package_versions()},
                "process_args": args,
                "paths": {
                    "model_dir": str(model_dir.relative_to(self.workspace.root)).replace("\\", "/"),
                    "result_dir": str(result_dir.relative_to(self.workspace.root)).replace("\\", "/"),
                    "log": str(log_path.relative_to(self.workspace.root)).replace("\\", "/"),
                    "runner_result": str(runner_result.relative_to(self.workspace.root)).replace("\\", "/"),
                },
                "started_at": utc_now_iso(),
                "ended_at": None,
                "metrics": MetricsSummary().to_dict(),
                "checkpoints": [],
                "error": None,
            }
            record_path = result_dir / RUN_RECORD_NAME
            _safe_json_write(record_path, record)
            self.workspace.record_event("training.started", {"run_id": run_id, "experiment_index": config.experiment_index, "config": config.to_dict(), "dataset_manifest_sha256": dataset.manifest_sha256, "base_weight_sha256": weight_hash})
            try:
                process, executor = self._dispatch(run_id, args)
            except Exception as error:
                record["status"] = "failed"
                record["outcome"] = "failed"
                record["ended_at"] = utc_now_iso()
                record["error"] = f"Could not start training process: {error}"
                _safe_json_write(record_path, record)
                self.workspace.record_event("training.failed", {"run_id": run_id, "error": str(error)}, outcome="failed")
                raise TrainingError(record["error"]) from error
            record["executor"] = executor
            _safe_json_write(record_path, record)
            job = TrainingJob(run_id, process, record, record_path, log_path, on_output, on_progress, on_complete)
            self._workspace_jobs[workspace_key] = job
            job.thread = threading.Thread(target=self._monitor, args=(job,), name=f"training-{run_id}", daemon=True)
            job.thread.start()
            return TrainingRun(run_id, record_path, log_path, model_dir, result_dir, "running")

    def _dispatch(self, run_id: str, args: list[str]) -> tuple[ProcessLike, dict[str, Any]]:
        """Prefer the configured training server; fall back to this PC silently."""
        if self.remote_client is None:
            return self._launch(args), {"kind": "local"}
        try:
            process = self.remote_client.launch(args)
        except Exception as error:  # any dispatch failure keeps the candidate training locally
            self.workspace.record_event("training.remote.fallback", {"run_id": run_id, "server": self.remote_client.base_url, "reason": str(error)})
            return self._launch(args), {"kind": "local", "remote_server": self.remote_client.base_url, "remote_fallback_reason": str(error)}
        executor = process.executor_record
        self.workspace.record_event("training.remote.dispatched", {"run_id": run_id, "server": executor["server"], "job_id": executor["job_id"]})
        return process, executor

    def _launch(self, args: list[str]) -> ProcessLike:
        # The Windows locale can be cp1252 while YOLO prints UTF-8/ANSI output.
        # Replace unexpected bytes so a log line cannot abort process supervision.
        kwargs: dict[str, Any] = {
            "args": args,
            "cwd": str(self.workspace.root),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return self.process_factory(**kwargs)

    def _monitor(self, job: "TrainingJob") -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with job.log_path.open("a", encoding="utf-8") as log:
                stream = job.process.stdout
                if stream is not None:
                    try:
                        for raw_line in stream:
                            line = str(raw_line).rstrip("\r\n")
                            job.last_lines.append(line)
                            del job.last_lines[:-8]
                            log.write(line + "\n")
                            log.flush()
                            if job.on_output:
                                job.on_output(line)
                            match = _PROGRESS_PATTERN.search(line)
                            if match and job.on_progress:
                                job.on_progress(int(match.group(1)), int(match.group(2)))
                    finally:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()
                return_code = job.process.wait()
            self._finalize(job, return_code)
        except Exception as error:
            detail = f"Training monitor failed: {error}"
            try:
                if job.process.poll() is None:
                    if os.name == "nt" and getattr(job.process, "pid", None):
                        subprocess.run(
                            ["taskkill", "/PID", str(job.process.pid), "/T", "/F"],
                            capture_output=True,
                            check=False,
                        )
                    if job.process.poll() is None:
                        job.process.terminate()
                    try:
                        job.process.wait()
                    except Exception:
                        job.process.kill()
            except Exception as stop_error:
                detail += f"; could not stop child process: {stop_error}"
            self._finalize(job, -1, detail)

    def _finalize(self, job: "TrainingJob", return_code: int, monitor_error: str | None = None) -> None:
        record = job.record
        result_dir = self.workspace.resolve_inside("results", "training", job.run_id)
        model_dir = self.workspace.resolve_inside("models", job.run_id)
        runner_result = result_dir / "runner_result.json"
        cancelled = job.cancel_requested
        checkpoints = _checkpoint_hashes(self.workspace, model_dir)
        metrics = parse_metrics(result_dir, runner_result, model_dir)
        if cancelled:
            status, outcome = "cancelled", "cancelled"
            error = "Training process cancelled by the candidate."
            event_type = "training.cancelled"
        elif return_code != 0 or monitor_error:
            status, outcome = "failed", "failed"
            detail = job.last_lines[-1] if job.last_lines else "See the recorded training log."
            error = monitor_error or f"Training process exited with code {return_code}. Last output: {detail}"
            event_type = "training.failed"
        elif not checkpoints:
            status, outcome = "failed", "failed"
            error = "Training finished without a checkpoint artifact."
            event_type = "training.failed"
        else:
            status, outcome, error = "completed", "completed", None
            event_type = "training.completed"
        record.update({"status": status, "outcome": outcome, "ended_at": utc_now_iso(), "metrics": metrics.to_dict(), "checkpoints": checkpoints, "error": error, "return_code": return_code})
        _safe_json_write(job.record_path, record)
        payload = {"run_id": job.run_id, "status": status, "metrics": metrics.to_dict(), "checkpoints": checkpoints}
        if error:
            payload["error"] = error
        self.workspace.record_event(event_type, payload, outcome=outcome)
        if status == "completed":
            self.workspace.record_event("training.metrics.completed", {"run_id": job.run_id, "metrics": metrics.to_dict()})
        with self._jobs_lock:
            self._workspace_jobs.pop(self.workspace.root, None)
        if job.on_complete:
            job.on_complete(record)

    def cancel_active(self, run_id: str | None = None) -> bool:
        with self._jobs_lock:
            job = self._workspace_jobs.get(self.workspace.root)
        if job is None or (run_id is not None and job.run_id != run_id):
            return False
        job.cancel_requested = True
        self.workspace.record_event("training.cancellation.requested", {"run_id": job.run_id})
        try:
            if os.name == "nt" and getattr(job.process, "pid", None):
                subprocess.run(["taskkill", "/PID", str(job.process.pid), "/T", "/F"], capture_output=True, check=False)
                if job.process.poll() is None:
                    job.process.terminate()
            else:
                job.process.terminate()
            # A remote job finishes and downloads its artifacts in the monitor
            # thread; waiting here would block the caller on the network.
            if not getattr(job.process, "is_remote", False):
                try:
                    job.process.wait()
                except Exception:
                    job.process.kill()
        except Exception as error:
            self.workspace.record_event("training.cancellation.error", {"run_id": job.run_id, "error": str(error)}, outcome="failed")
        return True

    def select_production_model(self, run_id: str, checkpoint_path: Path | None = None) -> Path:
        record = self.load_run(run_id)
        if record.get("status") != "completed":
            raise TrainingError("Only a completed run can provide the production model.")
        checkpoints = record.get("checkpoints", [])
        if not isinstance(checkpoints, list) or not checkpoints:
            raise TrainingError("The completed run has no checkpoint hashes.")
        selected = checkpoint_path
        if selected is None:
            preferred = [item for item in checkpoints if str(item.get("path", "")).lower().endswith("weights/best.pt")]
            selected = self.workspace.root / str((preferred[0] if preferred else checkpoints[0])["path"])
        selected = _path_inside(self.workspace.root, selected, "Selected checkpoint")
        if not selected.is_file() or selected.suffix.lower() != ".pt":
            raise TrainingError("The selected checkpoint must be a workspace-owned .pt file.")
        validate_model_path(selected)
        digest = sha256_file(selected)
        matching = next((item for item in checkpoints if item.get("path") == str(selected.relative_to(self.workspace.root)).replace("\\", "/")), None)
        if matching is None or matching.get("sha256") != digest:
            raise TrainingError("Selected checkpoint hash does not match the recorded run artifact.")
        selection = {"schema_version": "m6.model-selection.v1", "run_id": run_id, "path": str(selected.relative_to(self.workspace.root)).replace("\\", "/"), "sha256": digest, "selected_at": utc_now_iso(), "candidate_visible": True}
        selection_path = self.workspace.resolve_inside("results", "selected_model.json")
        _safe_json_write(selection_path, selection)
        self.workspace.record_event("training.model.selected", selection, artifact_hashes={"checkpoint": digest})
        return selected


class TrainingJob:
    def __init__(self, run_id: str, process: ProcessLike, record: dict[str, Any], record_path: Path, log_path: Path, on_output: Callable[[str], None] | None, on_progress: Callable[[int, int], None] | None, on_complete: Callable[[dict[str, Any]], None] | None) -> None:
        self.run_id = run_id
        self.process = process
        self.record = record
        self.record_path = record_path
        self.log_path = log_path
        self.on_output = on_output
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.cancel_requested = False
        self.last_lines: list[str] = []
        self.thread: threading.Thread | None = None
