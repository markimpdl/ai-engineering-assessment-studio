"""LAN training server: runs candidate YOLO training jobs on one shared machine.

The desktop keeps ownership of run records, hashes, and the four-experiment
rule. This server only executes ``ai_assessment.services.training_runner`` for
an authenticated candidate, streams its console output with server paths
rewritten to the candidate's own paths, and returns the run artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import yaml
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool


API_VERSION = "m6.remote-training.v1"
_RUN_ID = re.compile(r"^[A-Za-z0-9_\-]{1,120}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WEIGHTS_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,120}\.pt$")
_IDENTITY = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_CUDA_DEVICE = re.compile(r"^(cuda(:\d+)?|\d+(,\d+)*)$", re.IGNORECASE)
ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_STATUSES = frozenset({"finished", "cancelled", "error"})


class TrainingServerError(RuntimeError):
    status_code = 400
    code = "training_server_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AuthenticationError(TrainingServerError):
    status_code = 401
    code = "authentication_required"


class NotFoundError(TrainingServerError):
    status_code = 404
    code = "not_found"


class ConflictError(TrainingServerError):
    status_code = 409
    code = "conflict"


class InputLimitError(TrainingServerError):
    status_code = 413
    code = "input_limit_exceeded"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TrainingServerConfig:
    data_dir: Path
    # token -> candidate/student identifier
    students: dict[str, str]
    # Open mode trusts the LAN: clients identify themselves by candidate ID and
    # workspace session ID instead of presenting a token.
    open_mode: bool = False
    max_concurrent: int = 4
    python_executable: str = sys.executable
    # Used when the client sends a blank or CUDA device; "mps" on Apple silicon.
    accelerator: str = "mps"
    # The desktop default is workers=0; the shared server loads images faster with a few.
    default_workers: int = 2
    max_epochs: int = 1000
    max_image_size: int = 2048
    max_batch: int = 256
    max_blob_bytes: int = 20 * 1024**3
    max_job_seconds: float = 6 * 3600


@dataclass
class TrainingJob:
    job_id: str
    student: str
    run_id: str
    directory: Path
    args: list[str]
    translations: list[tuple[str, str, str]]
    status: str = "queued"
    return_code: int | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    ended_at: str | None = None
    lines: list[str] = field(default_factory=list)
    process: Any = None
    cancel_requested: bool = False
    condition: threading.Condition = field(default_factory=threading.Condition)

    def public(self, queue_position: int | None = None) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "status": self.status,
            "return_code": self.return_code,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "line_count": len(self.lines),
            "queue_position": queue_position,
        }


def build_path_translator(translations: Iterable[tuple[str, str, str]]) -> Callable[[str], str]:
    """Return a function replacing server path prefixes with client paths.

    Each translation is ``(server_prefix, client_prefix, client_separator)``.
    The remainder of a matched path has its separators converted so a
    Windows candidate sees ``C:\\...\\weights\\best.pt`` as it would locally.
    """

    pairs = sorted(((s, c, sep) for s, c, sep in translations if s), key=lambda item: len(item[0]), reverse=True)
    if not pairs:
        return lambda text: text
    lookup = {server: (client, sep) for server, client, sep in pairs}
    pattern = re.compile("(" + "|".join(re.escape(server) for server, _, _ in pairs) + r")([^\s'\"\x1b]*)")

    def translate(text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            client, sep = lookup[match.group(1)]
            tail = match.group(2)
            return client + (tail.replace("/", sep) if sep != "/" else tail)

        return pattern.sub(replace, text)

    return translate


def _translate_value(value: Any, translate: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return translate(value)
    if isinstance(value, list):
        return [_translate_value(item, translate) for item in value]
    if isinstance(value, dict):
        return {key: _translate_value(item, translate) for key, item in value.items()}
    return value


def _translate_file(path: Path, translate: Callable[[str], str]) -> bytes:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(raw.decode("utf-8"))
            return (json.dumps(_translate_value(data, translate), indent=2, sort_keys=True) + "\n").encode("utf-8")
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw.decode("utf-8"))
            return yaml.safe_dump(_translate_value(data, translate), sort_keys=False).encode("utf-8")
    except (UnicodeDecodeError, ValueError, yaml.YAMLError):
        return raw
    return raw


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise TrainingServerError(f"Dataset archive contains an unsafe path: {member.filename}")
            destination = (root / Path(*name.parts)).resolve()
            if destination != root and root not in destination.parents:
                raise TrainingServerError(f"Dataset archive escapes its folder: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("ultralytics", "torch", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _bounded_int(params: dict[str, Any], key: str, low: int, high: int) -> int:
    value = params.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise InputLimitError(f"Parameter {key} must be an integer between {low} and {high}.")
    return value


class BlobWriter:
    def __init__(self, target: Path, sha256: str, max_bytes: int) -> None:
        self.target = target
        self.sha256 = sha256
        self.max_bytes = max_bytes
        self.temporary = target.with_name(f"{sha256}.{uuid.uuid4().hex}.part")
        self.handle = self.temporary.open("wb")
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if self.size > self.max_bytes:
            raise InputLimitError("Upload exceeds the server blob size limit.")
        self.digest.update(chunk)
        self.handle.write(chunk)

    def commit(self) -> None:
        self.handle.close()
        if self.digest.hexdigest() != self.sha256:
            raise TrainingServerError("Uploaded content does not match its SHA-256 identifier.")
        os.replace(self.temporary, self.target)

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        self.temporary.unlink(missing_ok=True)


class TrainingServer:
    """Queue, run, and stream candidate training jobs with bounded concurrency."""

    def __init__(self, config: TrainingServerConfig, *, process_factory: Callable[..., Any] = subprocess.Popen) -> None:
        if config.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.config = config
        self.process_factory = process_factory
        self.data_dir = config.data_dir.expanduser().resolve()
        self.blob_dir = self.data_dir / "blobs"
        self.jobs_dir = self.data_dir / "jobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, TrainingJob] = {}
        self._queue: deque[str] = deque()
        self._running = 0
        self._load_finished_jobs()

    # -- identity -----------------------------------------------------------------
    def authenticate(self, authorization: str | None, candidate_id: str | None = None, client_id: str | None = None) -> str:
        presented = authorization[len("Bearer "):].strip() if authorization and authorization.startswith("Bearer ") else ""
        if presented:
            for token, student in self.config.students.items():
                if secrets.compare_digest(presented.encode(), token.encode()):
                    return student
        if self.config.open_mode:
            candidate = (candidate_id or "").strip()
            client = (client_id or "").strip()
            if _IDENTITY.match(candidate) and _IDENTITY.match(client):
                # Several PCs may share a candidate ID such as C014; the workspace
                # session ID keeps their jobs apart.
                return f"{candidate}_{client[:12]}"
            raise AuthenticationError("Open mode requires X-Candidate-ID and X-Client-ID headers or a registered token.")
        if not presented:
            raise AuthenticationError("A bearer token is required.")
        raise AuthenticationError("Unknown training token.")

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "api_version": API_VERSION,
                "auth": "open" if self.config.open_mode else "token",
                "accelerator": self.config.accelerator,
                "max_concurrent": self.config.max_concurrent,
                "running": self._running,
                "queued": len(self._queue),
            }

    def environment(self) -> dict[str, Any]:
        return {
            "python": sys.version,
            "platform": sys.platform,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "accelerator": self.config.accelerator,
            "packages": _package_versions(),
        }

    # -- blobs ----------------------------------------------------------------
    def _blob_path(self, sha256: str) -> Path:
        if not _SHA256.match(sha256):
            raise TrainingServerError("Blob identifiers must be lowercase SHA-256 hex digests.")
        return self.blob_dir / sha256

    def has_blob(self, sha256: str) -> bool:
        return self._blob_path(sha256).is_file()

    def open_blob(self, sha256: str) -> "BlobWriter | None":
        """Start a content-addressed upload, or return None if it already exists."""
        target = self._blob_path(sha256)
        if target.is_file():
            return None
        return BlobWriter(target, sha256, self.config.max_blob_bytes)

    def store_blob(self, sha256: str, chunks: Iterable[bytes]) -> bool:
        """Store a content-addressed upload. Returns False if it already existed."""
        writer = self.open_blob(sha256)
        if writer is None:
            return False
        try:
            for chunk in chunks:
                writer.write(chunk)
            writer.commit()
        finally:
            writer.abort()
        return True

    # -- jobs -----------------------------------------------------------------
    def _resolve_device(self, requested: str) -> str:
        value = requested.strip()
        if value.lower() == "cpu":
            return "cpu"
        if not value or value.lower() == "mps" or _CUDA_DEVICE.match(value):
            return self.config.accelerator
        raise InputLimitError(f"Unsupported training device: {requested}")

    def create_job(self, student: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = str(payload.get("run_id", ""))
        if not _RUN_ID.match(run_id):
            raise TrainingServerError("run_id is missing or contains unsupported characters.")
        dataset_sha = str(payload.get("dataset_sha256", ""))
        weights_sha = str(payload.get("weights_sha256", ""))
        weights_name = str(payload.get("weights_name", ""))
        if not _WEIGHTS_NAME.match(weights_name):
            raise TrainingServerError("weights_name must be a plain .pt file name.")
        data_yaml_name = str(payload.get("data_yaml", "data.yaml"))
        data_yaml_parts = PurePosixPath(data_yaml_name).parts
        if PurePosixPath(data_yaml_name).is_absolute() or ".." in data_yaml_parts or not data_yaml_name.endswith((".yaml", ".yml")):
            raise TrainingServerError("data_yaml must be a relative YAML path inside the dataset archive.")
        for sha in (dataset_sha, weights_sha):
            if not self.has_blob(sha):
                raise NotFoundError(f"Upload {sha} before creating the job.")
        params = payload.get("params")
        if not isinstance(params, dict):
            raise TrainingServerError("params must be an object.")
        epochs = _bounded_int(params, "epochs", 1, self.config.max_epochs)
        imgsz = _bounded_int(params, "imgsz", 32, self.config.max_image_size)
        batch = _bounded_int(params, "batch", -1, self.config.max_batch)
        seed = _bounded_int(params, "seed", 0, 2**31 - 1)
        workers = _bounded_int(params, "workers", 0, 64) or self.config.default_workers
        patience = _bounded_int(params, "patience", 0, 100000)
        device = self._resolve_device(str(params.get("device", "")))
        amp = bool(params.get("amp", False))
        client_paths = payload.get("client_paths")
        if not isinstance(client_paths, dict):
            raise TrainingServerError("client_paths must be an object.")
        separator = str(client_paths.get("sep", "/"))
        if separator not in {"/", "\\"}:
            raise TrainingServerError("client_paths.sep must be / or \\.")

        with self._lock:
            for job in self._jobs.values():
                if job.student == student and job.status in ACTIVE_STATUSES:
                    raise ConflictError(f"Training job {job.run_id} is already active for this candidate.")
                if job.student == student and job.run_id == run_id:
                    raise ConflictError(f"Run {run_id} already exists on the server.")
            job_id = uuid.uuid4().hex
            directory = self.jobs_dir / student / job_id
            directory.mkdir(parents=True, exist_ok=False)

        data_dir = directory / "data"
        project_dir = directory / "models"
        result_dir = directory / "result"
        weights_dir = directory / "weights"
        try:
            _safe_extract(self._blob_path(dataset_sha), data_dir)
            data_yaml = data_dir / Path(*data_yaml_parts)
            if not data_yaml.is_file():
                raise TrainingServerError(f"Dataset archive does not contain {data_yaml_name}.")
            weights_dir.mkdir()
            weights = weights_dir / weights_name
            try:
                os.link(self._blob_path(weights_sha), weights)
            except OSError:
                shutil.copyfile(self._blob_path(weights_sha), weights)
            project_dir.mkdir()
            result_dir.mkdir()
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

        args = [
            self.config.python_executable, "-X", "utf8", "-u", "-m", "ai_assessment.services.training_runner",
            "--weights", str(weights),
            "--data", str(data_yaml),
            "--project", str(project_dir),
            "--name", run_id,
            "--result-json", str(result_dir / "runner_result.json"),
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--seed", str(seed),
            "--workers", str(workers),
            "--patience", str(patience),
            "--device", device,
        ]
        if amp:
            args.append("--amp")

        translations: list[tuple[str, str, str]] = []
        for server_path, key in ((weights, "weights"), (data_dir, "data_root"), (project_dir, "project"), (result_dir, "result_dir")):
            client_value = client_paths.get(key)
            if not isinstance(client_value, str) or not client_value:
                continue
            for variant in {str(server_path), str(server_path.resolve())}:
                translations.append((variant, client_value, separator))

        job = TrainingJob(job_id, student, run_id, directory, args, translations)
        self._write_job_record(job)
        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
        self._schedule()
        return self.job_status(student, job_id) | {"environment": self.environment()}

    def _get(self, student: str, job_id: str) -> TrainingJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.student != student:
            raise NotFoundError("Training job not found.")
        return job

    def job_status(self, student: str, job_id: str) -> dict[str, Any]:
        job = self._get(student, job_id)
        with self._lock:
            position = list(self._queue).index(job_id) + 1 if job_id in self._queue else None
        return job.public(position)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at)
            queue = list(self._queue)
        return [job.public(queue.index(job.job_id) + 1 if job.job_id in queue else None) | {"student": job.student} for job in jobs]

    def read_logs(self, student: str, job_id: str, offset: int, wait_seconds: float = 0.0) -> dict[str, Any]:
        job = self._get(student, job_id)
        offset = max(0, offset)
        deadline = time.monotonic() + max(0.0, min(wait_seconds, 30.0))
        with job.condition:
            while len(job.lines) <= offset and job.status not in TERMINAL_STATUSES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                job.condition.wait(remaining)
            lines = job.lines[offset:offset + 5000]
        return {"lines": lines, "next_offset": offset + len(lines)} | self.job_status(student, job_id)

    def cancel(self, student: str, job_id: str) -> dict[str, Any]:
        job = self._get(student, job_id)
        with self._lock:
            job.cancel_requested = True
            if job.status == "queued":
                if job_id in self._queue:
                    self._queue.remove(job_id)
                self._finish(job, "cancelled", None, "Cancelled before the job started.")
                return self.job_status(student, job_id)
            process = job.process
        if process is not None and job.status == "running":
            self._stop_process(process)
        return self.job_status(student, job_id)

    def artifacts_path(self, student: str, job_id: str) -> Path:
        job = self._get(student, job_id)
        if job.status not in TERMINAL_STATUSES:
            raise ConflictError("Artifacts are available after the job ends.")
        archive = job.directory / "artifacts.zip"
        if not archive.is_file():
            raise NotFoundError("This job produced no artifacts.")
        return archive

    # -- execution -------------------------------------------------------------
    def _schedule(self) -> None:
        while True:
            with self._lock:
                if self._running >= self.config.max_concurrent or not self._queue:
                    return
                job = self._jobs[self._queue.popleft()]
                self._running += 1
                job.status = "running"
                job.started_at = utc_now_iso()
            threading.Thread(target=self._run, args=(job,), name=f"train-{job.job_id}", daemon=True).start()

    def _append(self, job: TrainingJob, line: str) -> None:
        with job.condition:
            job.lines.append(line)
            job.condition.notify_all()

    def _run(self, job: TrainingJob) -> None:
        translate = build_path_translator(job.translations)
        log_path = job.directory / "job.log"
        timer: threading.Timer | None = None
        return_code: int | None = None
        error: str | None = None
        try:
            kwargs: dict[str, Any] = {
                "args": job.args, "cwd": str(job.directory), "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1,
            }
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                kwargs["start_new_session"] = True
            process = self.process_factory(**kwargs)
            with self._lock:
                job.process = process
                cancel_now = job.cancel_requested
            if cancel_now:
                self._stop_process(process)
            timer = threading.Timer(self.config.max_job_seconds, self._time_limit, args=(job,))
            timer.daemon = True
            timer.start()
            with log_path.open("a", encoding="utf-8") as log:
                stream = process.stdout
                if stream is not None:
                    for raw_line in stream:
                        line = translate(str(raw_line).rstrip("\r\n"))
                        log.write(line + "\n")
                        self._append(job, line)
            return_code = process.wait()
        except Exception as exc:  # launch or supervision failure
            error = f"Training server could not run the job: {exc}"
            self._append(job, error)
            process = job.process
            if process is not None:
                self._stop_process(process)
        finally:
            if timer is not None:
                timer.cancel()
        try:
            self._build_artifacts(job, translate)
        except Exception as exc:
            error = error or f"Training server could not package artifacts: {exc}"
            self._append(job, error)
        if job.cancel_requested:
            status = "cancelled"
        elif error:
            status = "error"
        else:
            status = "finished"
        with self._lock:
            self._running -= 1
            self._finish(job, status, return_code, error)
        self._schedule()

    def _time_limit(self, job: TrainingJob) -> None:
        if job.status == "running" and job.process is not None:
            self._append(job, f"Training server time limit of {self.config.max_job_seconds:.0f} s reached; stopping the job.")
            job.error = "Training server time limit reached."
            self._stop_process(job.process)

    @staticmethod
    def _stop_process(process: Any) -> None:
        try:
            if process.poll() is not None:
                return
            pid = getattr(process, "pid", None)
            if os.name != "nt" and pid:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
            else:
                process.terminate()
            try:
                process.wait(timeout=15)
            except TypeError:
                process.wait()
            except subprocess.TimeoutExpired:
                if os.name != "nt" and pid:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    process.kill()
        except Exception:
            pass

    def _finish(self, job: TrainingJob, status: str, return_code: int | None, error: str | None) -> None:
        with job.condition:
            job.status = status
            job.return_code = return_code
            job.error = error or job.error
            job.ended_at = utc_now_iso()
            job.process = None
            job.condition.notify_all()
        self._write_job_record(job)

    def _build_artifacts(self, job: TrainingJob, translate: Callable[[str], str]) -> None:
        project_dir = job.directory / "models"
        result_dir = job.directory / "result"
        archive = job.directory / "artifacts.zip"
        temporary = archive.with_suffix(".zip.part")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for prefix, root in (("models", project_dir), ("result", result_dir)):
                if not root.is_dir():
                    continue
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    name = f"{prefix}/{path.relative_to(root).as_posix()}"
                    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
                        bundle.writestr(name, _translate_file(path, translate))
                    else:
                        bundle.write(path, name)
        os.replace(temporary, archive)

    def _write_job_record(self, job: TrainingJob) -> None:
        record = job.public() | {"student": job.student, "args": job.args}
        path = job.directory / "job.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _load_finished_jobs(self) -> None:
        """Reload jobs from an earlier server process so artifacts stay downloadable."""
        for path in sorted(self.jobs_dir.glob("*/*/job.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            job = TrainingJob(record["job_id"], record["student"], record["run_id"], path.parent, list(record.get("args", [])), [])
            job.created_at = record.get("created_at", job.created_at)
            job.started_at = record.get("started_at")
            job.ended_at = record.get("ended_at")
            job.return_code = record.get("return_code")
            job.error = record.get("error")
            job.status = record.get("status", "error")
            log_path = path.parent / "job.log"
            if log_path.is_file():
                job.lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if job.status not in TERMINAL_STATUSES:
                job.status = "error"
                job.error = "The training server restarted before this job finished."
                job.ended_at = utc_now_iso()
                job.lines.append(job.error)
                self._write_job_record(job)
            self._jobs[job.job_id] = job


def create_app(server: TrainingServer) -> FastAPI:
    app = FastAPI(title="AI Assessment Training Server", version=API_VERSION)

    def identity(
        authorization: str | None = Header(default=None),
        x_candidate_id: str | None = Header(default=None),
        x_client_id: str | None = Header(default=None),
    ) -> str:
        return server.authenticate(authorization, x_candidate_id, x_client_id)

    @app.exception_handler(TrainingServerError)
    async def training_error(_request: Request, exc: TrainingServerError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return server.health()

    @app.head("/api/v1/blobs/{sha256}")
    def blob_exists(sha256: str, student: str = Depends(identity)) -> None:
        if not server.has_blob(sha256):
            raise HTTPException(status_code=404)

    @app.put("/api/v1/blobs/{sha256}")
    async def put_blob(sha256: str, request: Request, student: str = Depends(identity)) -> dict[str, Any]:
        writer = await run_in_threadpool(server.open_blob, sha256)
        if writer is None:
            return {"sha256": sha256, "created": False}
        try:
            async for chunk in request.stream():
                if chunk:
                    await run_in_threadpool(writer.write, chunk)
            await run_in_threadpool(writer.commit)
        finally:
            writer.abort()
        return {"sha256": sha256, "created": True}

    @app.post("/api/v1/jobs")
    def create_job(payload: dict[str, Any] = Body(...), student: str = Depends(identity)) -> dict[str, Any]:
        return server.create_job(student, payload)

    @app.get("/api/v1/jobs/{job_id}")
    def job_status(job_id: str, student: str = Depends(identity)) -> dict[str, Any]:
        return server.job_status(student, job_id)

    @app.get("/api/v1/jobs/{job_id}/logs")
    def job_logs(job_id: str, offset: int = 0, wait: float = 0.0, student: str = Depends(identity)) -> dict[str, Any]:
        return server.read_logs(student, job_id, offset, wait)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, student: str = Depends(identity)) -> dict[str, Any]:
        return server.cancel(student, job_id)

    @app.get("/api/v1/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str, student: str = Depends(identity)) -> FileResponse:
        path = server.artifacts_path(student, job_id)
        return FileResponse(path, media_type="application/zip", filename=f"{job_id}.zip")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        rows = "".join(
            f"<tr><td>{job['student']}</td><td>{job['run_id']}</td><td>{job['status']}</td>"
            f"<td>{job['queue_position'] or ''}</td><td>{job['started_at'] or ''}</td><td>{job['ended_at'] or ''}</td>"
            f"<td>{'' if job['return_code'] is None else job['return_code']}</td></tr>"
            for job in reversed(server.list_jobs())
        )
        state = server.health()
        return (
            "<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=5><title>Training Server</title>"
            "<style>body{font-family:system-ui;margin:24px;color:#153c2d}table{border-collapse:collapse;width:100%}"
            "td,th{border-bottom:1px solid #d5e3dc;padding:6px 8px;text-align:left;font-size:14px}</style>"
            f"<h1>Training Server</h1><p>Running {state['running']} of {state['max_concurrent']} &middot; queued {state['queued']}</p>"
            "<table><tr><th>Candidate</th><th>Run</th><th>Status</th><th>Queue</th><th>Started</th><th>Ended</th><th>Exit</th></tr>"
            f"{rows}</table>"
        )

    return app
