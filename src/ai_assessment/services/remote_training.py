"""Run M6 training on an operator-provided LAN training server.

``RemoteTrainingClient.launch`` accepts the same argument list the local
child process would receive and returns a process-like object, so
``TrainingService`` keeps its run records, hashes, logs, and progress
handling unchanged. Artifacts are extracted into the exact local paths
Ultralytics would have written before ``wait()`` returns.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

SERVER_URL_ENV = "AI_TRAINING_SERVER_URL"
SERVER_TOKEN_ENV = "AI_TRAINING_SERVER_TOKEN"
DEFAULT_SERVER_PORT = 8766
_HOSTNAME = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*$")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
# Ultralytics label caches are machine-specific and change after each local scan.
_SKIPPED_DATASET_SUFFIXES = {".cache"}


class RemoteTrainingUnavailable(RuntimeError):
    """The server could not accept the job; local training remains available."""


class RemoteTrainingError(RuntimeError):
    """A dispatched remote job failed in a way the run record must show."""


def parse_server_address(text: str) -> str | None:
    """Return ``http://host:port`` for a typed server address, or None when blank.

    Accepts an IPv4 address or host name with an optional ``:port`` and an
    optional ``http://`` prefix. Raises ValueError for anything else.
    """
    value = text.strip()
    if not value:
        return None
    if value.lower().startswith("http://"):
        value = value[len("http://"):]
    value = value.rstrip("/")
    host, separator, port_text = value.partition(":")
    port = DEFAULT_SERVER_PORT
    if separator:
        if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
            raise ValueError(f"Invalid server port: {port_text or '(empty)'}")
        port = int(port_text)
    try:
        ipaddress.IPv4Address(host)
    except ValueError:
        if host.replace(".", "").isdigit() or not _HOSTNAME.match(host):
            raise ValueError(f"Invalid server address: {text.strip()}") from None
    return f"http://{host}:{port}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_archive(root: Path, target: Path) -> str:
    """Write a byte-stable archive of a dataset folder and return its SHA-256."""
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() in _SKIPPED_DATASET_SUFFIXES:
                continue
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=_ZIP_TIMESTAMP)
            info.external_attr = 0o644 << 16
            with path.open("rb") as source, bundle.open(info, "w") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
    return _sha256(target)


def _argument(args: list[str], flag: str) -> str | None:
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


@dataclass(frozen=True)
class RemoteTrainingClient:
    base_url: str
    token: str = ""
    candidate_id: str = ""
    client_id: str = ""
    connect_timeout: float = 3.0
    read_timeout: float = 120.0
    poll_wait_seconds: float = 15.0
    # How long a dispatched job may stay unreachable before the run is failed.
    disconnect_grace_seconds: float = 600.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RemoteTrainingClient | None":
        values = os.environ if environ is None else environ
        try:
            url = parse_server_address(values.get(SERVER_URL_ENV, ""))
        except ValueError:
            return None
        if url is None:
            return None
        return cls(url, values.get(SERVER_TOKEN_ENV, "").strip())

    def _request(self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any):
        import requests

        headers = dict(kwargs.pop("headers", {}) or {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.candidate_id:
            headers["X-Candidate-ID"] = self.candidate_id
        if self.client_id:
            headers["X-Client-ID"] = self.client_id
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            timeout=(self.connect_timeout, timeout if timeout is not None else self.read_timeout),
            **kwargs,
        )
        return response

    @staticmethod
    def _error_text(response: Any) -> str:
        try:
            return str(response.json()["error"]["message"])
        except Exception:
            return f"HTTP {response.status_code}"

    def health(self) -> dict[str, Any]:
        """Check the server; raises RemoteTrainingUnavailable when it cannot train."""
        import requests

        try:
            response = self._request("GET", "/api/v1/health", timeout=5)
        except requests.RequestException as error:
            raise RemoteTrainingUnavailable(
                f"Training server unreachable at {self.base_url}; check the address, the network, and that the server is running."
            ) from error
        if response.status_code != 200:
            raise RemoteTrainingUnavailable(f"Training server health check failed: {self._error_text(response)}")
        try:
            payload = response.json()
        except ValueError as error:
            raise RemoteTrainingUnavailable("The address answered, but it is not an AI Assessment training server.") from error
        if not isinstance(payload, dict) or not str(payload.get("api_version", "")).startswith("m6.remote-training."):
            raise RemoteTrainingUnavailable("The address answered, but it is not an AI Assessment training server.")
        if payload.get("auth") == "token" and not self.token:
            raise RemoteTrainingUnavailable("The training server requires a token (AI_TRAINING_SERVER_TOKEN).")
        return payload

    def _ensure_blob(self, path: Path, sha256: str) -> None:
        response = self._request("HEAD", f"/api/v1/blobs/{sha256}", timeout=10)
        if response.status_code == 200:
            return
        if response.status_code != 404:
            raise RemoteTrainingUnavailable(f"Blob check failed: {self._error_text(response)}")
        with path.open("rb") as handle:
            response = self._request("PUT", f"/api/v1/blobs/{sha256}", data=handle, timeout=600)
        if response.status_code != 200:
            raise RemoteTrainingUnavailable(f"Upload failed: {self._error_text(response)}")

    def launch(self, args: list[str]) -> "RemoteProcess":
        """Dispatch the run described by local training_runner arguments."""
        import requests

        weights = _argument(args, "--weights")
        data_yaml = _argument(args, "--data")
        project = _argument(args, "--project")
        name = _argument(args, "--name")
        result_json = _argument(args, "--result-json")
        if not all((weights, data_yaml, project, name, result_json)):
            raise RemoteTrainingUnavailable("Training arguments are incomplete for remote dispatch.")
        weights_path = Path(str(weights))
        data_path = Path(str(data_yaml))
        dataset_root = data_path.parent
        params = {
            "epochs": int(_argument(args, "--epochs") or 0),
            "imgsz": int(_argument(args, "--imgsz") or 0),
            "batch": int(_argument(args, "--batch") or 0),
            "seed": int(_argument(args, "--seed") or 0),
            "workers": int(_argument(args, "--workers") or 0),
            "patience": int(_argument(args, "--patience") or 0),
            "device": _argument(args, "--device") or "",
            "amp": "--amp" in args,
        }
        self.health()
        try:
            with tempfile.TemporaryDirectory(prefix="ai-assessment-upload-") as scratch:
                archive = Path(scratch) / "dataset.zip"
                dataset_sha = build_dataset_archive(dataset_root, archive)
                self._ensure_blob(archive, dataset_sha)
            weights_sha = _sha256(weights_path)
            self._ensure_blob(weights_path, weights_sha)
            payload = {
                "run_id": name,
                "dataset_sha256": dataset_sha,
                "weights_sha256": weights_sha,
                "weights_name": weights_path.name,
                "data_yaml": data_path.relative_to(dataset_root).as_posix(),
                "params": params,
                "client_paths": {
                    "sep": os.sep,
                    "weights": str(weights_path),
                    "data_root": str(dataset_root),
                    "project": str(project),
                    "result_dir": str(Path(str(result_json)).parent),
                },
            }
            response = self._request("POST", "/api/v1/jobs", json=payload, timeout=300)
        except requests.RequestException as error:
            raise RemoteTrainingUnavailable(f"Training server unreachable: {error}") from error
        if response.status_code != 200:
            raise RemoteTrainingUnavailable(f"Training server rejected the job: {self._error_text(response)}")
        job = response.json()
        return RemoteProcess(self, job, Path(str(project)), Path(str(result_json)).parent)


class RemoteProcess:
    """ProcessLike adapter over one server-side job."""

    # pid 0 keeps TrainingService from issuing taskkill for a local process tree.
    pid = 0
    is_remote = True

    def __init__(self, client: RemoteTrainingClient, job: Mapping[str, Any], project_dir: Path, result_dir: Path) -> None:
        self.client = client
        self.job_id = str(job["job_id"])
        self.environment = job.get("environment")
        self.project_dir = project_dir
        self.result_dir = result_dir
        self.stdout = _RemoteLogStream(self)
        self._status: Mapping[str, Any] = job
        self._return_code: int | None = None
        self._wait_lock = threading.Lock()
        self._cancel_requested = False
        self._lost_contact = False

    @property
    def executor_record(self) -> dict[str, Any]:
        return {"kind": "remote", "server": self.client.base_url, "job_id": self.job_id, "server_environment": self.environment}

    @staticmethod
    def _code_for(status: Mapping[str, Any]) -> int:
        code = status.get("return_code")
        if isinstance(code, int):
            return code
        return 1 if status.get("status") == "cancelled" else -1

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        """Retry transient network failures within the disconnect grace period."""
        import requests

        if self._lost_contact:
            raise RemoteTrainingError("Lost contact with the training server.")
        deadline = time.monotonic() + self.client.disconnect_grace_seconds
        delay = 1.0
        while True:
            try:
                response = self.client._request(method, path, **kwargs)
            except requests.RequestException as error:
                if time.monotonic() >= deadline:
                    self._lost_contact = True
                    raise RemoteTrainingError(f"Lost contact with the training server: {error}") from error
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
                continue
            if response.status_code >= 500 and time.monotonic() < deadline:
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
                continue
            if response.status_code != 200:
                raise RemoteTrainingError(f"Training server error: {RemoteTrainingClient._error_text(response)}")
            return response

    def iter_lines(self) -> Iterator[str]:
        offset = 0
        while True:
            response = self._call(
                "GET",
                f"/api/v1/jobs/{self.job_id}/logs",
                params={"offset": offset, "wait": self.client.poll_wait_seconds},
                timeout=self.client.poll_wait_seconds + 30,
            )
            payload = response.json()
            self._status = payload
            lines = payload.get("lines", [])
            yield from (str(line) for line in lines)
            offset = int(payload.get("next_offset", offset + len(lines)))
            if payload.get("status") in {"finished", "cancelled", "error"} and offset >= int(payload.get("line_count", offset)):
                return

    def poll(self) -> int | None:
        if self._return_code is not None:
            return self._return_code
        try:
            response = self.client._request("GET", f"/api/v1/jobs/{self.job_id}", timeout=10)
            if response.status_code == 200:
                self._status = response.json()
        except Exception:
            return None
        if self._status.get("status") in {"finished", "cancelled", "error"}:
            return self._code_for(self._status)
        return None

    def wait(self) -> int:
        with self._wait_lock:
            if self._return_code is not None:
                return self._return_code
            while self._status.get("status") not in {"finished", "cancelled", "error"}:
                response = self._call(
                    "GET",
                    f"/api/v1/jobs/{self.job_id}/logs",
                    params={"offset": 1 << 30, "wait": self.client.poll_wait_seconds},
                    timeout=self.client.poll_wait_seconds + 30,
                )
                self._status = response.json()
            self._download_artifacts()
            self._return_code = self._code_for(self._status)
            return self._return_code

    def _download_artifacts(self) -> None:
        import requests

        response = None
        try:
            response = self.client._request("GET", f"/api/v1/jobs/{self.job_id}/artifacts", timeout=600, stream=True)
        except requests.RequestException as error:
            raise RemoteTrainingError(f"Could not download training artifacts: {error}") from error
        if response.status_code == 404:
            return
        if response.status_code != 200:
            raise RemoteTrainingError(f"Could not download training artifacts: {RemoteTrainingClient._error_text(response)}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.result_dir, prefix=".artifacts-", suffix=".zip", delete=False) as handle:
            archive = Path(handle.name)
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
        try:
            targets = {"models": self.project_dir, "result": self.result_dir}
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    parts = PurePosixPath(member.filename).parts
                    if member.is_dir() or len(parts) < 2 or parts[0] not in targets or ".." in parts:
                        continue
                    root = targets[parts[0]].resolve()
                    destination = (root / Path(*parts[1:])).resolve()
                    if root not in destination.parents:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, destination.open("wb") as output:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(chunk)
        finally:
            archive.unlink(missing_ok=True)

    def terminate(self) -> None:
        if self._cancel_requested:
            return
        self._cancel_requested = True
        try:
            self.client._request("POST", f"/api/v1/jobs/{self.job_id}/cancel", timeout=30)
        except Exception:
            pass

    def kill(self) -> None:
        self.terminate()


class _RemoteLogStream:
    def __init__(self, process: RemoteProcess) -> None:
        self.process = process

    def __iter__(self) -> Iterator[str]:
        return self.process.iter_lines()

    def close(self) -> None:
        pass
