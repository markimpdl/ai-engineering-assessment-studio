from __future__ import annotations

import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import uvicorn
from PySide6.QtWidgets import QApplication

from ai_assessment.core.workspace import CandidateWorkspace
from ai_assessment.modules.training import TrainingPage
from ai_assessment.services.remote_training import (
    SERVER_TOKEN_ENV,
    SERVER_URL_ENV,
    RemoteTrainingClient,
    RemoteTrainingUnavailable,
    build_dataset_archive,
    parse_server_address,
)
from ai_assessment.services.training import TrainingConfig, TrainingService, sha256_file
from server.training_server import (
    AuthenticationError,
    ConflictError,
    InputLimitError,
    NotFoundError,
    TrainingServer,
    TrainingServerConfig,
    TrainingServerError,
    build_path_translator,
    create_app,
)


class FakeRunner:
    """Simulates training_runner on the server: prints server paths and writes artifacts."""

    def __init__(self, args: list[str], *, block: bool = False) -> None:
        self.args = args
        self.pid = None
        self.terminated = False
        self.release = threading.Event()
        if not block:
            self.release.set()
        project = Path(self._arg("--project"))
        name = self._arg("--name")
        result_json = Path(self._arg("--result-json"))
        data_root = Path(self._arg("--data")).parent
        (project / name / "weights").mkdir(parents=True)
        (project / name / "weights" / "best.pt").write_bytes(b"remote best checkpoint")
        (project / name / "args.yaml").write_text(f"data: {data_root}/data.yaml\nproject: {project}\n", encoding="utf-8")
        (project / f"{name}_val").mkdir()
        (project / f"{name}_val" / "confusion_matrix.png").write_bytes(b"png")
        result_json.write_text(json.dumps({
            "metrics": {"metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.6},
            "per_class": [],
            "confusion_matrix": str(project / f"{name}_val" / "confusion_matrix.png"),
        }), encoding="utf-8")
        self.lines = [
            f"Resolved dataset root: {data_root}\n",
            "Epoch 1/2\n",
            "Epoch 2/2\n",
            f"Results saved to {project / name}\n",
        ]
        self.stdout = self._stream()

    def _arg(self, flag: str) -> str:
        return self.args[self.args.index(flag) + 1]

    def _stream(self):
        yield from self.lines
        self.release.wait(5)

    def poll(self) -> int | None:
        return (1 if self.terminated else 0) if self.release.is_set() else None

    def wait(self, timeout: float | None = None) -> int:
        self.release.wait(5)
        return 1 if self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True
        self.release.set()

    def kill(self) -> None:
        self.terminate()


class LiveServer:
    def __init__(self, server: TrainingServer) -> None:
        self.uvicorn = uvicorn.Server(uvicorn.Config(create_app(server), host="127.0.0.1", port=0, log_level="warning"))
        self.thread = threading.Thread(target=self.uvicorn.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.uvicorn.started and time.monotonic() < deadline:
            time.sleep(0.02)
        port = self.uvicorn.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    def __exit__(self, *_exc) -> None:
        self.uvicorn.should_exit = True
        self.thread.join(10)


class M6RemoteTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.workspace = CandidateWorkspace.create(base / "C014", candidate_id="C014")
        self.subset = self.workspace.root / "dataset" / "subset-test"
        for split in ("train", "valid", "test"):
            (self.subset / split / "images").mkdir(parents=True)
            (self.subset / split / "labels").mkdir(parents=True)
            (self.subset / split / "images" / "a.jpg").write_bytes(b"jpg")
            (self.subset / split / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (self.subset / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 4\nnames: [Ajwa, Galaxy, Medjool, Meneifi]\n",
            encoding="utf-8",
        )
        (self.subset / "manifest.json").write_text(json.dumps({
            "schema_version": "m3.dataset.v1",
            "status": "complete",
            "selected_order": [4, 8, 9, 12],
            "selected_class_mapping": {
                "4": {"new_id": 0, "name": "Ajwa"},
                "8": {"new_id": 1, "name": "Galaxy"},
                "9": {"new_id": 2, "name": "Medjool"},
                "12": {"new_id": 3, "name": "Meneifi"},
            },
        }), encoding="utf-8")
        self.weights = base / "base.pt"
        self.weights.write_bytes(b"local test weights")
        self.runners: list[FakeRunner] = []
        self.block_runner = False
        self.server_dir = base / "server"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def server_factory(self, **kwargs) -> FakeRunner:
        runner = FakeRunner(list(kwargs["args"]), block=self.block_runner)
        self.runners.append(runner)
        return runner

    def make_server(self, **overrides) -> TrainingServer:
        values = {"data_dir": self.server_dir, "students": {"token-C014": "C014", "token-C015": "C015"}, "accelerator": "mps"}
        values.update(overrides)
        return TrainingServer(TrainingServerConfig(**values), process_factory=self.server_factory)

    def local_factory_must_not_run(self, **_kwargs):
        raise AssertionError("local training must not start while the server accepts the job")

    def wait_idle(self, service: TrainingService) -> None:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and service.active_run_id is not None:
            time.sleep(0.02)
        self.assertIsNone(service.active_run_id)

    def test_remote_run_matches_local_record_paths_logs_and_progress(self) -> None:
        server = self.make_server()
        with LiveServer(server) as url:
            service = TrainingService(self.workspace, process_factory=self.local_factory_must_not_run, remote_client=RemoteTrainingClient(url, "token-C014", poll_wait_seconds=0.5))
            lines: list[str] = []
            progress: list[tuple[int, int]] = []
            run = service.start_training(self.subset, self.weights, TrainingConfig(experiment_index=1, epochs=2), on_output=lines.append, on_progress=lambda a, b: progress.append((a, b)))
            self.wait_idle(service)
        record = service.load_run(run.run_id)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["executor"]["kind"], "remote")
        self.assertEqual(record["executor"]["server_environment"]["accelerator"], "mps")
        best = run.model_dir / "weights" / "best.pt"
        self.assertEqual(best.read_bytes(), b"remote best checkpoint")
        self.assertIn({"path": f"models/{run.run_id}/weights/best.pt", "sha256": sha256_file(best), "bytes": best.stat().st_size}, record["checkpoints"])
        self.assertEqual(record["metrics"]["map50"], 0.8)
        self.assertEqual(Path(record["metrics"]["confusion_matrix"]), run.model_dir.parent / f"{run.run_id}_val" / "confusion_matrix.png")
        self.assertIn(f"Resolved dataset root: {self.subset}", lines)
        self.assertIn(f"Results saved to {run.model_dir}", lines)
        self.assertFalse(any(str(self.server_dir.resolve()) in line for line in lines))
        self.assertEqual(progress[-1], (2, 2))
        self.assertIn(str(self.subset), (run.model_dir / "args.yaml").read_text(encoding="utf-8"))
        self.assertEqual(self.runners[0].args[self.runners[0].args.index("--device") + 1], "mps")
        self.assertEqual(self.runners[0].args[self.runners[0].args.index("--workers") + 1], "2")
        self.assertIn(f"Results saved to {run.model_dir}", run.log_path.read_text(encoding="utf-8"))

    def test_unreachable_server_falls_back_to_local_training(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]
        local_runs: list[list[str]] = []

        def local_factory(**kwargs):
            args = list(kwargs["args"])
            local_runs.append(args)
            return FakeRunner(args)

        service = TrainingService(self.workspace, process_factory=local_factory, remote_client=RemoteTrainingClient(f"http://127.0.0.1:{closed_port}", "token-C014", connect_timeout=0.5))
        run = service.start_training(self.subset, self.weights, TrainingConfig(experiment_index=1, epochs=2))
        self.wait_idle(service)
        record = service.load_run(run.run_id)
        self.assertEqual(len(local_runs), 1)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["executor"]["kind"], "local")
        self.assertIn("unreachable", record["executor"]["remote_fallback_reason"])

    def test_remote_cancellation_records_cancelled_run_without_blocking(self) -> None:
        self.block_runner = True
        server = self.make_server()
        with LiveServer(server) as url:
            service = TrainingService(self.workspace, process_factory=self.local_factory_must_not_run, remote_client=RemoteTrainingClient(url, "token-C014", poll_wait_seconds=0.5))
            run = service.start_training(self.subset, self.weights, TrainingConfig(experiment_index=1))
            deadline = time.monotonic() + 5
            while not self.runners and time.monotonic() < deadline:
                time.sleep(0.02)
            started = time.monotonic()
            self.assertTrue(service.cancel_active(run.run_id))
            self.assertLess(time.monotonic() - started, 2)
            self.wait_idle(service)
        self.assertEqual(service.load_run(run.run_id)["status"], "cancelled")
        self.assertTrue(self.runners[0].terminated)

    def test_authentication_isolation_and_one_active_job_per_candidate(self) -> None:
        self.block_runner = True
        server = self.make_server(max_concurrent=1)
        with self.assertRaises(AuthenticationError):
            server.authenticate("Bearer wrong")
        dataset_sha, weights_sha = self.upload(server)
        first = server.create_job("C014", self.job_payload("run_a", dataset_sha, weights_sha))
        with self.assertRaises(ConflictError):
            server.create_job("C014", self.job_payload("run_b", dataset_sha, weights_sha))
        second = server.create_job("C015", self.job_payload("run_c", dataset_sha, weights_sha))
        self.assertEqual(second["status"], "queued")
        self.assertEqual(second["queue_position"], 1)
        with self.assertRaises(NotFoundError):
            server.job_status("C015", first["job_id"])
        self.assertEqual(server.cancel("C015", second["job_id"])["status"], "cancelled")
        server.cancel("C014", first["job_id"])
        deadline = time.monotonic() + 5
        while server.job_status("C014", first["job_id"])["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(server.job_status("C014", first["job_id"])["status"], "cancelled")

    def test_parameter_bounds_device_mapping_and_blob_integrity(self) -> None:
        server = self.make_server()
        with self.assertRaises(TrainingServerError):
            server.store_blob("0" * 64, [b"not matching"])
        self.assertFalse(server.has_blob("0" * 64))
        dataset_sha, weights_sha = self.upload(server)
        payload = self.job_payload("run_a", dataset_sha, weights_sha)
        payload["params"]["epochs"] = 10**6
        with self.assertRaises(InputLimitError):
            server.create_job("C014", payload)
        payload = self.job_payload("run_a", dataset_sha, weights_sha)
        payload["params"]["device"] = "../../etc"
        with self.assertRaises(InputLimitError):
            server.create_job("C014", payload)
        payload = self.job_payload("../escape", dataset_sha, weights_sha)
        with self.assertRaises(TrainingServerError):
            server.create_job("C014", payload)
        self.assertEqual(server._resolve_device("0"), "mps")
        self.assertEqual(server._resolve_device("cpu"), "cpu")

    def test_windows_path_translation_and_stable_dataset_archive(self) -> None:
        translate = build_path_translator([("/srv/jobs/x/models", r"C:\ws\C014\models", "\\")])
        self.assertEqual(translate("Results saved to \x1b[1m/srv/jobs/x/models/run_1/weights\x1b[0m"), "Results saved to \x1b[1mC:\\ws\\C014\\models\\run_1\\weights\x1b[0m")
        first = build_dataset_archive(self.subset, Path(self.temp_dir.name) / "one.zip")
        (self.subset / "train" / "labels.cache").write_bytes(b"machine specific")
        time.sleep(0.01)
        second = build_dataset_archive(self.subset, Path(self.temp_dir.name) / "two.zip")
        self.assertEqual(first, second)

    def make_workspace(self, name: str) -> CandidateWorkspace:
        workspace = CandidateWorkspace.create(Path(self.temp_dir.name) / name, candidate_id="C014")
        target = workspace.root / "dataset" / "subset-test"
        import shutil

        shutil.copytree(self.subset, target)
        return workspace

    def test_open_mode_keeps_pcs_sharing_a_candidate_id_apart(self) -> None:
        self.block_runner = True
        server = self.make_server(students={}, open_mode=True)
        with self.assertRaises(AuthenticationError):
            server.authenticate(None)
        with self.assertRaises(AuthenticationError):
            server.authenticate(None, "C014", "../x")
        with LiveServer(server) as url, patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SERVER_TOKEN_ENV, None)
            services = []
            for name in ("pc1", "pc2"):
                workspace = self.make_workspace(name)
                service = TrainingService(workspace, process_factory=self.local_factory_must_not_run)
                self.assertEqual(service.set_server_address(url.removeprefix("http://")), (url, None))
                service.remote_client = RemoteTrainingClient(url, candidate_id="C014", client_id=workspace.session.session_id, poll_wait_seconds=0.5)
                service.start_training(workspace.root / "dataset" / "subset-test", self.weights, TrainingConfig(experiment_index=1))
                services.append(service)
            deadline = time.monotonic() + 5
            while len(self.runners) < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(self.runners), 2)
            self.assertEqual(server.health()["running"], 2)
            self.assertEqual(len({job["student"] for job in server.list_jobs()}), 2)
            for service in services:
                service.cancel_active()
            for service in services:
                self.wait_idle(service)

    def test_server_address_setting_selects_server_or_this_pc(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SERVER_URL_ENV, None)
            os.environ.pop(SERVER_TOKEN_ENV, None)
            service = TrainingService(self.workspace)
            self.assertIsNone(service.remote_client)
            self.assertEqual(service.set_server_address(" 192.168.1.228 "), ("http://192.168.1.228:8766", None))
            client = TrainingService(self.workspace).remote_client
            self.assertEqual(client.base_url, "http://192.168.1.228:8766")
            self.assertEqual(client.candidate_id, "C014")
            self.assertEqual(client.client_id, self.workspace.session.session_id)
            self.assertEqual(TrainingService(self.workspace).server_address(), "192.168.1.228")
            url, error = service.set_server_address("999.1.1.1")
            self.assertIsNone(url)
            self.assertIn("Invalid server address", error)
            self.assertIsNone(service.remote_client)
            self.assertIsNone(TrainingService(self.workspace).remote_client)
            self.assertEqual(service.set_server_address(""), (None, None))
            self.assertIsNone(TrainingService(self.workspace).remote_client)
        self.assertEqual(parse_server_address("http://mac.local:9000/"), "http://mac.local:9000")
        for invalid in ("192.168.1", "a b", "1.2.3.4:0", "1.2.3.4:x"):
            with self.assertRaises(ValueError):
                parse_server_address(invalid)

    def test_health_requires_training_server_and_token_when_configured(self) -> None:
        with LiveServer(self.make_server()) as url:
            self.assertEqual(RemoteTrainingClient(url, "token-C014").health()["auth"], "token")
            with self.assertRaises(RemoteTrainingUnavailable):
                RemoteTrainingClient(url).health()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed = probe.getsockname()[1]
        with self.assertRaisesRegex(RemoteTrainingUnavailable, "unreachable"):
            RemoteTrainingClient(f"http://127.0.0.1:{closed}", connect_timeout=0.5).health()

    def test_training_page_server_field_and_test_button(self) -> None:
        app = QApplication.instance() or QApplication([])
        server = self.make_server(students={}, open_mode=True)
        with LiveServer(server) as url, patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SERVER_URL_ENV, None)
            os.environ.pop(SERVER_TOKEN_ENV, None)
            page = TrainingPage(self.workspace)
            self.assertEqual(page.server_edit.text(), "")
            page.server_edit.setText("not an ip!")
            page.server_test_button.click()
            self.assertIn("Training will run on this PC", page.status.text())
            self.assertIsNone(page.service.remote_client)
            page.server_edit.setText(url.removeprefix("http://"))
            page.server_test_button.click()
            deadline = time.monotonic() + 10
            while "reachable" not in page.status.text() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            self.assertIn(f"Training server {url} is reachable (mps; 0/4 jobs running", page.status.text())
            reopened = TrainingPage(self.workspace)
            self.assertEqual(reopened.server_edit.text(), url.removeprefix("http://"))
            reopened.deleteLater()
            page.server_edit.setText("")
            page.server_test_button.click()
            self.assertEqual(page.status.text(), "No training server set. Training will run on this PC.")
            self.assertIsNone(page.service.remote_client)
            page.deleteLater()

    def upload(self, server: TrainingServer) -> tuple[str, str]:
        archive = Path(self.temp_dir.name) / "dataset.zip"
        dataset_sha = build_dataset_archive(self.subset, archive)
        server.store_blob(dataset_sha, [archive.read_bytes()])
        weights_sha = sha256_file(self.weights)
        server.store_blob(weights_sha, io.BytesIO(self.weights.read_bytes()))
        return dataset_sha, weights_sha

    @staticmethod
    def job_payload(run_id: str, dataset_sha: str, weights_sha: str) -> dict:
        return {
            "run_id": run_id,
            "dataset_sha256": dataset_sha,
            "weights_sha256": weights_sha,
            "weights_name": "base.pt",
            "data_yaml": "data.yaml",
            "params": {"epochs": 2, "imgsz": 640, "batch": 8, "seed": 0, "workers": 0, "patience": 20, "device": "", "amp": False},
            "client_paths": {"sep": "/", "project": "/client/models", "result_dir": "/client/results"},
        }


if __name__ == "__main__":
    unittest.main()
