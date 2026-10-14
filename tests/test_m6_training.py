from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from ai_assessment.core.workspace import CandidateWorkspace
from ai_assessment.services.training import (
    TrainingConfig,
    TrainingError,
    TrainingService,
    parse_metrics,
    resolve_base_weights,
    sha256_file,
    validate_subset,
)


class FakeStream:
    def __init__(self, lines: list[str], release: threading.Event | None = None) -> None:
        self.lines = lines
        self.release = release

    def __iter__(self):
        for line in self.lines:
            yield line
        if self.release is not None:
            self.release.wait(2)


class FakeProcess:
    def __init__(self, args: list[str], return_code: int = 0, block: bool = False) -> None:
        self.args = args
        self.pid = 4242
        self.return_code = return_code
        self.terminated = False
        self.release = threading.Event() if block else None
        self.stdout = FakeStream(["Epoch 1/2", "Epoch 2/2"], self.release)

        if block:
            self.stdout = FakeStream([], self.release)

    def poll(self) -> int | None:
        return self.return_code if self.terminated or self.release is None else None

    def wait(self) -> int:
        if self.release is not None:
            self.release.wait(2)
        return 1 if self.terminated else self.return_code

    def terminate(self) -> None:
        self.terminated = True
        if self.release is not None:
            self.release.set()

    def kill(self) -> None:
        self.terminate()


class M6TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = CandidateWorkspace.create(Path(self.temp_dir.name) / "C014", candidate_id="C014")
        self.subset = self.workspace.root / "dataset" / "subset-test"
        for split in ("train", "valid", "test"):
            (self.subset / split / "images").mkdir(parents=True)
            (self.subset / split / "labels").mkdir(parents=True)
        (self.subset / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 4\nnames: [Ajwa, Galaxy, Medjool, Meneifi]\n",
            encoding="utf-8",
        )
        (self.subset / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "m3.dataset.v1",
                    "status": "complete",
                    "selected_order": [4, 8, 9, 12],
                    "selected_class_mapping": {
                        "4": {"new_id": 0, "name": "Ajwa"},
                        "8": {"new_id": 1, "name": "Galaxy"},
                        "9": {"new_id": 2, "name": "Medjool"},
                        "12": {"new_id": 3, "name": "Meneifi"},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.weights = Path(self.temp_dir.name) / "base.pt"
        self.weights.write_bytes(b"local test weights")
        self.processes: list[FakeProcess] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def factory(self, **kwargs) -> FakeProcess:
        args = list(kwargs["args"])
        process = FakeProcess(args)
        project = Path(args[args.index("--project") + 1])
        name = args[args.index("--name") + 1]
        result_json = Path(args[args.index("--result-json") + 1])
        (project / name / "weights").mkdir(parents=True, exist_ok=True)
        (project / name / "weights" / "best.pt").write_bytes(b"best checkpoint")
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(
            json.dumps(
                {
                    "metrics": {
                        "metrics/mAP50(B)": 0.91,
                        "metrics/mAP50-95(B)": 0.73,
                        "metrics/precision(B)": 0.88,
                        "metrics/recall(B)": 0.86,
                    },
                    "per_class": [{"class_name": "Ajwa", "map50": 0.9}],
                }
            ),
            encoding="utf-8",
        )
        self.processes.append(process)
        return process

    def wait_for_completion(self, service: TrainingService) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and service.active_run_id is not None:
            time.sleep(0.01)
        self.assertIsNone(service.active_run_id)

    def test_subset_validation_and_weight_requirements(self) -> None:
        validated = validate_subset(self.workspace, self.subset)
        self.assertEqual(validated.class_names, ("Ajwa", "Galaxy", "Medjool", "Meneifi"))
        with self.assertRaises(TrainingError):
            validate_subset(self.workspace, Path(self.temp_dir.name) / "outside")
        with self.assertRaises(TrainingError):
            TrainingService(self.workspace).start_training(
                self.subset,
                Path(self.temp_dir.name) / "wrong.onnx",
                TrainingConfig(1),
            )

    def test_distinct_run_records_include_hashes_metrics_and_arguments(self) -> None:
        service = TrainingService(self.workspace, process_factory=self.factory)
        first = service.start_training(self.subset, self.weights, TrainingConfig(1, epochs=2))
        self.wait_for_completion(service)
        record = service.load_run(first.run_id)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["metrics"]["map50"], 0.91)
        self.assertEqual(record["base_weights"]["sha256"], sha256_file(self.weights))
        self.assertIn("--data", record["process_args"])
        self.assertNotIn("--split", record["process_args"])
        self.assertFalse(record["config"]["amp"])
        self.assertNotIn("--amp", record["process_args"])
        repeated = service.start_training(self.subset, self.weights, TrainingConfig(1, epochs=2))
        self.wait_for_completion(service)
        self.assertNotEqual(repeated.run_id, first.run_id)
        self.assertEqual(service.load_run(repeated.run_id)["config_fingerprint"], record["config_fingerprint"])
        second = service.start_training(self.subset, self.weights, TrainingConfig(2, epochs=3, seed=5, amp=True))
        self.wait_for_completion(service)
        self.assertEqual(service.load_run(second.run_id)["status"], "completed")
        self.assertTrue(service.load_run(second.run_id)["config"]["amp"])
        self.assertIn("--amp", service.load_run(second.run_id)["process_args"])

    def test_more_than_four_completed_runs_keep_separate_records_and_results(self) -> None:
        from ai_assessment.services.results import ResultsService

        service = TrainingService(self.workspace, process_factory=self.factory)
        run_ids: list[str] = []
        for number in range(1, 7):
            run = service.start_training(self.subset, self.weights, TrainingConfig(number, epochs=2))
            self.wait_for_completion(service)
            run_ids.append(run.run_id)
            self.assertEqual(service.load_run(run.run_id)["status"], "completed")
        self.assertEqual(len(set(run_ids)), 6)
        self.assertEqual(len(service.list_runs()), 6)
        self.assertEqual(len(ResultsService(self.workspace).completed_best_checkpoints()), 6)
        self.assertEqual(service.load_run(run_ids[-1])["experiment_index"], 6)
        with self.assertRaisesRegex(TrainingError, "positive integer"):
            service.start_training(self.subset, self.weights, TrainingConfig(0))

    def test_training_page_suggests_next_run_number(self) -> None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from ai_assessment.modules.training import TrainingPage

        service = TrainingService(self.workspace, process_factory=self.factory)
        for number in range(1, 6):
            run = service.start_training(self.subset, self.weights, TrainingConfig(number, epochs=2))
            self.wait_for_completion(service)
            self.assertEqual(service.load_run(run.run_id)["status"], "completed")
        app = QApplication.instance() or QApplication([])
        page = TrainingPage(self.workspace)
        try:
            self.assertEqual(page.run_number.value(), 6)
            self.assertEqual(page.runs_table.rowCount(), 5)
        finally:
            page.shutdown()
            page.close()
            del app

    def test_failed_process_retains_failed_record(self) -> None:
        def failed_factory(**kwargs) -> FakeProcess:
            process = FakeProcess(list(kwargs["args"]), return_code=7)
            self.processes.append(process)
            return process

        service = TrainingService(self.workspace, process_factory=failed_factory)
        run = service.start_training(self.subset, self.weights, TrainingConfig(1))
        self.wait_for_completion(service)
        record = service.load_run(run.run_id)
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["error"])
        self.assertTrue(run.log_path.is_file())

    def test_failed_run_can_retry_same_configuration(self) -> None:
        def failed_factory(**kwargs) -> FakeProcess:
            process = FakeProcess(list(kwargs["args"]), return_code=7)
            self.processes.append(process)
            return process

        service = TrainingService(self.workspace, process_factory=failed_factory)
        failed = service.start_training(self.subset, self.weights, TrainingConfig(1))
        self.wait_for_completion(service)
        self.assertEqual(service.load_run(failed.run_id)["status"], "failed")

        service.process_factory = self.factory
        retry = service.start_training(self.subset, self.weights, TrainingConfig(1))
        self.wait_for_completion(service)
        self.assertEqual(service.load_run(retry.run_id)["status"], "completed")
        self.assertEqual(service.load_run(failed.run_id)["status"], "failed")

    def test_cancellation_retains_cancelled_record(self) -> None:
        def blocking_factory(**kwargs) -> FakeProcess:
            args = list(kwargs["args"])
            process = FakeProcess(args, block=True)
            self.processes.append(process)
            return process

        service = TrainingService(self.workspace, process_factory=blocking_factory)
        run = service.start_training(self.subset, self.weights, TrainingConfig(1))
        time.sleep(0.05)
        self.assertTrue(service.cancel_active(run.run_id))
        self.wait_for_completion(service)
        self.assertEqual(service.load_run(run.run_id)["status"], "cancelled")

    def test_missing_metrics_are_unavailable_and_selection_records_exact_hash(self) -> None:
        result_dir = Path(self.temp_dir.name) / "metrics"
        result_dir.mkdir()
        summary = parse_metrics(result_dir, None)
        self.assertIsNone(summary.map50)
        service = TrainingService(self.workspace, process_factory=self.factory)
        run = service.start_training(self.subset, self.weights, TrainingConfig(1))
        self.wait_for_completion(service)
        selected = service.select_production_model(run.run_id)
        selection = json.loads((self.workspace.root / "results" / "selected_model.json").read_text(encoding="utf-8"))
        self.assertEqual(selection["run_id"], run.run_id)
        self.assertEqual(selection["sha256"], sha256_file(selected))
        self.assertTrue(any(event.event_type == "training.model.selected" for event in self.workspace.events.read_events()))

    def test_selection_rejects_checkpoint_outside_workspace(self) -> None:
        service = TrainingService(self.workspace, process_factory=self.factory)
        run = service.start_training(self.subset, self.weights, TrainingConfig(1))
        self.wait_for_completion(service)
        with self.assertRaises(TrainingError):
            service.select_production_model(run.run_id, self.weights)

    def test_official_name_downloads_once_and_records_exact_cached_file(self) -> None:
        calls: list[Path] = []

        def fake_download(target: Path) -> Path:
            calls.append(target)
            target.write_bytes(b"official checkpoint")
            return target

        service = TrainingService(self.workspace, process_factory=self.factory)
        output: list[str] = []
        with patch("ai_assessment.services.training._download_official_weights", side_effect=fake_download):
            first = service.start_training(
                self.subset, Path("yolo26n.pt"), TrainingConfig(1), on_output=output.append
            )
            self.wait_for_completion(service)
        cached = self.workspace.root / "models" / "base_weights" / "yolo26n.pt"
        self.assertEqual(calls, [cached])
        record = service.load_run(first.run_id)
        self.assertEqual(record["base_weights"]["path"], str(cached))
        self.assertEqual(record["base_weights"]["sha256"], sha256_file(cached))
        self.assertEqual(record["base_weights"]["origin"], "ultralytics_official_download")
        self.assertEqual(record["process_args"][record["process_args"].index("--weights") + 1], str(cached))
        self.assertTrue(any("Downloading" in line for line in output))
        self.assertTrue(any(event.event_type == "training.base_weights.downloaded" for event in self.workspace.events.read_events()))

        with patch("ai_assessment.services.training._download_official_weights", side_effect=AssertionError("redownload")):
            second = service.start_training(self.subset, Path("yolo26n.pt"), TrainingConfig(2))
            self.wait_for_completion(service)
        self.assertEqual(service.load_run(second.run_id)["base_weights"]["origin"], "workspace_cache")

    def test_unknown_name_and_url_never_trigger_download(self) -> None:
        service = TrainingService(self.workspace, process_factory=self.factory)
        with patch("ai_assessment.services.training._download_official_weights", side_effect=AssertionError("network")):
            for raw in ("unapproved.pt", "https://example.com/model.pt", "other/yolo26n.pt"):
                with self.subTest(raw=raw), self.assertRaises(TrainingError):
                    service.start_training(self.subset, Path(raw), TrainingConfig(1))
        self.assertFalse(service.list_runs())

    def test_download_failure_is_visible_and_does_not_start_a_run(self) -> None:
        service = TrainingService(self.workspace, process_factory=self.factory)
        with patch("ai_assessment.services.training._download_official_weights", side_effect=ConnectionError("offline")):
            with self.assertRaisesRegex(TrainingError, "GitHub release downloads"):
                service.start_training(self.subset, Path("yolo11s.pt"), TrainingConfig(1))
        self.assertFalse(service.list_runs())
        self.assertFalse((self.workspace.root / "models" / "base_weights" / "yolo11s.pt").is_file())

    def test_concurrent_lookups_write_one_cached_checkpoint(self) -> None:
        calls: list[Path] = []

        def fake_download(target: Path) -> Path:
            calls.append(target)
            time.sleep(0.05)
            target.write_bytes(b"single checkpoint")
            return target

        with patch("ai_assessment.services.training._download_official_weights", side_effect=fake_download):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(resolve_base_weights, self.workspace, Path("yolo26s.pt"))
                    for _ in range(2)
                ]
                results = [future.result() for future in futures]
        self.assertEqual(len(calls), 1)
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual({source["origin"] for _, source in results}, {"ultralytics_official_download", "workspace_cache"})

    def test_monitor_survives_unexpected_output_byte(self) -> None:
        def child_factory(**kwargs):
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["errors"], "replace")
            self.assertEqual(kwargs["args"][1:3], ["-X", "utf8"])
            args = kwargs["args"]
            model_dir = Path(args[args.index("--project") + 1]) / args[args.index("--name") + 1]
            (model_dir / "weights").mkdir(parents=True)
            (model_dir / "weights" / "best.pt").write_bytes(b"checkpoint")
            kwargs["args"] = [
                sys.executable, "-X", "utf8", "-u", "-c",
                "import os; os.write(1, b'valid\\ninvalid:\\x81\\n')",
            ]
            return subprocess.Popen(**kwargs)

        service = TrainingService(self.workspace, process_factory=child_factory)
        run = service.start_training(self.subset, self.weights, TrainingConfig(1))
        self.wait_for_completion(service)
        record = service.load_run(run.run_id)
        self.assertEqual(record["status"], "completed", record["error"])
        self.assertIn("invalid:\ufffd", run.log_path.read_text(encoding="utf-8"))

    def test_monitor_failure_stops_child_before_marking_run_failed(self) -> None:
        processes: list[FakeProcess] = []

        def broken_stream():
            yield "before failure\n"
            raise UnicodeDecodeError("charmap", b"\x81", 0, 1, "undefined")

        def child_factory(**kwargs):
            process = FakeProcess(list(kwargs["args"]), block=True)
            process.stdout = broken_stream()
            processes.append(process)
            return process

        service = TrainingService(self.workspace, process_factory=child_factory)
        run = service.start_training(self.subset, self.weights, TrainingConfig(1))
        self.wait_for_completion(service)
        self.assertTrue(processes[0].terminated)
        self.assertEqual(service.load_run(run.run_id)["status"], "failed")
        self.assertIn("Training monitor failed", service.load_run(run.run_id)["error"])
