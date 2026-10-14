from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import requests

from ai_assessment.core.workspace import CandidateWorkspace
from ai_assessment.services.robot import (
    POSITIONS,
    RobotCapabilities,
    RequestsRobotAdapter,
    RobotResponse,
    RobotResponseError,
    RobotService,
    RobotTimeoutError,
    RobotUncertainError,
    RobotSafetyError,
)
from ai_assessment.services.simulator import EmbeddedSimulator


class FakeAdapter:
    base_url = "fake://robot"
    capabilities = RobotCapabilities(simulator=True, cartesian_move=True)

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.timeout = False
        self.cancel_after_first = False
        self.cancel_event: threading.Event | None = None

    def _response(self, data: dict | None = None) -> RobotResponse:
        return RobotResponse("POST", "/command", 200, {"ok": True, **(data or {})}, 1)

    def health(self, deep: bool = False) -> RobotResponse:
        return RobotResponse("GET", "/health", 200, {"status": "ok", "deep": deep}, 1)

    def positions(self) -> RobotResponse:
        return RobotResponse("GET", "/positions", 200, {"ok": True, "positions": POSITIONS}, 1)

    def suction_state(self) -> RobotResponse:
        return self._response({"on": False})

    def move(self, location: str, height: str, duration_ms: int) -> RobotResponse:
        self.calls.append(("move", location, height, duration_ms))
        if self.timeout:
            raise RobotTimeoutError("request timed out")
        if self.cancel_after_first and len([call for call in self.calls if call[0] == "move"]) == 1 and self.cancel_event:
            self.cancel_event.set()
        return self._response({"location": location, "height": height})

    def suction(self, state: str) -> RobotResponse:
        self.calls.append(("suction", state))
        return self._response({"state": state})

    def move_xyz(self, x: float, y: float, z: float, duration_ms: int) -> RobotResponse:
        self.calls.append(("move_xyz", x, y, z, duration_ms))
        return self._response({"position": {"x": x, "y": y, "z": z}})


class PhysicalFakeAdapter(FakeAdapter):
    capabilities = RobotCapabilities()


class StaticResponse:
    def __init__(self, status_code: int, data: object) -> None:
        self.status_code = status_code
        self.data = data

    def json(self) -> object:
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


class StaticSession:
    def __init__(self, response: StaticResponse) -> None:
        self.response = response

    def request(self, *_args, **_kwargs) -> StaticResponse:
        return self.response


class M5RobotTests(unittest.TestCase):
    def tearDown(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def workspace(self) -> CandidateWorkspace:
        self.temp_dir = tempfile.TemporaryDirectory()
        return CandidateWorkspace.create(Path(self.temp_dir.name) / "C014", candidate_id="C014")

    def test_position_contract_matches_workbook_values(self) -> None:
        self.assertEqual(POSITIONS["P1"]["low"], (-3.0, -130.0, 59.0))
        self.assertEqual(POSITIONS["P5"]["high"], (-124.0, 28.0, 89.0))
        self.assertEqual(set(POSITIONS), {"P1", "P2", "P3", "P4", "P5"})

    def test_move_and_suction_use_same_contract(self) -> None:
        adapter = FakeAdapter()
        service = RobotService(self.workspace(), adapter, sleep_fn=lambda _seconds: None)
        service.move("P2", "high", 1000)
        service.suction("on")
        self.assertEqual(adapter.calls, [("move", "P2", "high", 1000), ("suction", "on")])

    def test_physical_starts_disarmed_and_requires_ready_health_to_arm(self) -> None:
        adapter = PhysicalFakeAdapter()
        service = RobotService(self.workspace(), adapter, sleep_fn=lambda _seconds: None)
        with self.assertRaises(RobotSafetyError):
            service.move("P1", "high", 1000)
        service.arm_physical()
        self.assertTrue(service.physical_armed)
        service.move("P1", "high", 1000)
        self.assertIn(("move", "P1", "high", 1000), adapter.calls)

    def test_supervised_sequence_uses_clearance_route(self) -> None:
        adapter = FakeAdapter()
        service = RobotService(self.workspace(), adapter, sleep_fn=lambda _seconds: None)
        result = service.run_pick_place("P1", "P4", 100, progress=None)
        self.assertEqual(result.status, "completed")
        self.assertEqual(
            adapter.calls,
            [
                ("move", "P1", "high", 100),
                ("move", "P1", "low", 100),
                ("suction", "on"),
                ("move", "P1", "high", 100),
                ("move", "P4", "high", 100),
                ("move", "P4", "low", 100),
                ("suction", "off"),
                ("move", "P4", "high", 100),
            ],
        )

    def test_timeout_is_uncertain_and_not_retried(self) -> None:
        adapter = FakeAdapter()
        adapter.timeout = True
        service = RobotService(self.workspace(), adapter, sleep_fn=lambda _seconds: None)
        with self.assertRaises(RobotUncertainError):
            service.move("P1", "high", 1000)
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(service.workspace.events.read_events()[-1].outcome, "uncertain")

    def test_invalid_json_and_http_errors_are_rejected(self) -> None:
        invalid = RequestsRobotAdapter(
            "http://robot",
            capabilities=RobotCapabilities(),
            session=StaticSession(StaticResponse(200, ValueError("bad json"))),
        )
        with self.assertRaises(RobotResponseError):
            invalid.health()
        failed = RequestsRobotAdapter(
            "http://robot",
            capabilities=RobotCapabilities(),
            session=StaticSession(StaticResponse(503, {"ok": False, "error": "offline"})),
        )
        with self.assertRaises(RobotResponseError):
            failed.health()

    def test_cancellation_prevents_later_steps_without_claiming_stop(self) -> None:
        adapter = FakeAdapter()
        cancel = threading.Event()
        adapter.cancel_after_first = True
        adapter.cancel_event = cancel
        service = RobotService(self.workspace(), adapter, sleep_fn=lambda _seconds: None)
        result = service.run_pick_place("P1", "P2", 100, cancel_event=cancel)
        self.assertEqual(result.status, "cancelled_uncertain")
        self.assertEqual(adapter.calls, [("move", "P1", "high", 100)])
        self.assertIn("not stopped", result.message)

    def test_xyz_is_simulator_only(self) -> None:
        workspace = self.workspace()
        simulator = RobotService(workspace, FakeAdapter(), sleep_fn=lambda _seconds: None)
        simulator.move_xyz(-3, -130, 89, 100)
        physical = RobotService(workspace, PhysicalFakeAdapter(), sleep_fn=lambda _seconds: None)
        with self.assertRaises(RobotSafetyError):
            physical.move_xyz(-3, -130, 89, 100)

    def test_embedded_simulator_external_requests_and_busy_response(self) -> None:
        simulator = EmbeddedSimulator()
        simulator.start()
        try:
            health = requests.get(f"{simulator.base_url}/health", timeout=2)
            self.assertEqual(health.status_code, 200)
            accepted = requests.post(
                f"{simulator.base_url}/command",
                json={"command": "move", "location": "P2", "height": "high", "duration_ms": 100},
                timeout=2,
            )
            self.assertEqual(accepted.status_code, 200)
            busy = requests.post(
                f"{simulator.base_url}/command",
                json={"command": "move", "location": "P3", "height": "low", "duration_ms": 100},
                timeout=2,
            )
            self.assertEqual(busy.status_code, 409)
            positions = requests.get(f"{simulator.base_url}/positions", timeout=2).json()
            self.assertEqual(positions["positions"]["P3"]["low"], [-124.0, -98.0, 59.0])
            xyz = requests.post(
                f"{simulator.base_url}/command",
                json={"command": "move", "x": -3, "y": -130, "z": 89, "duration_ms": 100},
                timeout=2,
            )
            self.assertIn(xyz.status_code, {200, 409})
        finally:
            simulator.stop()



class RobotPageMappingTests(unittest.TestCase):
    def test_changing_one_class_destination_does_not_change_the_selected_class(self) -> None:
        import json
        from unittest import mock

        from PySide6.QtWidgets import QApplication

        from ai_assessment.modules.robot import RobotPage

        app = QApplication.instance() or QApplication([])
        del app
        with tempfile.TemporaryDirectory() as directory:
            workspace = CandidateWorkspace.create(Path(directory) / "C014", candidate_id="C014")
            manifest = workspace.root / "dataset" / "subset" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "selected_order": [3, 7],
                        "selected_class_mapping": {"3": {"name": "date"}, "7": {"name": "lemon"}},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(RobotPage, "_start_embedded_simulator"):
                page = RobotPage(workspace)
            try:
                first, second = (page.mapping_table.cellWidget(row, 1) for row in range(2))
                self.assertEqual(page.trial_class.currentText(), "date")

                first.setCurrentText("P2")
                second.setCurrentText("P3")

                self.assertEqual(first.currentText(), "P2")
                self.assertEqual(second.currentText(), "P3")
                self.assertEqual(page.trial_destination.currentText(), "P2")

                # The trial selector still edits the selected class only.
                page.trial_destination.setCurrentText("P4")
                self.assertEqual(first.currentText(), "P4")
                self.assertEqual(second.currentText(), "P3")
                page.trial_class.setCurrentText("lemon")
                self.assertEqual(page.trial_destination.currentText(), "P3")
            finally:
                page.shutdown()
