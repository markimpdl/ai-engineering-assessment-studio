from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ai_assessment.app import load_workspace, main
from ai_assessment.core.workspace import CandidateWorkspace, WORKSPACE_DIRECTORIES


class CandidateWorkspaceTests(unittest.TestCase):
    def test_create_workspace_writes_expected_structure_and_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "C014"

            workspace = CandidateWorkspace.create(root, candidate_id="C014")

            self.assertTrue((root / "session.json").exists())
            self.assertTrue((root / "project.json").exists())
            for directory in WORKSPACE_DIRECTORIES:
                self.assertTrue((root / directory).is_dir(), directory)
            self.assertEqual(workspace.session.candidate_id, "C014")
            self.assertEqual(len(workspace.events.read_events()), 1)

            reopened = CandidateWorkspace.open(root)

            self.assertEqual(reopened.session.session_id, workspace.session.session_id)
            self.assertEqual(reopened.session.candidate_id, "C014")

    def test_event_acknowledgements_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = CandidateWorkspace.create(
                Path(temp_dir) / "C014",
                candidate_id="C014",
            )
            event = workspace.record_event("workspace.reopened", {"source": "test"})

            first_ack = workspace.events.acknowledge(
                event.event_id,
                receipt_id="receipt-001",
                server_timestamp="2026-09-26T08:00:00Z",
            )
            second_ack = workspace.events.acknowledge(
                event.event_id,
                receipt_id="receipt-ignored",
                server_timestamp="2026-09-26T08:05:00Z",
            )

            self.assertEqual(first_ack, second_ack)
            self.assertEqual(first_ack.receipt_id, "receipt-001")
            self.assertEqual(len(workspace.events.read_acknowledgements()), 1)
            pending_ids = {item.event_id for item in workspace.events.pending_events()}
            self.assertNotIn(event.event_id, pending_ids)

    def test_event_queue_separates_pending_and_acknowledged_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = CandidateWorkspace.create(
                Path(temp_dir) / "C014",
                candidate_id="C014",
            )
            first = workspace.record_event("training.started")
            second = workspace.record_event("training.finished")

            pending_before = {
                item.event_id for item in workspace.events.pending_events()
            }
            self.assertIn(first.event_id, pending_before)
            self.assertIn(second.event_id, pending_before)

            workspace.events.acknowledge(
                first.event_id,
                receipt_id="server-receipt-001",
                server_timestamp="2026-09-26T08:00:00Z",
            )

            pending_after = {
                item.event_id for item in workspace.events.pending_events()
            }
            self.assertNotIn(first.event_id, pending_after)
            self.assertIn(second.event_id, pending_after)

    def test_existing_workspace_can_open_without_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "C014"
            created = CandidateWorkspace.create(root, candidate_id="C014")

            reopened = load_workspace(root)

            self.assertEqual(reopened.session.session_id, created.session.session_id)
            self.assertEqual(reopened.session.candidate_id, "C014")

    def test_new_workspace_takes_candidate_id_from_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = load_workspace(Path(temp_dir) / "candidate_workspaces" / "aluno1")

            self.assertEqual(workspace.session.candidate_id, "aluno1")

    def test_default_workspace_folder_still_yields_c014(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = load_workspace(Path(temp_dir) / "C014")

            self.assertEqual(workspace.session.candidate_id, "C014")

    def test_folder_name_is_sanitised_for_the_training_server_identity(self) -> None:
        from ai_assessment.app import candidate_id_from_workspace

        self.assertEqual(candidate_id_from_workspace(Path("/tmp/João Silva")), "Jo-o-Silva")
        self.assertEqual(candidate_id_from_workspace(Path("/tmp/aluno_02.b")), "aluno_02.b")
        self.assertEqual(candidate_id_from_workspace(Path("/tmp/---")), "C014")

    def test_candidate_id_mismatch_exits_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "C014"
            CandidateWorkspace.create(root, candidate_id="C014")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "--workspace",
                            str(root),
                            "--candidate-id",
                            "C999",
                            "--headless-check",
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Candidate ID mismatch", stderr.getvalue())
            self.assertIn("C014", stderr.getvalue())
            self.assertIn("C999", stderr.getvalue())

    def test_resolve_inside_rejects_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = CandidateWorkspace.create(Path(temp_dir) / "C014", candidate_id="C014")

            with self.assertRaises(ValueError):
                workspace.resolve_inside("..", "other-candidate", "submission.json")


if __name__ == "__main__":
    unittest.main()
