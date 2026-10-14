from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from .core.workspace import CandidateWorkspace
from .modules.pages import PAGES, PageDefinition


DEFAULT_CANDIDATE_ID = "C014"
_CANDIDATE_ID_CHARS = re.compile(r"[^A-Za-z0-9_.\-]+")


def candidate_id_from_workspace(path: Path) -> str:
    """Derive a candidate ID from the workspace folder name (e.g. ``aluno1``).

    Only characters accepted by the training server identity rule are kept, so the
    same ID can be sent as ``X-Candidate-ID``. Falls back to the default ID when the
    folder name has nothing usable.
    """

    name = _CANDIDATE_ID_CHARS.sub("-", path.resolve().name).strip("-.")
    return name[:64] or DEFAULT_CANDIDATE_ID


def default_workspace_root() -> Path:
    configured = os.environ.get("AI_ASSESSMENT_WORKSPACE")
    if configured:
        return Path(configured)
    return Path.cwd() / "candidate_workspaces" / DEFAULT_CANDIDATE_ID


def load_workspace(path: Path, candidate_id: str | None = None) -> CandidateWorkspace:
    if (path / "session.json").exists():
        workspace = CandidateWorkspace.open(path)
        if (
            candidate_id is not None
            and candidate_id != workspace.session.candidate_id
        ):
            raise ValueError(
                "Candidate ID mismatch: workspace contains "
                f"{workspace.session.candidate_id!r}, but --candidate-id requested "
                f"{candidate_id!r}."
            )
        workspace.record_event("workspace.opened", {"source": "app"})
        return workspace
    return CandidateWorkspace.create(
        path,
        candidate_id=candidate_id or candidate_id_from_workspace(path),
        mode="training",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Engineering Assessment Studio")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace_root(),
        help="Candidate workspace folder. Defaults to ./candidate_workspaces/C014.",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help=(
            "Candidate identifier for a newly created workspace. "
            "Defaults to the workspace folder name; when reopening, omit it or match session.json."
        ),
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="Create/open the workspace and print a compact status without launching Qt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        workspace = load_workspace(args.workspace, args.candidate_id)
    except ValueError as exc:
        parser.error(str(exc))

    if args.headless_check:
        pending = len(workspace.events.pending_events())
        print(
            f"workspace={workspace.root} candidate={workspace.session.candidate_id} "
            f"session={workspace.session.session_id} pending_events={pending}"
        )
        return 0

    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QFrame,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QScrollArea,
            QStackedWidget,
            QVBoxLayout,
            QWidget,
        )
        from .modules.calibration import CalibrationPage
        from .modules.dataset import DatasetPage
        from .modules.ai_assistant import AIAssistantPage
        from .modules.ide import IDEPage
        from .modules.robot import RobotPage
        from .modules.results import ResultsPage
        from .modules.training import TrainingPage
        from .modules.vision import VisionPage
        from .services.ai_context import AIContextBuffer
        from .services.ai_ide_transfer import AIIdeCoordinator
    except ImportError as exc:
        print(
            "PySide6, NumPy, OpenCV, and PyYAML are required to launch the desktop shell. "
            "Run with --headless-check to verify workspace services.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    class Page(QWidget):
        def __init__(self, definition: PageDefinition) -> None:
            super().__init__()
            layout = QVBoxLayout(self)
            layout.setContentsMargins(28, 28, 28, 28)
            layout.setSpacing(14)

            title = QLabel(definition.title)
            title.setObjectName("pageTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignLeft)

            heading = QHBoxLayout()
            heading.addWidget(title)
            heading.addStretch(1)

            summary = QLabel(definition.summary)
            summary.setWordWrap(True)
            summary.setObjectName("summary")

            workspace_label = QLabel(f"Workspace: {workspace.root}")
            workspace_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            workspace_label.setWordWrap(True)
            workspace_label.setObjectName("workspacePath")

            layout.addLayout(heading)
            layout.addWidget(summary)
            layout.addSpacing(12)
            layout.addWidget(workspace_label)
            layout.addStretch(1)

    def scrollable(page: QWidget) -> QScrollArea:
        """Host a page in a scroll area so its minimum size never forces the window size.

        Without this, the stacked widget's minimum size is the largest page's minimum
        (over 1100 px tall), which exceeds 1080p screens and makes maximized windows
        overflow and overlap their controls.
        """

        area = QScrollArea()
        area.setObjectName("pageScroll")
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(page)
        return area

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("AI Engineering Assessment Studio")
            self.setMinimumSize(960, 600)
            self.resize(1160, 760)

            container = QWidget()
            root = QHBoxLayout(container)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            sidebar_frame = QFrame()
            sidebar_frame.setObjectName("sidebar")
            sidebar_layout = QVBoxLayout(sidebar_frame)
            sidebar_layout.setContentsMargins(18, 20, 18, 20)
            sidebar_layout.setSpacing(12)

            product = QLabel("Assessment Studio")
            product.setObjectName("product")
            candidate = QLabel(
                f"{workspace.session.candidate_id}  |  {workspace.session.mode.title()}"
            )
            candidate.setObjectName("candidate")
            workspace_folder = QLabel(f"Workspace: {workspace.root.name}")
            workspace_folder.setObjectName("workspaceFolder")
            workspace_folder.setWordWrap(True)
            workspace_folder.setToolTip(str(workspace.root))
            workspace_folder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            self.nav = QListWidget()
            self.nav.setObjectName("navigation")
            for page in PAGES:
                self.nav.addItem(QListWidgetItem(page.title))

            sidebar_layout.addWidget(product)
            sidebar_layout.addWidget(candidate)
            sidebar_layout.addWidget(workspace_folder)
            sidebar_layout.addWidget(self.nav, 1)

            self.stack = QStackedWidget()
            self.calibration_page = None
            self.dataset_page = None
            self.vision_page = None
            self.robot_page = None
            self.training_page = None
            self.results_page = None
            self.ide_page = None
            self.ai_page = None
            context_buffer = AIContextBuffer()
            coordinator = AIIdeCoordinator(self)
            ai_index = next(index for index, page in enumerate(PAGES) if page.title == "AI Assistant")
            ide_index = next(index for index, page in enumerate(PAGES) if page.title == "IDE")
            coordinator.ask_requested.connect(lambda _request: self.nav.setCurrentRow(ai_index))
            coordinator.handoff_requested.connect(lambda _handoff: self.nav.setCurrentRow(ide_index))
            for page in PAGES:
                if page.title == "Calibration":
                    self.calibration_page = CalibrationPage(workspace)
                    self.stack.addWidget(scrollable(self.calibration_page))
                elif page.title == "Dataset":
                    self.dataset_page = DatasetPage(workspace)
                    self.stack.addWidget(scrollable(self.dataset_page))
                elif page.title == "Vision":
                    self.vision_page = VisionPage(workspace)
                    self.stack.addWidget(scrollable(self.vision_page))
                elif page.title == "Robot":
                    self.robot_page = RobotPage(workspace)
                    self.stack.addWidget(scrollable(self.robot_page))
                elif page.title == "Training":
                    self.training_page = TrainingPage(workspace)
                    self.stack.addWidget(scrollable(self.training_page))
                elif page.title == "Results":
                    self.results_page = ResultsPage(workspace)
                    self.stack.addWidget(scrollable(self.results_page))
                elif page.title == "IDE":
                    self.ide_page = IDEPage(workspace, context_buffer, coordinator)
                    self.stack.addWidget(scrollable(self.ide_page))
                elif page.title == "AI Assistant":
                    self.ai_page = AIAssistantPage(workspace, context_buffer, coordinator)
                    self.stack.addWidget(scrollable(self.ai_page))
                else:
                    self.stack.addWidget(scrollable(Page(page)))
            self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
            self.nav.setCurrentRow(0)

            root.addWidget(sidebar_frame)
            root.addWidget(self.stack, 1)
            self.setCentralWidget(container)
            self.statusBar().showMessage(self.status_message())

        def status_message(self) -> str:
            total = len(workspace.events.read_events())
            pending = len(workspace.events.pending_events())
            return f"{total} local events, {pending} awaiting acknowledgement"

        def shutdown(self) -> None:
            if self.calibration_page is not None:
                self.calibration_page.shutdown()
            if self.dataset_page is not None:
                self.dataset_page.shutdown()
            if self.vision_page is not None:
                self.vision_page.shutdown()
            if self.robot_page is not None:
                self.robot_page.shutdown()
            if self.training_page is not None:
                self.training_page.shutdown()
            if self.results_page is not None:
                self.results_page.shutdown()
            if self.ide_page is not None:
                self.ide_page.shutdown()
            if self.ai_page is not None:
                self.ai_page.shutdown()

    app = QApplication([sys.argv[0]])
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background: #f8faf9;
            color: #17211d;
            font-family: Segoe UI, Arial, sans-serif;
            font-size: 14px;
        }
        #sidebar {
            background: #ecf1ef;
            border-right: 1px solid #d5ded9;
            min-width: 250px;
            max-width: 250px;
        }
        #product {
            font-size: 20px;
            font-weight: 700;
        }
        #candidate {
            color: #53635c;
        }
        #workspaceFolder {
            color: #0c3525;
            background: #e2ece7;
            border-radius: 6px;
            padding: 6px 8px;
            font-size: 13px;
        }
        #navigation {
            background: transparent;
            border: 0;
            outline: 0;
        }
        #navigation::item {
            min-height: 34px;
            padding: 6px 10px;
            border-radius: 6px;
        }
        #navigation::item:selected {
            background: #d7e7df;
            color: #0c3525;
        }
        #pageTitle {
            font-size: 30px;
            font-weight: 700;
        }
        #summary {
            color: #394942;
            font-size: 16px;
        }
        #workspacePath {
            color: #56665f;
            background: #eef4f1;
            border: 1px solid #d6e1dc;
            border-radius: 6px;
            padding: 10px;
        }
        QPushButton, QToolButton {
            background: #ffffff;
            color: #17304d;
            border: 1px solid #8fa5b5;
            border-radius: 5px;
            padding: 7px 13px;
            min-height: 32px;
            font-weight: 600;
        }
        QPushButton:hover, QToolButton:hover {
            background: #e8f3f2;
            border-color: #087f75;
        }
        QPushButton:pressed, QToolButton:pressed {
            background: #d3e9e6;
            border-color: #05665e;
        }
        QPushButton:focus, QToolButton:focus {
            border: 2px solid #087f75;
            padding: 6px 12px;
        }
        QPushButton:disabled, QToolButton:disabled {
            background: #eef2f3;
            color: #89979e;
            border-color: #cbd5d9;
        }
        QToolButton::menu-button {
            border-left: 1px solid #b6c5cb;
            width: 20px;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background: #ffffff;
            color: #17304d;
            border: 1px solid #9eb0bc;
            border-radius: 5px;
            padding: 6px 9px;
            min-height: 32px;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #087f75;
            padding: 5px 8px;
        }
        QComboBox::drop-down {
            border-left: 1px solid #c2d0d5;
            width: 26px;
        }
        QCheckBox {
            spacing: 7px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QGroupBox {
            background: #ffffff;
            border: 1px solid #c7d4d8;
            border-radius: 6px;
            margin-top: 9px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #173c2b;
            font-weight: 700;
        }
        QTabWidget::pane {
            border: 1px solid #c7d4d8;
            background: #ffffff;
        }
        QTabBar::tab {
            background: #eaf0f1;
            color: #53656b;
            border: 1px solid #c7d4d8;
            padding: 7px 13px;
            min-width: 92px;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            color: #173c2b;
            font-weight: 700;
            border-bottom-color: #ffffff;
        }
        QStatusBar {
            background: #eef4f1;
            color: #53635c;
            border-top: 1px solid #d5ded9;
        }
        """
    )
    window = MainWindow()
    app.aboutToQuit.connect(window.shutdown)
    screen = app.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        window.resize(
            max(window.minimumWidth(), min(1480, int(available.width() * 0.9))),
            max(window.minimumHeight(), min(960, int(available.height() * 0.9))),
        )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
