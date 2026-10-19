from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageDefinition:
    title: str
    summary: str
    milestone: str


PAGES = [
    PageDefinition(
        "Home",
        "Session status, stage deadlines, connectivity, and alerts.",
        "M1",
    ),
    PageDefinition(
        "Dataset",
        "Browse classes, create the four-class subset, and review provenance.",
        "M3",
    ),
    PageDefinition(
        "Training",
        "Configure recorded YOLO runs and compare model checkpoints.",
        "M6",
    ),
    PageDefinition(
        "Remote Training",
        "Submit datasets and YOLO training to the GPU cluster.",
        "M6",
    ),
    PageDefinition(
        "Vision",
        "Inspect camera input, ROI processing, detections, and centers.",
        "M4",
    ),
    PageDefinition(
        "Calibration",
        "Fit transforms, test clicks, and preserve geometry metadata.",
        "M2",
    ),
    PageDefinition(
        "Robot",
        "Operate the simulator or authorized arm through the shared command API.",
        "M5",
    ),
    PageDefinition(
        "IDE",
        "Edit candidate files and run workspace-local Python code.",
        "M7",
    ),
    PageDefinition(
        "AI Assistant",
        "Ask through the organizer gateway with audited usage.",
        "M8",
    ),
    PageDefinition(
        "Results",
        "Review candidate-visible metrics and simulator outcomes.",
        "M6",
    ),
    PageDefinition(
        "Submission",
        "Validate and lock the final evidence snapshot.",
        "M9",
    ),
]
