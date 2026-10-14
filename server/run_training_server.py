from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
from pathlib import Path

import uvicorn

from .training_server import TrainingServer, TrainingServerConfig, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LAN training server for AI Assessment Studio candidates")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address; 0.0.0.0 accepts candidates on the local network")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", default=str(Path.home() / "ai-assessment-training-server"))
    parser.add_argument(
        "--student",
        action="append",
        default=[],
        metavar="ID[=TOKEN]",
        help="Token mode only: register a candidate. Without =TOKEN a token is generated once and kept in students.json.",
    )
    parser.add_argument(
        "--token-mode",
        action="store_true",
        help="Require a per-candidate token (set with setx AI_TRAINING_SERVER_TOKEN) instead of open LAN mode",
    )
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--accelerator", default=None, help="Device for blank/CUDA requests (default: mps on Apple silicon, else 0)")
    parser.add_argument("--default-workers", type=int, default=2, help="Data loader workers used when the candidate leaves workers at 0")
    parser.add_argument("--max-job-hours", type=float, default=6.0)
    parser.add_argument("--no-caffeinate", action="store_true", help="Allow macOS to sleep while the server runs")
    return parser


def load_students(data_dir: Path, requested: list[str]) -> dict[str, str]:
    """Return token -> candidate, persisting tokens so each PC is configured once."""
    path = data_dir / "students.json"
    by_student: dict[str, str] = {}
    if path.is_file():
        by_student = {str(key): str(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}
    for entry in requested:
        student, _, token = entry.partition("=")
        student = student.strip()
        if not student:
            raise SystemExit(f"Invalid --student value: {entry!r}")
        if token:
            by_student[student] = token.strip()
        else:
            by_student.setdefault(student, secrets.token_urlsafe(18))
    if not by_student:
        raise SystemExit("Register at least one candidate with --student ID (for example --student C014).")
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(by_student, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    tokens = {token: student for student, token in by_student.items()}
    if len(tokens) != len(by_student):
        raise SystemExit("Each candidate needs a distinct token.")
    return tokens


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # no packet is sent for UDP connect
            return probe.getsockname()[0]
    except OSError:
        return socket.gethostname()


def default_accelerator() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    open_mode = not args.token_mode
    if open_mode and not args.student:
        # Keep tokens from an earlier token-mode run valid for desktops still using them.
        students = load_students(data_dir, []) if (data_dir / "students.json").is_file() else {}
    else:
        students = load_students(data_dir, args.student)
    config = TrainingServerConfig(
        data_dir=data_dir,
        students=students,
        open_mode=open_mode,
        max_concurrent=args.max_concurrent,
        accelerator=args.accelerator or default_accelerator(),
        default_workers=args.default_workers,
        max_job_seconds=args.max_job_hours * 3600,
    )
    server = TrainingServer(config)
    if sys.platform == "darwin" and not args.no_caffeinate:
        # Keep the Mac awake (system sleep only) for as long as this server process lives.
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    url = f"http://{lan_address()}:{args.port}"
    print(f"Training server data: {data_dir}", flush=True)
    print(f"Accelerator: {config.accelerator}; concurrent jobs: {config.max_concurrent}", flush=True)
    print(f"Dashboard: {url}/", flush=True)
    if open_mode:
        print(f"Open LAN mode: on each Studio Training page, enter Server IP {lan_address()} and press Test server.", flush=True)
        if students:
            print(f"Earlier tokens for {', '.join(sorted(students.values()))} remain accepted.", flush=True)
    else:
        print("Configure each Windows PC once (then reopen the Studio):", flush=True)
        for token, student in sorted(students.items(), key=lambda item: item[1]):
            print(f"  {student}:  setx AI_TRAINING_SERVER_URL {url}  &&  setx AI_TRAINING_SERVER_TOKEN {token}", flush=True)
    uvicorn.run(create_app(server), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
