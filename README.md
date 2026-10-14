# AI Engineering Assessment Studio

AI Engineering date-fruit training and assessment workstation. M1 provides a PySide6 desktop shell with navigation across the planned application pages, a candidate workspace, append-only local event logs, and idempotent server-receipt acknowledgement storage. M2 adds an embedded four-point camera-to-robot calibration workspace. M3 adds Dataset Studio for read-only YOLO dataset inspection, four-class subset export, and printable object exports. M4 adds an embedded Vision Studio for camera preview, ROI processing, local YOLO inference, and full-frame detection centers. M5 adds supervised Robot Studio. M6 adds recorded YOLO training experiments and candidate-visible Results Studio. M7 adds a workspace-owned Python IDE and a process-backed terminal. M8 adds a separate organizer AI gateway and selected-context assistant page.

The M1-M8 shell does not connect to a competition organizer server by default. Local events remain pending until a future organizer integration records a server acknowledgement. Calibration is a 2D planar mapping tool only, Vision Studio never sends robot commands or performs automatic sorting, Dataset Studio does not train YOLO, Robot Studio does not sort live detections, Results Studio does not show hidden organizer scores, the IDE never arms the physical robot, and the AI page never calls OpenAI directly.

For a student-facing walkthrough with Windows screenshots, see [docs/STUDENT_VISUAL_GUIDE.md](docs/STUDENT_VISUAL_GUIDE.md).
The original offline walkthrough is available as a printable [PDF](output/pdf/student-visual-guide.pdf); the Markdown guide and Training section below describe the newer base-model download option.

Read [Architecture v1](docs/ARCHITECTURE_V1.md) and [AGENTS.md](AGENTS.md) for the project boundaries and acceptance gates.

## Windows setup

Use Anaconda Prompt or another shell where `conda` is initialized. The project baseline targets Python 3.12:

The easiest setup is to run [`install_windows.bat`](install_windows.bat) from either Command Prompt or PowerShell. It asks for the Conda environment name, creates it when needed, and reuses an existing Python 3.12 environment. On an NVIDIA PC where `nvidia-smi -L` works, it checks the existing PyTorch installation and, when needed, installs the official CUDA 12.8 PyTorch 2.9.1 / torchvision 0.24.1 wheels **before** the GUI package. It verifies CUDA device access and fails visibly if a detected GPU cannot run PyTorch. Without a working NVIDIA driver/GPU, it installs for CPU use. The batch file uses `conda run`, so open a new prompt and activate the named environment after it finishes. Rerun the installer with the same environment name to repair an earlier CPU-only installation.

For manual setup on an NVIDIA PC with a compatible driver, use the same CUDA wheels:

```bat
cd /d C:\Projects\ai-engineering-assessment-studio
conda create -n ai-assessment-studio python=3.12 -y
conda activate ai-assessment-studio
python -m pip install --force-reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[gui]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

On a CPU-only PC, omit the CUDA PyTorch command. If CUDA verification fails, check the [PyTorch Windows selector](https://pytorch.org/get-started/locally/) for a wheel supported by your GPU and driver.

If the repository is in a different folder, replace the `cd` path. The editable install makes the `ai_assessment` package available to the commands below.

## macOS setup (practice use)

Install a native macOS Conda distribution (Miniforge or Anaconda) for your Mac's architecture. From Terminal in this repository, run:

```bash
bash install_macos.sh
```

The script asks for an environment name (default `ai-assessment-studio`), creates or reuses a Python 3.12 Conda environment, installs the macOS PyTorch wheels and `.[gui]`, then checks that the application imports and that a CPU or MPS tensor can be created. If Conda is not found, initialize it with `conda init zsh` (or `conda init bash` for Bash), reopen Terminal, and rerun the script. Existing environments must use Python 3.12.

After setup, open a new Terminal in the repository:

```bash
conda activate ai-assessment-studio
python -m ai_assessment --headless-check --workspace candidate_workspaces/C014
python -m ai_assessment --workspace candidate_workspaces/C014
```

On Apple Silicon, select `mps` in Training when the installer reports `device: mps`; otherwise select `cpu`. Intel Macs install the last available compatible PyTorch 2.2.2 / torchvision 0.17.2 wheels and train on CPU. macOS does not use the Windows CUDA PyTorch wheels. This is a developer/practice install; the official assessment workstation remains the Windows 11 baseline. A real Mac test of the camera, training, and simulator is still needed.

## Run the shell and tests

Create or reopen the default `C014` workspace and launch the desktop shell. A new workspace takes its candidate ID from the folder name, so each student can use their own folder, for example `--workspace candidate_workspaces\aluno1`; the sidebar shows both the ID and the workspace folder. Pass `--candidate-id` only to override that name:

```bat
python -m ai_assessment --workspace candidate_workspaces\C014
```

The sidebar shows the candidate ID saved in the active workspace's `session.json`. For a new candidate workspace, use its ID as the folder name; the app will assign that ID automatically:

```bat
python -m ai_assessment --workspace candidate_workspaces\C015
```

Alternatively, `python -m ai_assessment --candidate-id C015` creates or opens `candidate_workspaces/C015` when no workspace path is provided. From inside an existing candidate workspace, the app reopens that workspace. From the repository root, it automatically opens a single existing candidate workspace; when several exist, provide `--workspace` so the app does not guess. Without any existing workspace or candidate ID, the practice default remains `C014`.

When reopening, the saved session ID must match a candidate-style folder name and any explicit `--candidate-id`. If an older run created `candidate_workspaces/C015/session.json` with `C014`, startup reports the mismatch instead of displaying the wrong identity. Preserve that workspace and its audit events; create a correctly identified workspace for new work rather than editing `session.json` by hand.

Run the non-GUI workspace check:

```bat
python -m ai_assessment --headless-check --workspace candidate_workspaces\C014
```

Run the full M1-M8 tests:

```bat
python -m unittest discover -s tests -v
```

Events are stored in `candidate_workspaces\C014\logs\events.jsonl`; server receipts are stored in `candidate_workspaces\C014\logs\event_acks.jsonl`.

## M2 calibration

Open the **Calibration** page from the application sidebar. Start a camera or open a saved image, freeze the full-frame image, select four image points, enter their matching robot X/Y coordinates, and choose exactly one transform:

- `2 x 3 Affine` for a plane where perspective is negligible.
- `3 x 3 Perspective` for a plane viewed at an angle.

After calculating the matrix, click the frozen image to see a rounded integer robot X/Y prediction. The 5x cursor magnifier shows a pixel-level view. The matrix display is rounded to two decimals, while the workspace file retains full precision.

Use **Save to workspace** and **Load workspace calibration** to persist the active calibration at `candidate_workspaces\C014\calibration\calibration.json`. The service reads legacy schema versions 1 and 2. It records the point pairs, transform, image geometry, coordinate convention, camera metadata, optional ROI metadata, and an image snapshot. A saved calibration cannot be applied to an image with a different width or height.

The sample package has no live robot credentials, API keys, copyrighted dataset payload, or candidate data.

## M3 Dataset Studio

Open **Dataset** in the sidebar and choose a local YOLO export whose root contains `data.yaml` plus image and label folders for `train`, `valid` or `val`, and `test`. Scanning is read-only and runs away from the Qt interface. The page reports class IDs and names, image/object counts by split, missing or malformed files, and previews boxes and segmentation polygons.

Select exactly four classes and arrange their order. **Export working subset** writes an atomic, workspace-owned subset below `candidate_workspaces\C014\dataset\`. The selected original IDs are remapped to new IDs `0` through `3`; split membership and annotation coordinates are preserved. Images containing both selected and unselected classes are excluded and reported rather than silently losing objects. Test images remain in the test split. Each export includes `data.yaml`, a versioned manifest, source/version metadata when available, counts, exclusions, and SHA-256 hashes.

The **Print exports** tab adapts the preserved legacy utility. It can write transparent PNGs, white-background JPEGs, and A4 PDFs with 6 x 4 cm cards, cut marks, shape cut lines, fit rotation, and class names under `candidate_workspaces\C014\exports\print\`. Box-only annotations are clearly reported as rectangular cutouts. The original dataset and legacy source are never modified. Dataset selection and successful exports are recorded in the local event log; no organizer signature or lock is claimed in M3.

The GUI extra installs the required M3 dependencies, including PyYAML and ReportLab:

```bat
python -m pip install -e ".[gui]"
```

The full M1-M8 test suite remains:

```bat
python -m unittest discover -s tests -v
```

## M4 Vision Studio

Open **Vision** in the sidebar. Choose **Camera** for a live preview, **Video file (loop)** for a local file, or **RTSP stream** for a live network source. For RTSP, paste an address such as `rtsp://127.0.0.1:8554/camera`; Vision selects the RTSP input when you finish editing, then click **Start stream**. The URL remains visible and editable even while an input is running. To use a changed URL, click **Stop**, then **Start stream**. Select **Hide RTSP URL** if you need to obscure it on screen. Vision reconnects if the server is unavailable or a stream drops; **Stop** ends retries. Local video playback repeats from the beginning and follows the video's frame rate (with a fallback for files without reliable FPS metadata). OpenCV handles video decoding, so usable formats and codecs depend on the installed OpenCV build. Video frames are processed locally, without copying the file into the candidate workspace; audio is not played. Capture, video decoding, network reads, model loading, and inference run outside the Qt interface. Select an existing local `.pt` or supported `.onnx` file with **Browse model...**. The app never downloads weights or silently substitutes cached base weights for trained checkpoints.

When **Results** has a selected production model, Vision shows that run ID and the full `models/<run-id>/weights/best.pt` checkpoint path separately from the current Vision profile model. Use **Use selected Results model** to replace any older profile path and load the selected checkpoint with its recorded SHA-256. Vision also lists completed training runs from the current candidate workspace; choose a run and use **Use run best.pt** to load that run's trained checkpoint. Missing or changed recorded checkpoints are reported before loading. Cached base weights under `models/base_weights/` are not treated as trained checkpoints. The manual `.pt`/`.onnx` browse path remains available for local experiments.

Vision Studio shows the untouched full-frame input image beside the corrected ROI sent to YOLO. Drag an ROI on the raw image and tune confidence, software brightness and contrast for all three inputs. For camera input, request exposure, focus, or white-balance values; the camera panel shows requested values and driver read-back values because camera drivers may ignore or quantize settings. In video and RTSP modes, camera index and camera properties are disabled, while brightness and contrast remain available. A camera calibration is never applied to a prerecorded video or RTSP input. RTSP reads use the OpenCV FFmpeg backend with connection/read timeouts; the Windows environment must include that backend and a codec supported by the stream. The actual inference device is shown after model load.

Detection rows contain class ID/name, confidence, full-frame bounding box, full-frame center pixels, frame ID, and optional robot X/Y. For camera input, a saved M2 calibration is used only when its full-frame geometry matches the current frame; a mismatch is shown as a warning. A class ID is never treated as a robot position. **Save snapshot** writes the raw frame, annotated processed image, and metadata under `candidate_workspaces\C014\exports\vision\`. The chosen input, local video path, and adjustments are persisted at `candidate_workspaces\C014\camera\vision_profile.json`.

The RTSP URL is used only during the current session; it is never written to the Vision profile, snapshot metadata, or event log because it may include a username, password, or token. Paste the URL again after restarting the application. The Studio reads an RTSP publisher or server; it does not publish webcam video as an RTSP server. Calibration and Vision share a camera ownership guard, so starting one module while the other has the webcam reports which module must be stopped first. RTSP and local video do not acquire the webcam. Vision logs camera, video, or RTSP start/stop, model load, ROI changes, and snapshots without logging every frame.

## M5 Robot Studio

Open **Robot** in the sidebar. The page starts the preserved MaxArm simulator on a loopback port and uses the same HTTP contract as a physical MaxArm: `GET /health`, `/health/deep`, `/positions`, `/suction`, and `POST /command`. The simulator is also reachable by external Python clients while the application is open:

```bat
python -c "import requests; print(requests.post('http://127.0.0.1:PORT/command', json={'command':'suction','state':'on'}).json())"
```

Replace `PORT` with the port shown in the Robot page base URL. The default bind is local-only. **Expose simulator on LAN** is an explicit opt-in and restarts the simulator on `0.0.0.0`; use it only on a trusted network.

Robot Studio provides non-overlapping manual P1-P5 high/low moves, duration, suction, health, and positions controls. It displays the simulator current and target XYZ pose, animated progress, suction state, virtual objects, and command history. Simulator-only XYZ moves are bounds-checked. Physical MaxArm mode exposes only the documented position and suction contract, starts disarmed, requires a health/positions check and explicit session enable, and does not expose XYZ or software stop controls. The physical emergency-stop and power-isolation procedure remains outside the application.

The supervised trial uses the clearance route `source high -> source low -> suction on -> source high -> destination high -> destination low -> suction off -> destination high`. It serializes commands, waits for motion and release settling, handles busy/HTTP/timeout failures without blind retries, and reports an accepted-but-unconfirmed command as uncertain. Cancellation prevents later commands but never claims to stop a move already accepted by the controller. Four class names are loaded from the latest M3 subset manifest and mapped to destinations for trial selection; live Vision detections are not connected to robot movement in M5.

Robot commands, responses, state changes, errors, and trials are written to the candidate event log. Simulator state is available at `/sim/state`; the preserved prototype reset endpoints remain simulator-only. M5 tests include the external `requests.post(.../command)` path, position parity, busy handling, sequence order, timeout uncertainty, cancellation, and simulator-only XYZ capability. A physical MaxArm and supervised Windows hardware trial are still required before enabling real motion in production.

## M6 Training and Results

Open **Training** in the sidebar after exporting an M3 subset. Select a `dataset/subset-*/` folder and either browse an existing local `.pt` base-weight file or type an official detection model name such as `yolo26n.pt`. A missing named model is downloaded from official Ultralytics GitHub release assets when **Start training** is pressed. Training does not change the source dataset or use hidden assessment data. The service validates the complete four-class manifest, `train`, `val`, and `test` paths, and the contiguous class mapping before launching a child process. The child runs Ultralytics `train` and candidate-visible `val`; the preserved `test` split is not used by M6. The training runner writes a run-owned `resolved_data.yaml` with the selected subset's absolute root so Ultralytics finds its image folders regardless of the working directory. Existing M3 exports containing `path: .` work without re-exporting; the original subset and manifest remain unchanged.

Supported downloadable detection names are `yolo26`, `yolo11`, and `yolov8` with sizes `n`, `s`, `m`, `l`, or `x` and a `.pt` extension (for example, `yolo26n.pt` or `yolov8s.pt`). The model is cached under `candidate_workspaces\C014\models\base_weights\`; later runs reuse it. The run record stores its requested name, origin, absolute path, byte count, and SHA-256. Existing local `.pt` paths still work. Unknown names and arbitrary URLs never trigger a download. If the router blocks a download, the UI reports the error and no training run starts.

For the competition network, allow the official Ultralytics GitHub release downloads from `github.com/ultralytics/assets`, any GitHub API lookup needed by the installed version (`api.github.com`), and the GitHub release asset redirect host observed during a pilot. Keep network policy on the router or an organizer proxy; the candidate app itself cannot enforce Internet restrictions. Approve model families and preserve downloaded hashes in organizer evidence. Offline sites can place approved `.pt` weights locally before the event.

Training **Device** is blank by default: Ultralytics chooses CUDA device 0 when PyTorch can access a GPU and CPU otherwise. Choose `cpu` explicitly for a CPU smoke test. **Enable AMP** is unchecked by default: FP32 training can still use CUDA, while Ultralytics' AMP capability check may try to download a second `yolo26n.pt` into its own weights directory. Enable AMP only when that check model is available and its optional download is permitted by the router. The AMP choice is saved with each experiment. The Windows installer sets up CUDA PyTorch first on detected NVIDIA machines; verify a full GPU training run on the official image before competition use. No model weights, dataset, workspace, or generated run output belongs in Git.

Training allows additional local runs after the fourth. The **Run number** field defaults to one higher than the largest number already recorded, and can be changed to any positive integer. Repeating the same settings creates a new run ID and keeps both records for comparison. Each run keeps its ID, number, parameters, seed, manifest and base-weight hashes, environment/package versions, process arguments, timestamps, status, logs, metrics, and checkpoint hashes under `models/`, `results/training/`, and `logs/training/`. Failed and cancelled runs remain visible and can be retried. Only one training process can run at a time per workspace; cancellation releases the Windows child process tree. This local change does not revise the architecture's proposed four-experiment assessment scheme or establish official scoring rules.

Use **Results** to compare candidate-visible mAP50, mAP50-95, precision, recall, available per-class metrics, loss-curve rows, confusion-matrix artifacts, and latency. Values that were not produced are shown as `unavailable`. Training saves each run under `candidate_workspaces/<candidate-id>/models/<run-id>/`: use `weights/best.pt` for the trained model, while `results.csv` and training plots stay in the same run folder. Candidate-visible validation plots may appear in the sibling `models/<run-id>_val/` folder. The app's metadata is stored separately in `candidate_workspaces/<candidate-id>/results/training/<run-id>/run.json`, with the console log at `logs/training/<run-id>.log`. In **Results**, select the completed run to see absolute paths, open the run folder, copy the `best.pt` path, or choose **Select production model** to write `results/selected_model.json` with the exact SHA-256. In **Vision**, use **Use selected Results model** to load that exact checkpoint even if the saved camera profile points at an older model, or choose a completed run from the Vision run list and use **Use run best.pt**. Vision refreshes those choices when the page is opened and verifies recorded hashes before loading. Selection is a local candidate choice, not an organizer signature, lock, or hidden score.

For a short local smoke test, use a tiny four-class YOLO fixture and a local fake or approved `.pt` file, then run:

```bat
python -m unittest discover -s tests -p "test_m6*.py" -v
python -m ai_assessment --headless-check --workspace candidate_workspaces\C014
```

The complete suite remains:

```bat
python -m unittest discover -s tests -v
```

A real Windows GPU run is still required to verify Ultralytics, CUDA/PyTorch compatibility, training duration, checkpoint production, and metric plots on the official image.

### Optional LAN training server

An operator can run training for several candidate PCs on one faster machine on the same local network, such as an Apple silicon Mac. On the **Training** page, enter the server address in **Server IP** (below **Base weights**, for example `192.168.1.228`, or `host:port` when the server does not use port 8766) and press **Test server** to check it. With a valid address, the desktop uploads the selected subset and base weights, streams the server's console output into the same log and progress bar with server paths rewritten to the candidate's own paths, and extracts `best.pt`, `results.csv`, plots, and metrics into the same `models/<run-id>/` and `results/training/<run-id>/` folders before the run is recorded. A blank or invalid address trains on this PC. When the server is unreachable or rejects the job, training also starts on this PC and the status line says so. The address is saved in `results/training_server.json` in the candidate workspace. Run records, hashes, the four-experiment rule, cancellation, and production-model selection stay on the candidate PC; each `run.json` records an `executor` entry (server, job ID, and server environment, or the local fallback reason).

On the server machine (Python 3.12, from the repository root):

```sh
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[training-server]"
python -m server.run_training_server
```

The server prints its LAN address and a dashboard URL (`http://<server-ip>:8766/`) listing running, queued, and finished jobs. By default it runs in open LAN mode: each desktop identifies itself with its workspace candidate ID and session ID, so several PCs that share a candidate ID such as `C014` still get separate jobs. It runs up to `--max-concurrent` jobs at once (default 4; one active job per PC, further jobs wait in a queue), maps a blank or CUDA device to Apple `mps` (an explicit `cpu` stays CPU), uses `--default-workers` data loaders when the candidate leaves workers at 0, stops a job after `--max-job-hours`, and keeps macOS awake with `caffeinate` while it runs. Allow incoming connections for Python if the macOS firewall asks. Ultralytics prints the actual device (for example `MPS (Apple M5 Pro)`) in the training log.

For per-candidate tokens instead, start the server with `--token-mode --student C014 --student C015`; tokens are generated once and kept in `~/ai-assessment-training-server/students.json`, and each PC sets its token once with `setx AI_TRAINING_SERVER_TOKEN <token>`. An `AI_TRAINING_SERVER_URL` environment variable pre-fills **Server IP** until the field is first saved.

The server is a classroom convenience on a trusted network, not an assessment security boundary: traffic is plain HTTP, open mode trusts the identity each desktop sends, and the dashboard needs no token. If the network drops during a run, the desktop keeps retrying for ten minutes before recording the run as failed; closing the Studio cancels the remote job, as it cancels a local one.

## M8 AI Assistant and organizer gateway

The **AI Assistant** page calls a separately run organizer gateway over versioned HTTP. The desktop contains no OpenAI SDK or provider key. It sends only the question and an explicitly selected context type: IDE selection, error traceback, or training metrics. The entire workspace, dataset, model weights, hidden material, and terminal input are never attached automatically. Local `ai.request` and `ai.result` events contain hashes and sizes, not raw prompt or response text; they remain local events until a future server receipt exists.

For a labelled local practice session, install the server extra and start the deterministic fake-provider gateway from the repository root:

```bat
conda activate ai-assessment-studio
python -m pip install -e ".[server]"
python -m server.run_gateway --host 127.0.0.1 --port 8765 --practice-session C014 --practice-token practice-C014
```

In a second Anaconda Prompt, launch the desktop with the same operator-provisioned practice credential:

```bat
conda activate ai-assessment-studio
set AI_GATEWAY_URL=http://127.0.0.1:8765
set AI_GATEWAY_SESSION_TOKEN=practice-C014
python -m ai_assessment --workspace candidate_workspaces\C014
```

The page shows the approved model, session allowance, connection state, request progress, history, and quota errors. Retry is always an explicit action using the same idempotency key; a timeout is shown as uncertain rather than silently resent. A quick mock-provider check is:

```bat
python -c "import requests; h={'Authorization':'Bearer practice-C014','Idempotency-Key':'smoke-1'}; print(requests.post('http://127.0.0.1:8765/api/v1/messages',headers=h,json={'message':'Say hello','context':{'type':'selected_text','text':'print(1)'}}).json())"
```

The gateway enforces server-side session authorization, input/context bounds, per-session and global budgets, concurrent requests, rate limits, reservations, stale cleanup, and idempotency. Practice credentials are explicitly labelled and are not proof of competition identity. Assessment mode accepts only a trusted organizer-provisioned credential; candidate ID, local `session.json`, editable client values, and local events cannot activate it. The competitor can inspect or script the desktop client, so all quota and access rules live on the gateway.

For optional organizer-controlled live verification, install `.[server-live]`, set `OPENAI_API_KEY`, `AI_GATEWAY_MODEL`, and the server-side price assumptions on the organizer machine, then start with `--provider openai`. Never set the provider key in the candidate desktop environment or pass it to IDE/training processes. Provider tools, web access, uploads, and autonomous actions are disabled by this adapter.

M9 still needs competition-day identity, signed activation, official server deployment, retention/access policy approval, network outage rehearsal, and organizer receipt integration. The local practice gateway is not an assessment security boundary.

### IDE to AI to IDE workflow

In **IDE**, select code and choose **Ask AI > Explain**, **Fix**, **Improve**, or **Generate code**. The application moves to **AI Assistant** with the exact question and selected code visible for review; press **Send** deliberately. With no selection, the IDE asks whether the current file may be attached. Responses keep each fenced code block separate. Choose **Send to IDE** on one block, then select **New file**, **Insert at cursor**, or **Replace selection**. Replacement shows a diff first. The result opens as a modified unsaved draft and is never automatically saved or run. If the source file or selection changed while the request was in flight, the IDE warns and asks for a destination again. Model-suggested paths are ignored; all eventual saves still use the candidate workspace path checks.

## M7 IDE and terminal

Open **IDE** to browse the candidate workspace and edit Python files under its `src/` directory. New files default there. Save and Save As validate every path through the candidate workspace boundary and reject `..`, symlink, junction, dataset, and repository-source writes. The editor has tabs, line numbers, modified indicators, and unsaved-change prompts. Saved files can import workspace code because the runner adds the candidate workspace `src/` to `PYTHONPATH`; they can inspect local training outputs and call the local M5 simulator URL shown in Robot Studio. Running code never arms a physical robot.

**Run Python** launches the saved file with the configured local interpreter, an argument array, and the candidate workspace as its working directory. Output streams live, exit status and duration are shown, and complete output is kept in `logs/execution/`. The Stop button owns only the IDE process tree and does not stop M6 training. Repeated runs are rejected while one run is active. Candidate execution is not an OS security sandbox; do not treat it as a boundary for untrusted code. Provider keys and hidden assessment material are not injected into the child environment.

The terminal is a real `cmd.exe` process on Windows (or `/bin/sh` on other platforms) with stdin and streamed stdout/stderr. It starts in the candidate workspace and can be restarted independently. Because it is pipe-based, full Windows console features, terminal control sequences, and some secure interactive prompts may be limited. Terminal input and passwords are not recorded.

### Windows M7 smoke checklist

1. Open IDE, create a Python file, edit it, save it, close and reopen the tab, and confirm the modified marker and save prompt.
2. Run a script that prints Unicode and both normal output and a traceback; confirm the streams and exit code appear.
3. Run a long-lived script that starts a child process, press Stop, and confirm the script and child exit without changing a training job.
4. Start, stop, and restart the terminal; run a harmless command from the workspace.
5. Start the M5 local simulator, copy its displayed URL, and run a candidate script using `requests` against `/health` or `/positions`. Confirm the simulator responds and no physical-robot enable action is created.
