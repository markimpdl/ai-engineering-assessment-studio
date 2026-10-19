# Direct remote GPU training (trusted-operator pilot)

Assessment Studio Training includes **Remote GPU Cluster** controls and retains local M6 training.

## Connect
Run the Studio from branch `feature/remote-gpu-cluster-training`. Enter the manager's LAN URL (for example `http://192.168.1.100:8000`) and the existing management API key. Select the pre-staged dataset ID `dates-v16`, model, epochs, image size and batch. Submit, Refresh, and Download selected result.

The desktop calls the v0.8 management API directly:
- POST /jobs (student_id from active workspace, mode=train, dataset/model/epochs/imgsz/batch)
- GET /jobs
- GET /jobs/{id}
- POST /jobs/{id}/cancel
- GET /jobs/{id}/artifact

Downloaded ZIP files land under `candidate_workspaces/C###/exports/remote_training/`.

## Security
**For organizer-operated trusted workstations only.** The existing manager key is administrator-grade and gives extensive access to the entire cluster. Displaying only the current candidate's jobs is UI filtering, **not authorization**. Do not distribute the key to competitors, bake it into an installer, store it in the workspace, or expose the manager to an untrusted LAN. Prefer a private management network and HTTPS/VPN. Before student deployment, implement scoped keys directly in manager API (or a centralized trust boundary).

## Known limits
The direct client references a dataset already staged on the GPU manager; it does not transmit selected M3 subsets. Downloaded artifacts are not automatically registered with Results/Vision. Validate exact API paths and responses against your deployed v0.8 manager; no physical cluster E2E verification has yet occurred.


## Upload support with real GPU manager v0.8 app.py

The Studio chooses an exported four-class subset, packages it into a ZIP, computes SHA-256, and uploads using `POST /datasets/upload` with multipart fields `file`, `student_id`, and `sha256`. The returned `dataset_id` becomes the job's dataset ID.

**Deployment:** The manager-side implementation is supplied separately as `app.py` in the tested upgrade ZIP. Replace `manager/app.py` on the management host after backing it up, then rebuild/restart the manager container. The added endpoint saves an immutable, content-addressed subset directory into the *existing* `DATASETS_DIR`; the existing `GET /internal/dataset/{name}` route then distributes it to whichever GPU worker claims training.

A successful POST should return `{"dataset_id":"subset-c014-...","sha256":"...","reused":false}`. Do not deploy the earlier prototype `cluster_dataset_upload_router.py`, which has been removed.

**Configuration:** `MAX_DATASET_UPLOAD_MB` defaults to 1000 and `MAX_DATASET_UNPACKED_MB` to 3000. Persist and share the existing `DATASETS_DIR` manager Docker volume across rebuilds. FastAPI multipart handling requires `python-multipart`, already needed by your worker artifact upload route.

**Test:** Use the Studio's Upload selected subset button, then Start remote training. Confirm dataset directory under `DATASETS_DIR` and worker training can load the uploaded `data.yaml`. Keep direct administrator access restricted to trusted organizer machines.

## Chained training from prior best.pt
Remote Training now offers **Starting weights** with official base weights or completed remote runs belonging to the current candidate. Choose a previous run, and the UI pins its YOLO architecture and submits the optional `source_job_id`. It stays selectable after automatic status refresh.

Install the companion GPU Cluster v0.9 manager and worker patch from `gpu-cluster-v09-chained-training.zip` on the corresponding machines first. The manager serves the previous run's `weights/best.pt` to an authenticated worker; the worker mounts it read-only inside the existing trainer image. Chained runs keep their own unique job ID and artifacts. The source training must be completed and have a valid checkpoint.

This **continues weights from best.pt**, not a literal resume of optimizer state/epoch count from last.pt. The source architecture remains the same (yolo26n -> yolo26n, or yolo26s -> yolo26s). Multiple successive chains are supported.
