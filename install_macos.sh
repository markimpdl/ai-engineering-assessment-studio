#!/usr/bin/env bash
# AI Engineering Assessment Studio macOS bootstrapper.
# Run from Terminal with: bash install_macos.sh
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This installer must be run on macOS." >&2
    exit 1
fi

cd "$(dirname "$0")"

env_name="ai-assessment-studio"
if [[ -t 0 ]]; then
    read -r -p "Enter a Conda environment name [${env_name}]: " answer
    env_name="${answer:-$env_name}"
fi
if [[ ! "$env_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid environment name. Use letters, numbers, dots, underscores, and hyphens." >&2
    exit 1
fi

conda_cmd=""
if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    conda_cmd="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
    conda_cmd="$(command -v conda)"
else
    for candidate in \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/mambaforge/bin/conda" \
        "/opt/homebrew/bin/conda" \
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            conda_cmd="$candidate"
            break
        fi
    done
fi

if [[ -z "$conda_cmd" ]]; then
    echo "Conda was not found. Install Miniforge or Anaconda for your Mac's architecture," >&2
    echo "then reopen Terminal or run: conda init zsh" >&2
    exit 1
fi

echo "Using Conda: $conda_cmd"
echo "Target environment: $env_name"

if ! env_list="$("$conda_cmd" env list)"; then
    echo "Conda could not list environments." >&2
    exit 1
fi
if printf '%s\n' "$env_list" | awk -v name="$env_name" '$1 == name { found = 1 } END { exit !found }'; then
    echo "Reusing existing environment: $env_name"
    "$conda_cmd" run --no-capture-output -n "$env_name" python -c 'import sys; print("Python", sys.version.split()[0]); assert sys.version_info[:2] == (3, 12), "Existing environment must use Python 3.12"'
else
    echo "Creating Python 3.12 environment: $env_name"
    "$conda_cmd" create -n "$env_name" python=3.12 pip -y
fi

echo "Updating pip..."
"$conda_cmd" run --no-capture-output -n "$env_name" python -m pip install --upgrade pip

# The last published Intel Mac wheels are older than the Apple Silicon wheels.
if [[ "$(uname -m)" == "x86_64" ]]; then
    echo "Installing the available Intel Mac PyTorch wheels (CPU training)..."
    "$conda_cmd" run --no-capture-output -n "$env_name" python -m pip install 'numpy>=1.26,<2' 'torch==2.2.2' 'torchvision==0.17.2'
else
    echo "Installing Apple Silicon PyTorch and torchvision..."
    "$conda_cmd" run --no-capture-output -n "$env_name" python -m pip install torch torchvision
fi

echo "Installing the desktop application..."
"$conda_cmd" run --no-capture-output -n "$env_name" python -m pip install -e '.[gui]'

echo "Verifying imports and the available compute device..."
"$conda_cmd" run --no-capture-output -n "$env_name" python -c 'import PySide6, ai_assessment, cv2, ultralytics, torch; device = "mps" if torch.backends.mps.is_available() else "cpu"; torch.zeros(1, device=device); print("Application imports OK; PyTorch:", torch.__version__, "device:", device)'

echo
echo "Setup complete. Open a new Terminal in this repository and run:"
echo "  conda activate $env_name"
echo "  python -m ai_assessment --headless-check --workspace candidate_workspaces/C014"
echo "  python -m ai_assessment --workspace candidate_workspaces/C014"
echo "In Training, choose device 'mps' when the check above reports mps; otherwise choose 'cpu'."
echo "No model weights or datasets were downloaded by this installer."
