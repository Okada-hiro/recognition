#!/usr/bin/env bash
set -euo pipefail

# Reception recognition POC environment bootstrapper.
# Uses YOLO (PyTorch) for person detection and InsightFace (ONNX Runtime)
# for face detection + face recognition.
#
# Usage:
#   bash environment.sh
#
# Optional environment variables:
#   PYTHON_BIN=python3.10
#   VENV_DIR=.venv
#   SKIP_VENV=1
#   FORCE_CPU=1
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
#   INSIGHTFACE_MODEL_NAME=buffalo_l

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
SKIP_VENV="${SKIP_VENV:-0}"
FORCE_CPU="${FORCE_CPU:-0}"
INSIGHTFACE_MODEL_NAME="${INSIGHTFACE_MODEL_NAME:-buffalo_l}"

cd "$REPO_ROOT"

echo "[1/7] Checking Python version..."
"$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10) or (major, minor) > (3, 12):
    raise SystemExit(
        f"Python {major}.{minor} is not recommended. Use Python 3.10-3.12."
    )
print(f"Using Python {major}.{minor}")
PY

if [[ "$SKIP_VENV" != "1" ]]; then
  echo "[2/7] Creating virtual environment at $VENV_DIR ..."
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "venv creation failed. Trying virtualenv fallback..."
    "$PYTHON_BIN" -m pip install --upgrade virtualenv
    "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "[2/7] SKIP_VENV=1, using current Python environment."
fi

echo "[3/7] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

GPU_AVAILABLE=0
if [[ "$FORCE_CPU" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_AVAILABLE=1
fi

if [[ "$GPU_AVAILABLE" == "1" ]]; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
  ORT_SPEC="${ORT_SPEC:-onnxruntime-gpu}"
  echo "[4/7] NVIDIA GPU detected. Installing CUDA-enabled builds..."
else
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
  ORT_SPEC="${ORT_SPEC:-onnxruntime}"
  echo "[4/7] No NVIDIA GPU detected. Installing CPU builds..."
fi

python -m pip install --upgrade \
  torch torchvision torchaudio \
  --index-url "$TORCH_INDEX_URL"

echo "[5/7] Installing runtime dependencies..."
python -m pip install --upgrade \
  "$ORT_SPEC" \
  onnx \
  opencv-python \
  numpy \
  pillow \
  matplotlib \
  pyyaml \
  requests \
  scipy \
  psutil \
  polars \
  flask \
  flask-cors \
  shapely \
  gdown \
  pandas \
  tqdm \
  mtcnn \
  fire \
  gunicorn \
  lightphe \
  lightdsa \
  python-dotenv \
  fastapi \
  "uvicorn[standard]" \
  websockets \
  easydict \
  scikit-learn \
  scikit-image \
  albumentations \
  prettytable

echo "[6/7] Installing local repository packages..."
python -m pip install --upgrade -e "$REPO_ROOT/ultralytics"

echo "[7/7] Verifying imports and accelerator visibility..."
python - <<'PY'
import sys
from pathlib import Path

import onnxruntime
import torch
from ultralytics import YOLO

repo_root = Path.cwd()
insightface_root = repo_root / "insightface" / "python-package"
if str(insightface_root) not in sys.path:
    sys.path.insert(0, str(insightface_root))
from insightface.app import FaceAnalysis

print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.device_count", torch.cuda.device_count())
    print("torch.cuda.current_device", torch.cuda.current_device())
    print("torch.cuda.device_name", torch.cuda.get_device_name(torch.cuda.current_device()))

print("onnxruntime", onnxruntime.__version__)
print("onnxruntime.providers", onnxruntime.get_available_providers())
print("ultralytics", YOLO.__name__)
print("insightface", FaceAnalysis.__name__)
PY

cat <<'EOF'

Environment setup finished.

Recommended commands:
  source .venv/bin/activate
  python -m recognition.main --device auto

Video processing example:
  python -m recognition.main --input-video movie.mp4 --output-video movie_annotated.mp4 --device auto --no-display

Database layout:
  data_base/<person_id>/<image>.jpg

Single-page reception UI:
  python -m recognition.runpod_recognition_browser

CPU-only setup:
  FORCE_CPU=1 bash environment.sh
EOF
