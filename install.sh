#!/usr/bin/env bash
# install.sh — set up everything needed to run High Fantasy Word Explorer
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${PYTORCH_VENV}"
OVIE_DIR="${OVIE_PATH:-$(dirname "$SCRIPT_DIR")/ovie}"

# ── 1. Python / venv ──────────────────────────────────────────────────────────
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: PyTorch venv not found at $VENV"
    echo "  Create one with CUDA-enabled PyTorch first, then re-run this script."
    echo "  Example:"
    echo "    python3 -m venv $VENV"
    echo "    source $VENV/bin/activate"
    echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    exit 1
fi

echo "Using venv: $VENV"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

# ── 2. CUDA check ─────────────────────────────────────────────────────────────
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available in $VENV"
    echo "  Make sure you installed a CUDA-enabled PyTorch wheel."
    exit 1
fi
echo "CUDA OK (torch $("$PYTHON" -c 'import torch; print(torch.__version__)'))"

# ── 3. Python packages ────────────────────────────────────────────────────────
echo "Installing / verifying Python packages …"
"$PIP" install --quiet --upgrade \
    "miro-t2i" \
    "pygame>=2.6" \
    "pillow>=11" \
    "numpy>=2" \
    "einops>=0.8" \
    "timm>=1" \
    "accelerate>=1" \
    "safetensors" \
    "diffusers>=0.38" \
    "transformers>=4.50" \
    "huggingface-hub>=0.34" \
    "omegaconf>=2.3" \
    "pyyaml"

# ── 4. OVIE repository ────────────────────────────────────────────────────────
if [[ ! -d "$OVIE_DIR" ]]; then
    echo "Cloning OVIE into $OVIE_DIR …"
    git clone --depth 1 https://github.com/kyutai-labs/ovie "$OVIE_DIR"
else
    echo "OVIE repo found at $OVIE_DIR"
fi

# Quick smoke-test: can we import the two key OVIE modules?
OVIE_OK=$("$PYTHON" - <<EOF
import sys
sys.path.insert(0, "$OVIE_DIR")
try:
    from models.models import OVIEModel
    from utils.pose_enc import extri_intri_to_pose_encoding
    print("ok")
except Exception as e:
    print(f"fail:{e}")
EOF
)
if [[ "$OVIE_OK" != "ok" ]]; then
    echo "WARNING: OVIE import check failed: $OVIE_OK"
    echo "  You may need to install additional OVIE dependencies manually."
    echo "  See $OVIE_DIR/pyproject.toml"
else
    echo "OVIE import OK"
fi

# ── 5. Pre-download model weights (optional but convenient) ───────────────────
read -r -p "Pre-download model weights now? (~several GB, skip with N) [y/N] " ans
if [[ "${ans,,}" == "y" ]]; then
    echo "Downloading MIRO weights …"
    "$PYTHON" -c "
from miro import MiroPipeline
import torch
MiroPipeline.from_pretrained('nicolas-dufour/miro')
print('MIRO weights OK')
"
    echo "Downloading OVIE weights …"
    "$PYTHON" - <<EOF
import sys
sys.path.insert(0, "$OVIE_DIR")
from models.models import OVIEModel
OVIEModel.from_pretrained("kyutai/ovie", revision="v1.0")
print("OVIE weights OK")
EOF
fi

echo ""
echo "All done. Run the game with:  ./run.sh"
