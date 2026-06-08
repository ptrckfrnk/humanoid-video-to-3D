#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  video-to-3D  —  platform-aware installer
#
#  Run AFTER activating the conda environment:
#    conda env create -f environment.yml
#    conda activate video-to-3d
#    bash install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║     video-to-3D  Environment Setup           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Detect hardware ──────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
    PLATFORM="cuda"
    echo "  ✓  NVIDIA GPU detected — will install CUDA 12.1 PyTorch + VGGT-Omega"
elif [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM="mps"
    echo "  ✓  macOS detected — will install standard PyTorch (Apple Silicon MPS)"
else
    PLATFORM="cpu"
    echo "  ⚠  No GPU detected — CPU mode (slow but functional)"
fi
echo ""

# ── Step 1: PyTorch ──────────────────────────────────────────────────────────
echo "Step 1/4  Installing PyTorch..."
if [[ "$PLATFORM" == "cuda" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$PLATFORM" == "mps" ]]; then
    pip install torch torchvision torchaudio
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ── Step 2: VGGT (CVPR 2025 Best Paper) ─────────────────────────────────────
echo ""
echo "Step 2/4  Installing VGGT (CVPR 2025 Best Paper)..."
pip install git+https://github.com/facebookresearch/vggt.git

# ── Step 3: VGGT-Omega (CVPR 2026 Oral — CUDA only) ─────────────────────────
if [[ "$PLATFORM" == "cuda" ]]; then
    echo ""
    echo "Step 3/4  Installing VGGT-Omega (CVPR 2026 Oral)..."
    pip install git+https://github.com/facebookresearch/vggt-omega.git
else
    echo ""
    echo "Step 3/4  VGGT-Omega skipped (CUDA not available — using VGGT-1B)"
fi

# ── Step 4: SAM 2.1 ──────────────────────────────────────────────────────────
echo ""
echo "Step 4/4  Installing SAM 2.1..."
pip install git+https://github.com/facebookresearch/sam2.git

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Setup complete!                             ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Pre-download model checkpoints (optional):  ║"
echo "║    python scripts/download_models.py         ║"
echo "║                                              ║"
echo "║  Run the pipeline:                           ║"
echo "║    python run.py <video.mp4>                 ║"
echo "║    python run.py <video.mp4> --semantic      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
