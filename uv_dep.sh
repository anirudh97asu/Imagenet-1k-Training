#!/bin/bash

# =============================================================================
# Remove all CUDA 12.9+ dependencies and let PyTorch manage CUDA
# =============================================================================

set -e

echo "================================================"
echo "Removing incompatible CUDA 12.9+ dependencies"
echo "================================================"

# List of CUDA packages to remove
cuda_packages=(
    "nvidia-cublas-cu12"
    "nvidia-cuda-cupti-cu12"
    "nvidia-cuda-nvrtc-cu12"
    "nvidia-cuda-runtime-cu12"
    "nvidia-cudnn-cu12"
    "nvidia-cufft-cu12"
    "nvidia-cufile-cu12"
    "nvidia-curand-cu12"
    "nvidia-cusolver-cu12"
    "nvidia-cusparse-cu12"
    "nvidia-cusparselt-cu12"
    "nvidia-nccl-cu12"
    "nvidia-nvjitlink-cu12"
    "nvidia-nvtx-cu12"
)

echo "Removing CUDA packages from pyproject.toml..."
for pkg in "${cuda_packages[@]}"; do
    echo "  Removing $pkg..."
    uv remove "$pkg" 2>/dev/null || echo "    (not found, skipping)"
done

echo ""
echo "✓ CUDA packages removed"
echo ""
echo "================================================"
echo "Installing PyTorch (will install compatible CUDA)"
echo "================================================"

# Remove torch first to clean state
uv remove torch torchvision torchaudio 2>/dev/null || true

# Add PyTorch fresh (it will install compatible CUDA 12.8)
uv add torch torchvision torchaudio

echo ""
echo "================================================"
echo "Installing torch-lr-finder"
echo "================================================"

uv add torch-lr-finder

echo ""
echo "================================================"
echo "✓ Installation complete!"
echo "================================================"
echo ""
echo "Verifying installation..."
uv pip list | grep -E "torch|nvidia"

echo ""
echo "================================================"
echo "Success! PyTorch now manages CUDA dependencies"
echo "================================================"