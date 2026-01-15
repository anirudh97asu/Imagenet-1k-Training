#!/bin/bash
# ============================================================================
# ResNet-50 Training Examples
# All optimizations: torch.compile, OneCycle, Progressive Resize, DDP, AMP
# ============================================================================
# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
echo -e "${BLUE}ResNet-50 Training Examples${NC}\n"
# ============================================================================
# EXAMPLE 1: Single GPU - Basic Training
# ============================================================================
example_1_basic_single_gpu() {
    echo -e "${GREEN}[1] Single GPU - Basic Training${NC}"
    echo "Command:"
    echo "python run.py \\"
    echo "  --data-path ./imagenet \\"
    echo "  --epochs 90 \\"
    echo "  --batch-size 256 \\"
    echo "  --lr 0.1"
    echo ""
    echo "Features:"
    echo "  ✓ ResNet-50 from model.py"
    echo "  ✓ torch.compile enabled (~10-20% speedup)"
    echo "  ✓ FP16 AMP enabled (~1.8x faster)"
    echo "  ✓ MultiStepLR scheduler"
    echo ""
}
# ============================================================================
# EXAMPLE 2: Single GPU with Progressive Resize + OneCycle
# ============================================================================
example_2_progressive_onecycle() {
    echo -e "${GREEN}[2] Single GPU - Progressive Resize + OneCycle${NC}"
    echo "Command:"
    echo "python run.py \\"
    echo "  --data-path ./imagenet \\"
    echo "  --epochs 90 \\"
    echo "  --batch-size 256 \\"
    echo "  --progressive-resize \\"
    echo "  --scheduler onecycle \\"
    echo "  --max-lr 0.1 \\"
    echo "  --pct-start 0.3"
    echo ""
    echo "Features:"
    echo "  ✓ ResNet-50 from model.py"
    echo "  ✓ torch.compile enabled"
    echo "  ✓ Progressive resize: 128→160→192→224"
    echo "  ✓ OneC

