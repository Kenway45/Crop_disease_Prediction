#!/bin/bash
cd "$(dirname "$0")"

echo "🚀 Starting Crop Disease Model Training..."
echo "============================================"
echo ""
echo "Training with:"
echo "  • Device: Apple MPS (GPU acceleration)"
echo "  • Epochs: 10"
echo "  • Images: 56,134 across 45 classes"
echo "  • Estimated time: 1-2 hours"
echo ""
echo "Progress will be displayed below..."
echo ""

python3 src/train_model.py 2>&1 | tee training.log

echo ""
echo "============================================"
echo "Training completed! Check training.log for full output."

