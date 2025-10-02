#!/bin/bash
# Run quantum training with LIVE output

echo "=========================================="
echo "🔮 Starting Quantum Training (LIVE)"
echo "=========================================="
echo ""
echo "You will see progress in real-time!"
echo "Training will take 15-30 minutes"
echo ""
echo "Press Ctrl+C to stop training"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Run with unbuffered output
python -u train_quantum_only.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

