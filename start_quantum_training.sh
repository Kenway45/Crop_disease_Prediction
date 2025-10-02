#!/bin/bash
# Start quantum training in background

echo "🔮 Starting quantum classifier training in background..."
echo ""
echo "Training will take approximately 15-30 minutes"
echo "Progress will be logged to: quantum_training.log"
echo ""

# Run training in background
nohup python train_quantum_only.py > quantum_training.log 2>&1 &
PID=$!

echo "Training started with PID: $PID"
echo "PID saved to quantum_training.pid"
echo $PID > quantum_training.pid

echo ""
echo "To monitor progress:"
echo "  tail -f quantum_training.log"
echo ""
echo "To check if still running:"
echo "  ps -p \$(cat quantum_training.pid)"
echo ""
echo "To stop training:"
echo "  kill \$(cat quantum_training.pid)"
echo ""

