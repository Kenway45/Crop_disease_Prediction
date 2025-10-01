#!/bin/bash
# Script to run training in background with logging

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Starting training in background..."
echo "Logs will be saved to training.log"
echo ""
echo "To monitor progress, run:"
echo "  tail -f training.log"
echo ""
echo "To check if training is still running:"
echo "  ps aux | grep train_model.py"
echo ""

nohup python3 src/train_model.py > training.log 2>&1 &

echo "Training started with PID: $!"
echo "You can close this terminal. Training will continue in background."

