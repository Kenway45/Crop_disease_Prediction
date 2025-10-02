#!/bin/bash
# Train Crop Disease Prediction Model with Quantum ML

echo "============================================================"
echo "🔮 Training Crop Disease Model with Quantum ML"
echo "============================================================"
echo ""

# Check if PennyLane is installed
echo "Checking quantum dependencies..."
if python -c "import pennylane" 2>/dev/null; then
    echo "✓ PennyLane is installed"
else
    echo "⚠️  PennyLane not found. Installing..."
    pip install pennylane pennylane-lightning
    if [ $? -eq 0 ]; then
        echo "✓ PennyLane installed successfully"
    else
        echo "❌ Failed to install PennyLane"
        exit 1
    fi
fi

echo ""
echo "Starting training (this may take 2-4 hours)..."
echo "Training will include:"
echo "  1. CNN training"
echo "  2. Embedding extraction"
echo "  3. PCA training"
echo "  4. Classical classifier training"
echo "  5. Quantum classifier training 🔮"
echo ""

# Run training
python src/train_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Training Complete!"
    echo "============================================================"
    echo ""
    echo "Models saved to artifacts/ directory:"
    ls -lh artifacts/models/*.pt artifacts/classifiers/*.joblib 2>/dev/null
    echo ""
    echo "Next steps:"
    echo "  1. Run demo server: python src/demo_server.py"
    echo "  2. Open browser: http://localhost:5000"
    echo ""
else
    echo ""
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi

