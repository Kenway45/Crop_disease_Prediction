#!/bin/bash
# Check quantum training status

echo "=========================================="
echo "🔮 Quantum Training Status"
echo "=========================================="
echo ""

# Check if process is running
if [ -f quantum_training.pid ]; then
    PID=$(cat quantum_training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Training is RUNNING (PID: $PID)"
        echo ""
        
        # Show how long it's been running
        RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
        echo "Runtime: $RUNTIME"
        echo ""
        
        # CPU usage
        CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
        echo "CPU Usage: ${CPU}%"
        echo ""
        
        # Memory usage
        MEM=$(ps -p $PID -o %mem= | tr -d ' ')
        echo "Memory: ${MEM}%"
        echo ""
        
        echo "Expected total time: 15-30 minutes"
        echo ""
        echo "To stop training: kill $PID"
    else
        echo "❌ Training process not running"
        echo ""
        
        # Check if quantum classifier was saved
        if [ -f artifacts/classifiers/quantum_clf.joblib ]; then
            echo "✅ Quantum classifier found!"
            ls -lh artifacts/classifiers/quantum_clf.joblib
            echo ""
            echo "Training completed successfully! 🎉"
        else
            echo "No quantum classifier found."
            echo "Training may have failed or not started yet."
        fi
    fi
else
    echo "No training PID file found"
    echo ""
    
    # Check if classifier exists
    if [ -f artifacts/classifiers/quantum_clf.joblib ]; then
        echo "✅ Quantum classifier exists!"
        ls -lh artifacts/classifiers/quantum_clf.joblib
    else
        echo "Start training with: python train_quantum_only.py"
    fi
fi

echo ""
echo "=========================================="

