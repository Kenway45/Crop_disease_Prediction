# 🔍 How to Monitor Quantum Training - Complete Guide

## 📊 Three Ways to Run & Monitor Training

---

## ✨ **OPTION 1: LIVE VIEW (Recommended)**

### **See training progress in real-time!**

```bash
./train_quantum_live.sh
```

**What you'll see:**
```
🔮 Training Quantum Classifier
   Qubits: 8, Layers: 4
   Classes: 45, Features: 128 → 8
   Training samples: 3456

Epoch 1/30: 100% |████████| 216/216 [01:23<00:00, loss=2.3451]
Epoch 2/30: 100% |████████| 216/216 [01:22<00:00, loss=2.1234]
Epoch 3/30: 100% |████████| 216/216 [01:21<00:00, loss=1.9876]
...
Epoch 5/30 - Loss: 1.8234, Acc: 45.67%
...
```

**Advantages:**
- ✅ Real-time progress bars
- ✅ See loss decreasing
- ✅ Watch accuracy improving
- ✅ Know exactly where you are

**Disadvantages:**
- ❌ Terminal must stay open
- ❌ Can't close window

**To stop:** Press `Ctrl+C`

---

## 🔄 **OPTION 2: BACKGROUND with Status Checks**

### **Run in background, check periodically**

**Start training:**
```bash
./start_quantum_training.sh
```

**Check status anytime:**
```bash
./check_quantum_status.sh
```

**Output:**
```
==========================================
🔮 Quantum Training Status
==========================================

✅ Training is RUNNING (PID: 29578)

Runtime: 05:42
CPU Usage: 98.5%
Memory: 2.1%

Expected total time: 15-30 minutes

To stop training: kill 29578
==========================================
```

**Advantages:**
- ✅ Can close terminal
- ✅ Computer can do other things
- ✅ Training continues even if disconnected

**Disadvantages:**
- ❌ No real-time progress
- ❌ Must manually check status

---

## 💻 **OPTION 3: Direct Python Command**

### **Most basic - run directly**

```bash
python -u train_quantum_only.py
```

**When to use:**
- Quick testing
- Debugging issues
- Want simplest method

---

## 📈 **Understanding the Output**

### **What Each Line Means:**

```
Epoch 5/30: 100% |████████| 216/216 [01:23<00:00, loss=1.8234]
  │    │     │       │       │       │         └─ Current loss
  │    │     │       │       │       └─ Time per batch
  │    │     │       │       └─ Estimated time remaining
  │    │     │       └─ Number of batches (3456 samples ÷ 16 batch_size)
  │    │     └─ Progress bar
  │    └─ Current epoch out of 30 total
  └─ Epoch number
```

### **Accuracy Reports (every 5 epochs):**

```
Epoch 5/30 - Loss: 1.8234, Acc: 45.67%
Epoch 10/30 - Loss: 1.2345, Acc: 62.34%
Epoch 15/30 - Loss: 0.8765, Acc: 72.11%
Epoch 20/30 - Loss: 0.6543, Acc: 78.45%
Epoch 25/30 - Loss: 0.5321, Acc: 81.23%
Epoch 30/30 - Loss: 0.4711, Acc: 84.12%

✓ Quantum classifier training complete!
```

**What good training looks like:**
- ✅ Loss goes DOWN over time
- ✅ Accuracy goes UP over time
- ✅ Steady improvement
- ✅ Final accuracy: 75-85%

---

## ⏱️ **Training Timeline**

```
Minute 0:  Initialization
Minute 1:  Epoch 1/30 - High loss, low accuracy
Minute 3:  Epoch 2-3/30 - Learning starting
Minute 5:  Epoch 5/30 - ~40-50% accuracy
Minute 10: Epoch 15/30 - ~65-75% accuracy
Minute 15: Epoch 25/30 - ~80-85% accuracy
Minute 20: Complete! Final evaluation
```

---

## 🎯 **Quick Commands Reference**

```bash
# LIVE training (see everything)
./train_quantum_live.sh

# Background training
./start_quantum_training.sh

# Check if running
./check_quantum_status.sh

# Check process
ps aux | grep train_quantum

# Stop training
kill $(cat quantum_training.pid)

# See if model exists
ls -lh artifacts/classifiers/quantum_clf.joblib
```

---

## 🐛 **Troubleshooting**

### **Q: I don't see any output!**
**A:** Use live training:
```bash
./train_quantum_live.sh
```

### **Q: Training seems stuck?**
**A:** Check if process is using CPU:
```bash
./check_quantum_status.sh
```
Should show high CPU usage (80-100%)

### **Q: How do I know it's working?**
**A:** Three signs:
1. Process shows in `ps aux | grep train_quantum`
2. CPU usage is high (check with Activity Monitor or `top`)
3. Python process is using memory (~2-3%)

### **Q: Can I speed it up?**
**A:** Yes! Edit `train_quantum_only.py`:
```python
# Line 75: Reduce samples
max_samples_per_class = 50  # Instead of 100

# Line 123: Reduce epochs
epochs = 15  # Instead of 30
```

### **Q: Training failed?**
**A:** Check error in log:
```bash
cat quantum_training.log
```

---

## 📱 **Alternative: Use Terminal Multiplexer**

### **Option A: Using tmux**
```bash
# Start tmux
tmux

# Run training
./train_quantum_live.sh

# Detach: Press Ctrl+B then D
# Reattach: tmux attach
```

### **Option B: Using screen**
```bash
# Start screen
screen

# Run training
./train_quantum_live.sh

# Detach: Press Ctrl+A then D
# Reattach: screen -r
```

---

## 🎬 **Complete Workflow Example**

### **Method 1: Watch it happen (Recommended for first time)**
```bash
cd /Users/jayadharunr/Crop_disease_Prediction
./train_quantum_live.sh

# Watch for 20 minutes, see progress bars
# See loss decrease, accuracy increase
# Training completes automatically
```

### **Method 2: Background & check**
```bash
cd /Users/jayadharunr/Crop_disease_Prediction
./start_quantum_training.sh

# Do other things for 20 minutes
# Check occasionally:
./check_quantum_status.sh

# When status shows completion:
# ✅ Quantum classifier found!
```

---

## 📊 **What Success Looks Like**

### **During Training:**
```
Epoch 1/30:  100% |████████| 216/216 [loss=2.34]
Epoch 5/30:  100% |████████| 216/216 [loss=1.82]  Acc: 45%
Epoch 10/30: 100% |████████| 216/216 [loss=1.23]  Acc: 62%
Epoch 15/30: 100% |████████| 216/216 [loss=0.88]  Acc: 72%
Epoch 20/30: 100% |████████| 216/216 [loss=0.65]  Acc: 78%
Epoch 25/30: 100% |████████| 216/216 [loss=0.53]  Acc: 81%
Epoch 30/30: 100% |████████| 216/216 [loss=0.47]  Acc: 84%

✓ Quantum classifier training complete!
```

### **After Completion:**
```bash
$ ./check_quantum_status.sh

✅ Quantum classifier found!
-rw-r--r-- 1 user staff 260K quantum_clf.joblib

Training completed successfully! 🎉
```

---

## 🚀 **What's Next After Training?**

Once you see:
```
✓ Quantum classifier training complete!
✓ Saved to: artifacts/classifiers/quantum_clf.joblib
```

**Run the demo:**
```bash
python src/demo_server.py
```

**You'll see:**
```
✓ Loaded quantum classifier 🔮

Starting Crop Disease Prediction Demo Server
Open your browser: http://localhost:5000
```

---

## 🎯 **My Recommendation**

**For your FIRST training run:**
```bash
./train_quantum_live.sh
```

**Why?**
- You can SEE it working
- Watch the progress bars
- See loss decreasing
- Feel confident it's training
- Learn what normal looks like

**For FUTURE runs:**
```bash
./start_quantum_training.sh
```

**Why?**
- You know it works
- Can do other things
- Just check status occasionally

---

## ✅ **Quick Start NOW**

**Ready to see live training? Run this:**

```bash
cd /Users/jayadharunr/Crop_disease_Prediction
./train_quantum_live.sh
```

**You'll see everything happen in real-time! 🔮**

---

**Happy Training! 🚀**

