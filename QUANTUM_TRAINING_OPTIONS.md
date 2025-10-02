# 🔮 Quantum Training Options Guide

## Overview

You have **5 different ways** to train quantum classifiers, ranging from ultra-fast toy examples to full production models.

---

## 🎯 Quick Comparison

| Option | Time | Classes | Qubits | Purpose |
|--------|------|---------|--------|---------|
| **Toy** | 2-3 min | 3 | 4 | Quick test/learning |
| **Simple** | 5-10 min | 10 | 6 | Fast demo |
| **Standard** | 15-30 min | 45 | 8 | Production ready |
| **Full Training** | 2-4 hours | 45 | 8 | Complete retrain |
| **Custom** | Variable | Custom | Custom | Your own config |

---

## 1️⃣ TOY Version (FASTEST - 2-3 minutes)

**Perfect for: Quick testing, learning, immediate results**

```bash
python train_quantum_toy.py
```

### What it does:
- ✅ Uses only **3 disease classes** (most common)
- ✅ **30 samples per class** (90 total)
- ✅ **4 qubits, 2 layers** (tiny circuit)
- ✅ **10 epochs**
- ✅ Training time: **~2-3 minutes**

### When to use:
- First time trying quantum ML
- Want immediate results
- Learning how quantum classifiers work
- Testing the setup

### Output:
```
✓ TOY Quantum Training Complete!

🎉 You just trained a quantum classifier!

What you learned:
  • Quantum circuits can classify data
  • Uses qubits and quantum gates
  • Trained with gradient descent
  • Achieved 55-70% accuracy on 3 classes
```

---

## 2️⃣ SIMPLE Version (FAST - 5-10 minutes)

**Perfect for: Quick demos, presentations, testing**

```bash
python train_quantum_simple.py
```

### What it does:
- ✅ Uses **10 disease classes** (most common)
- ✅ **50 samples per class** (500 total)
- ✅ **6 qubits, 3 layers** (moderate circuit)
- ✅ **15 epochs**
- ✅ Training time: **~5-10 minutes**

### When to use:
- Need a working demo quickly
- Presenting quantum ML concepts
- Testing with realistic data
- Don't need all 45 classes

### Output:
```
✓ FAST Quantum Training Complete!

This quantum classifier:
  • Works on 10 disease classes (most common ones)
  • Uses 6 qubits and 3 layers
  • Achieved 60-75% accuracy
```

---

## 3️⃣ STANDARD Version (RECOMMENDED - 15-30 minutes)

**Perfect for: Production use, full capability**

```bash
python train_quantum_only.py
```

### What it does:
- ✅ Uses **ALL 45 disease classes**
- ✅ **100 samples per class** (4,500 total)
- ✅ **8 qubits, 4 layers** (full circuit)
- ✅ **30 epochs**
- ✅ Training time: **~15-30 minutes**

### When to use:
- Want full quantum capability
- Production deployment
- Research paper
- Comparing with classical model

### Output:
```
✓ Quantum Classifier Training Complete!

✓ Train accuracy: 82.45%
✓ Test accuracy: 78.92%
✓ Saved to: artifacts/classifiers/quantum_clf.joblib

The server will now use BOTH classical and quantum predictions
```

---

## 4️⃣ FULL TRAINING (COMPLETE - 2-4 hours)

**Perfect for: Complete retrain, starting from scratch**

```bash
python src/train_model.py
```

### What it does:
- ✅ Trains **CNN from scratch** (ResNet18)
- ✅ Extracts **all embeddings**
- ✅ Trains **PCA**
- ✅ Trains **classical classifier**
- ✅ Trains **quantum classifier** (8 qubits, 45 classes)
- ✅ Training time: **~2-4 hours**

### When to use:
- Starting completely fresh
- Have new dataset
- Want to retrain everything
- Research purposes

---

## 5️⃣ CUSTOM Configuration

**Perfect for: Experimentation, research, specific needs**

### Create your own script:

```python
from quantum_classifier import QuantumClassifier

# Your custom configuration
quantum_clf = QuantumClassifier(
    n_features=128,      # From PCA
    n_classes=45,        # Or fewer
    n_qubits=10,         # More qubits = more capacity
    n_layers=6          # Deeper = more expressivity
)

quantum_clf.fit(
    X_train, 
    y_train,
    epochs=50,           # More epochs = better training
    batch_size=8,        # Smaller = slower but better
    learning_rate=0.01   # Adjust for convergence
)
```

### Parameters to adjust:

**Qubits:**
- More qubits = exponentially larger state space
- More expressivity but slower training
- Range: 4-12 (practical limit)

**Layers:**
- More layers = deeper circuit
- More complex patterns
- Range: 2-8

**Epochs:**
- More epochs = better convergence
- Diminishing returns after ~50
- Range: 10-100

**Batch size:**
- Smaller = better gradients, slower
- Larger = faster, noisier gradients
- Range: 4-32

---

## ⚡ Performance Comparison

### Training Speed:

```
Toy:      ▓░░░░░░░░░  2-3 min    (4 qubits, 3 classes)
Simple:   ▓▓▓░░░░░░░  5-10 min   (6 qubits, 10 classes)
Standard: ▓▓▓▓▓░░░░░  15-30 min  (8 qubits, 45 classes)
Full:     ▓▓▓▓▓▓▓▓▓▓  2-4 hours  (Complete retrain)
```

### Accuracy:

```
Toy:      ▓▓▓▓▓░░░░░  55-70%  (Limited classes)
Simple:   ▓▓▓▓▓▓░░░░  60-75%  (10 classes)
Standard: ▓▓▓▓▓▓▓▓░░  75-85%  (All 45 classes)
Classical: ▓▓▓▓▓▓▓▓▓▓  92-96%  (For reference)
```

---

## 🚀 Recommended Workflow

### First Time:
1. Start with **TOY** (2-3 min) - understand quantum basics
2. Try **SIMPLE** (5-10 min) - see realistic performance
3. Run **STANDARD** (15-30 min) - get production model

### For Production:
- Use **STANDARD** version
- Integrated with demo server
- Provides both classical + quantum predictions

### For Research:
- Start with **STANDARD**
- Experiment with **CUSTOM** configurations
- Try different quantum architectures

---

## 🎮 Try Right Now

### Fastest path (2 minutes):
```bash
python train_quantum_toy.py
```

### Best balance (5 minutes):
```bash
python train_quantum_simple.py
```

### Production ready (15 minutes):
```bash
python train_quantum_only.py
```

---

## 🔬 Understanding the Differences

### Why different versions?

**1. Problem Complexity:**
- 3 classes: Easier problem, faster training
- 10 classes: Moderate difficulty
- 45 classes: Full complexity

**2. Quantum Resources:**
- 4 qubits: 2⁴ = 16 dimensional space
- 6 qubits: 2⁶ = 64 dimensional space
- 8 qubits: 2⁸ = 256 dimensional space

**3. Training Time:**
- Grows with: samples × classes × qubits × layers × epochs
- Toy: 90 samples × simple circuit = fast
- Standard: 4,500 samples × complex circuit = slower

---

## 💡 Tips

### Speed up training:
```bash
# Reduce samples
max_samples_per_class = 50  # Instead of 100

# Reduce epochs
epochs = 15  # Instead of 30

# Increase batch size
batch_size = 32  # Instead of 16

# Reduce circuit size
n_qubits = 6    # Instead of 8
n_layers = 3    # Instead of 4
```

### Improve accuracy:
```bash
# More samples
max_samples_per_class = 200

# More epochs
epochs = 50

# Smaller batch size
batch_size = 8

# Larger circuit
n_qubits = 10
n_layers = 6
```

---

## 🐛 Troubleshooting

### Training too slow?
- Use TOY or SIMPLE version
- Reduce epochs
- Reduce samples per class
- Increase batch size

### Not enough accuracy?
- Use STANDARD version
- More epochs
- More samples
- Deeper circuit (more layers)

### Out of memory?
- Reduce batch size
- Reduce qubits
- Use TOY version

### Want to stop training?
- Press Ctrl+C
- Training will stop gracefully
- No files will be corrupted

---

## 📊 Which One Should I Choose?

### Use TOY if:
- ⏰ You have 2 minutes
- 📚 Learning quantum ML
- 🧪 Testing setup
- 🎮 Want immediate results

### Use SIMPLE if:
- ⏰ You have 5-10 minutes
- 🎤 Giving a demo
- 🔬 Quick experiment
- ✅ Need something that works

### Use STANDARD if:
- ⏰ You have 15-30 minutes
- 🚀 Production deployment
- 📊 Need all 45 classes
- 💯 Want best quantum accuracy

### Use FULL if:
- ⏰ You have 2-4 hours
- 🔄 Starting from scratch
- 📁 Have new data
- 🎓 Research project

---

## 🎯 Summary

**Quick test:** `python train_quantum_toy.py`  
**Fast demo:** `python train_quantum_simple.py`  
**Production:** `python train_quantum_only.py`  
**Complete:** `python src/train_model.py`

**All versions work with your existing models and data!**

---

**Ready to train? Pick one and run it!** 🚀🔮

