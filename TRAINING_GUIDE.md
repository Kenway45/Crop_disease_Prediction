# 🎓 Complete Training Guide

## ✅ Setup Complete!

All files are ready and the training environment is configured:

- ✅ **56,134 images** downloaded across **45 disease classes**
- ✅ **MPS (Apple GPU)** detected and configured
- ✅ Training optimized for **10 epochs** (faster than 20)
- ✅ All dependencies installed
- ✅ Project structure created

## 🚀 Run Training (Choose One Method)

### Method 1: Using the Script (Recommended)
```bash
cd /Users/jayadharunr/Crop_disease_Prediction
./run_training.sh
```

This will:
- Display progress in real-time
- Save logs to `training.log`
- Show estimated completion time

### Method 2: Direct Python Command
```bash
cd /Users/jayadharunr/Crop_disease_Prediction
python3 src/train_model.py
```

### Method 3: Background Training
```bash
cd /Users/jayadharunr/Crop_disease_Prediction
./train_background.sh

# Monitor progress:
tail -f training.log
```

## ⏱️ Expected Timeline

With Apple MPS (GPU):
- **Epoch 1-2**: ~15-20 minutes each (model learning initial patterns)
- **Epoch 3-10**: ~10-15 minutes each (fine-tuning)
- **Total CNN Training**: 1-2 hours
- **Embedding Extraction**: 15-20 minutes
- **PCA Training**: 2-3 minutes
- **Classifier Training**: 1-2 minutes

**Total Time**: ~1.5-2.5 hours

## 📊 Training Steps

The training will progress through these 5 steps:

### Step 1: Collecting Dataset ⏱️ ~2 minutes
```
✓ Found 56,134 images across 45 classes
✓ Split into train (44,907) and validation (11,227)
✓ Saved class names to artifacts/models/classes.json
```

### Step 2: Training CNN ⏱️ ~1-2 hours
```
Epoch 1/10: Training ResNet18 backbone
Epoch 2/10: Model learning disease patterns
...
Epoch 10/10: Final fine-tuning
✓ Best validation accuracy: 85-95%
✓ Saved to artifacts/models/best_cnn.pt
```

### Step 3: Extracting Embeddings ⏱️ ~15-20 minutes
```
Processing 56,134 images...
✓ Saved embeddings: (56134, 512)
✓ Saved to artifacts/embeddings/train_emb.npy
```

### Step 4: Training PCA ⏱️ ~2-3 minutes
```
Reducing 512 dimensions to 128
✓ PCA trained and saved
✓ Explained variance: ~95%
✓ Saved to artifacts/pca/pca.joblib
```

### Step 5: Training Classifier ⏱️ ~1-2 minutes
```
Training Logistic Regression on PCA features
✓ Train accuracy: 90-95%
✓ Test accuracy: 85-90%
✓ Saved to artifacts/classifiers/lr_clf.joblib
```

## 📝 Monitoring Progress

### Check if Training is Running
```bash
ps aux | grep train_model.py
```

### View Real-time Logs
```bash
tail -f training.log
```

### Check Artifacts Being Created
```bash
ls -lh artifacts/models/
ls -lh artifacts/embeddings/
ls -lh artifacts/pca/
ls -lh artifacts/classifiers/
```

### View Progress Summary
```bash
grep "Epoch" training.log
grep "✓" training.log
```

## 🎯 Expected Output

### During Training (Per Epoch)
```
Epoch 1/10 [Train]: 100%|███████| 1404/1404 [12:34<00:00, loss: 1.234, acc: 75.50%]
Epoch 1/10 [Val]:   100%|███████| 351/351 [02:15<00:00, loss: 0.987, acc: 78.20%]

Epoch 1: Train Acc: 75.50%, Val Acc: 78.20%
✓ Saved best model with validation accuracy: 78.20%
```

### After Training Completes
```
============================================================
TRAINING COMPLETE!
============================================================

Artifacts created:
  ✓ artifacts/models/best_cnn.pt
  ✓ artifacts/models/classes.json
  ✓ artifacts/embeddings/train_emb.npy
  ✓ artifacts/embeddings/train_labels.npy
  ✓ artifacts/pca/pca.joblib
  ✓ artifacts/classifiers/lr_clf.joblib

✓ All files are ready for demo_server.py!
```

## ✅ Verify Training Success

Run this after training completes:

```bash
cd /Users/jayadharunr/Crop_disease_Prediction

# Check all artifacts exist
echo "Checking artifacts..."
test -f artifacts/models/best_cnn.pt && echo "✓ CNN model exists"
test -f artifacts/models/classes.json && echo "✓ Classes file exists"
test -f artifacts/embeddings/train_emb.npy && echo "✓ Embeddings exist"
test -f artifacts/embeddings/train_labels.npy && echo "✓ Labels exist"
test -f artifacts/pca/pca.joblib && echo "✓ PCA exists"
test -f artifacts/classifiers/lr_clf.joblib && echo "✓ Classifier exists"

# Check file sizes
echo ""
echo "File sizes:"
ls -lh artifacts/models/best_cnn.pt
ls -lh artifacts/embeddings/train_emb.npy
ls -lh artifacts/pca/pca.joblib
ls -lh artifacts/classifiers/lr_clf.joblib
```

## 🎬 After Training: Run the Demo

Once all artifacts are created:

```bash
python3 src/demo_server.py
```

Then open: **http://localhost:5000**

## 🔧 Troubleshooting

### Training Interrupted?
Just restart it - it will use any existing artifacts:
```bash
python3 src/train_model.py
```

### Out of Memory?
Edit `src/train_model.py` line 239:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### Training Too Slow?
The script is already optimized to use MPS (Apple GPU). If it's still slow:
- Close other applications
- Make sure your Mac is plugged in (not on battery)
- Check Activity Monitor for high CPU/Memory usage

### Want to Test with Fewer Epochs?
Edit `src/train_model.py` line 240:
```python
NUM_EPOCHS = 5  # Reduce from 10
```

### Want to Use Subset of Data?
Add this after line 273 in `src/train_model.py`:
```python
# Use only 10,000 images for quick testing
if len(image_paths) > 10000:
    import random
    random.seed(42)
    indices = random.sample(range(len(image_paths)), 10000)
    image_paths = [image_paths[i] for i in indices]
    labels = [labels[i] for i in indices]
```

## 📊 Expected Performance

### Training Metrics
- **Initial Accuracy** (Epoch 1): 70-80%
- **Final Accuracy** (Epoch 10): 85-95%
- **Validation Accuracy**: 80-92%
- **Classifier Accuracy**: 85-90%

### Model Sizes
- **CNN Model**: ~45 MB
- **Embeddings**: ~110 MB
- **PCA Model**: ~5 MB
- **Classifier**: ~1 MB
- **Total**: ~160 MB

## 🎉 Success Checklist

After training completes:

- [ ] All 6 artifact files created
- [ ] Training log shows no errors
- [ ] Validation accuracy > 80%
- [ ] All files have reasonable sizes (not 0 KB)
- [ ] Demo server loads without errors
- [ ] Predictions work with confidence scores

## 📞 Need Help?

1. Check `training.log` for error messages
2. Verify all dependencies: `pip list | grep -E "torch|sklearn|flask"`
3. Check disk space: `df -h`
4. Check memory: `top` or Activity Monitor
5. Review README.md for additional help

---

**Ready to Train?**

```bash
cd /Users/jayadharunr/Crop_disease_Prediction
./run_training.sh
```

**Estimated completion time**: 1.5-2.5 hours

Good luck! 🌱🔬

