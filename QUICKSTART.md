# 🚀 Quick Start Guide

## Current Status

✅ **Project repository created**: https://github.com/Kenway45/Crop_disease_Prediction  
✅ **All code files pushed to GitHub**  
✅ **Kaggle API configured**  
✅ **Datasets downloaded**: 56,134 images across 45 disease classes  
✅ **Dependencies installed**  
⏳ **Training started** (will take 2-4 hours on GPU, longer on CPU)

## 📊 Dataset Summary

- **PlantVillage**: 38 plant disease classes (tomato, apple, grape, etc.)
- **Rice Leaf Diseases**: 4 rice disease classes
- **Cotton Leaf Disease**: 4 cotton disease classes
- **Total**: 56,134 images, 45 disease classes

## 🎯 Next Steps

### Option 1: Monitor Training Progress

```bash
cd /Users/jayadharunr/Crop_disease_Prediction

# Monitor training in real-time
tail -f training.log

# Or run in foreground (if not already running)
python3 src/train_model.py
```

### Option 2: Run Training in Background

```bash
cd /Users/jayadharunr/Crop_disease_Prediction
./train_background.sh

# Then monitor:
tail -f training.log
```

### Option 3: Check Training Status

```bash
# Check if training is running
ps aux | grep train_model.py

# Check if artifacts are being created
ls -lah artifacts/models/
ls -lah artifacts/embeddings/
ls -lah artifacts/pca/
ls -lah artifacts/classifiers/
```

## 🎬 After Training Completes

Once training is complete, you should see these files:

```
artifacts/
├── models/
│   ├── best_cnn.pt          ✓ (Created during training)
│   └── classes.json         ✓ (Already created)
├── embeddings/
│   ├── train_emb.npy        ⏳ (Will be created)
│   └── train_labels.npy     ⏳ (Will be created)
├── pca/
│   └── pca.joblib           ⏳ (Will be created)
└── classifiers/
    └── lr_clf.joblib        ⏳ (Will be created)
```

### Run the Demo Server

```bash
cd /Users/jayadharunr/Crop_disease_Prediction
python3 src/demo_server.py
```

Then open: **http://localhost:5000**

### Test on Mobile Phone

1. Find your computer's IP address:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

2. Start the server:
```bash
python3 src/demo_server.py
```

3. On your phone, open browser and go to:
```
http://YOUR_IP_ADDRESS:5000
```

4. Grant camera permissions when prompted

5. Point camera at plant leaf and click "Capture & Analyze"

## 📦 Project Files on GitHub

All files are now available at:
**https://github.com/Kenway45/Crop_disease_Prediction**

### Files Pushed:
- ✅ `src/download_data.py` - Dataset download script
- ✅ `src/train_model.py` - Complete training pipeline
- ✅ `src/demo_server.py` - Flask web server
- ✅ `templates/index.html` - Web UI
- ✅ `static/app.js` - Camera & prediction JavaScript
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Comprehensive documentation
- ✅ `.gitignore` - Git ignore rules

### Not Pushed (Too Large):
- ❌ `data/` - Datasets (5GB+)
- ❌ `artifacts/*.pt` - Model files (45MB+)
- ❌ `artifacts/*.npy` - Embeddings (100MB+)
- ❌ `artifacts/*.joblib` - PCA & classifiers (5MB+)

## 🔧 Troubleshooting

### Training Taking Too Long?
- Reduce epochs: Edit `src/train_model.py` and change `NUM_EPOCHS = 20` to `NUM_EPOCHS = 10`
- Reduce batch size if running out of memory: Change `BATCH_SIZE = 32` to `BATCH_SIZE = 16`

### Out of Memory?
```python
# In src/train_model.py, line 239:
BATCH_SIZE = 16  # Reduce from 32
```

### Want to Use a Subset of Data?
Add this to `collect_dataset()` function in `src/train_model.py`:
```python
# After line 132, add:
if len(image_paths) > 10000:  # Use only 10k images
    indices = np.random.choice(len(image_paths), 10000, replace=False)
    image_paths = [image_paths[i] for i in indices]
    labels = [labels[i] for i in indices]
```

## 📞 Getting Help

1. Check the main README: `README.md`
2. Check training logs: `training.log`
3. Check GitHub issues: https://github.com/Kenway45/Crop_disease_Prediction/issues

## 🎉 Success Checklist

After everything is complete:

- [ ] Training finished without errors
- [ ] All 6 artifact files created
- [ ] Demo server starts successfully
- [ ] Camera works in browser
- [ ] Predictions return results
- [ ] Mobile testing works
- [ ] Confidence scores displayed

---

**Training Started**: Check `training.log` for progress  
**Repository**: https://github.com/Kenway45/Crop_disease_Prediction  
**Local Path**: `/Users/jayadharunr/Crop_disease_Prediction`

