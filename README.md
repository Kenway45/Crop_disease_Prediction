# 🔮 Quantum-Enhanced Crop Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple.svg)](https://pennylane.ai)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)

> **Pure quantum machine learning system for crop disease detection using 8-qubit variational quantum circuits with PCA-enabled processing and intelligent leaf detection.**

**🆕 Version 2.0: Pure Quantum System with Visual Disease Detection!**

## ✨ Key Features

### 🔬 Pure Quantum Classification
- **8-qubit Variational Quantum Circuit (VQC)** with optimized architecture
- **PennyLane quantum framework** with lightning-fast simulation
- **PCA-enabled quantum processing** (2048→8 dimensions)
- **Pure quantum predictions** - no classical fallback

### 🍃 Intelligent Leaf Detection
- Automatic leaf/non-leaf image discrimination
- Green dominance analysis with texture heuristics
- Brightness validation
- Rejects invalid images before processing

### 🎨 Beautiful Black & White UI
- **Modern minimalist interface** with dark/light theme toggle
- **Real-time camera integration** (mobile & desktop)
- **Visual disease detection** with bounding boxes
- **Red overlay highlighting** on diseased regions
- **Confidence color coding** (green/yellow/red)

### 📊 Advanced Visualization
- Live camera preview
- Captured image display with overlays
- 🟢 Green bounding box for detected leaf
- 🔴 Red semi-transparent overlay on disease regions
- Top-3 quantum predictions with animated confidence bars

## 🏗️ Architecture

### Complete Pipeline:
```
Input Image
    ↓
🍃 Leaf Detection (Heuristic)
    ↓
CNN Feature Extraction (ResNet-50)
    ↓
PCA Dimensionality Reduction (2048→8)
    ↓
🔮 8-Qubit VQC Classification
    ↓
Disease Prediction + Confidence + Visualization
```

### Quantum Circuit:
- **8 qubits** encoded with PCA features
- **Strongly Entangling Layers** with 3 repetitions
- **Amplitude encoding** for classical→quantum mapping
- **Lightning simulator** for high performance

## 📦 Datasets

The system is trained on three major plant disease datasets from Kaggle:

1. **PlantVillage** - 38 classes of plant diseases
2. **Rice Leaf Diseases** - 4 classes of rice diseases  
3. **Cotton Leaf Disease** - 4 classes of cotton diseases

Total: ~50,000+ images across 40+ disease classes

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Kaggle account with API credentials
- 4GB+ RAM
- GPU (optional, but recommended for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Crop_disease_Prediction.git
cd Crop_disease_Prediction
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt

# Quantum dependencies (required for pure quantum system):
pip install pennylane pennylane-lightning
```

4. **Set up Kaggle API**

The kaggle.json file should already be set up in `~/.kaggle/kaggle.json`

If not, create it with:
```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```

Then set permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Training the Model

1. **Download datasets**
```bash
python src/download_data.py
```
This will download all three datasets (~5GB total). It may take 10-30 minutes depending on your internet speed.

2. **Train the complete model**
```bash
# Full training (CNN + Classical + Quantum)
python src/train_model.py
```
(Takes 2-4 hours)

**OR train quantum only (faster):**
```bash
# Quick quantum-only training (if you have pre-trained CNN)
python train_quantum_only.py
```
(Takes 15-30 minutes)

Training steps:
- Collects and preprocesses images
- Trains CNN (ResNet-50 backbone)
- Extracts embeddings from all images
- Trains PCA for dimensionality reduction (2048→8)
- **Trains pure quantum classifier** (8-qubit VQC) 🔮

All artifacts will be saved to `artifacts/` directory.

### Running the Demo

1. **Start the Flask server**
```bash
python src/demo_server.py
```

2. **Open your browser**
```
http://localhost:8080
```

3. **Use the camera**
   - Click **"📷 Start Camera"** to activate your camera
   - Point at a **plant leaf**
   - Click **"📸 Capture & Analyze"**
   - View quantum predictions with visual overlays
   - Toggle theme with 🌓 button (top right)

### UI Features:
- 🟢 **Green bounding box** - detected leaf region
- 🔴 **Red overlay** - disease regions highlighted
- 🎨 **Theme toggle** - switch between light/dark modes
- 📊 **Animated bars** - top-3 quantum predictions
- 🔮 **Confidence colors** - green (high), yellow (medium), red (low)

### Testing on Mobile

To test on your phone:

1. **Make sure your phone and computer are on the same network**

2. **Find your computer's IP address**
```bash
# On Mac/Linux:
ifconfig | grep "inet " | grep -v 127.0.0.1

# On Windows:
ipconfig
```

3. **Start the server**
```bash
python src/demo_server.py
```

4. **Open on your phone's browser**
```
http://YOUR_IP_ADDRESS:8080
```

5. **Grant camera permissions when prompted**

6. **Enjoy the responsive mobile UI with full visualization features!**

## 📁 Project Structure

```
Crop_disease_Prediction/
├── src/
│   ├── download_data.py           # Download datasets from Kaggle
│   ├── train_model.py             # Complete training pipeline
│   └── demo_server.py             # Flask web server (PURE QUANTUM)
├── templates/
│   └── index.html                 # Modern black/white UI with theme toggle
├── static/
│   └── app.js                     # Camera, visualization & overlay logic
├── artifacts/
│   ├── models/
│   │   ├── best_model.pt          # Trained CNN model
│   │   └── classes.json           # Class names
│   ├── embeddings/
│   │   ├── train_emb.npy          # Training embeddings
│   │   └── train_labels.npy       # Training labels
│   ├── pca/
│   │   └── pca_transform.joblib   # PCA transformer (2048→8)
│   └── classifiers/
│       └── quantum_classifier_full.joblib  # Pure quantum classifier 🔮
├── data/                          # Downloaded datasets (not in git)
├── train_quantum_only.py          # Quick quantum-only training
├── requirements.txt               # Python dependencies
├── GITHUB_DEPLOYMENT.md           # GitHub deployment guide 🚀
├── PURE_QUANTUM_SYSTEM.md         # Pure quantum system documentation 🔮
├── QUANTUM_GUIDE.md               # Technical quantum guide 🔬
├── QUANTUM_TRAINING_OPTIONS.md    # Training options guide
└── README.md                      # This file
```

## ✅ Pre-Launch Checklist

Before running the live demo, verify:

- [ ] `artifacts/models/best_model.pt` exists (CNN model)
- [ ] `artifacts/models/classes.json` exists
- [ ] `artifacts/embeddings/train_emb.npy` exists
- [ ] `artifacts/embeddings/train_labels.npy` exists
- [ ] `artifacts/pca/pca_transform.joblib` exists (2048→8 PCA)
- [ ] `artifacts/classifiers/quantum_classifier_full.joblib` exists (8-qubit VQC) 🔮
- [ ] All dependencies installed including PennyLane (`pip list`)
- [ ] Flask server starts without errors
- [ ] Browser allows camera access
- [ ] Server is reachable on network

## 🔧 Troubleshooting

### Camera not working
- Make sure you're using HTTPS or localhost (HTTP)
- Check browser permissions for camera access
- Try a different browser (Chrome/Firefox recommended)

### Model not loading
- Verify all files exist in `artifacts/` directory
- Check file permissions
- Run quantum training: `python train_quantum_only.py`

### "No leaf detected" error
- Make sure you're capturing a **plant leaf** (not other objects)
- Ensure good lighting conditions
- The leaf should be green and fill most of the frame
- System uses heuristics: green dominance, texture, brightness

### Low quantum confidence
- Image quality may be poor - try better lighting
- Leaf may be too small in frame - zoom in
- System warns if confidence < 30%

### Out of memory
- Reduce batch size in training scripts
- Use smaller PCA components (currently 8 for quantum)
- Use CPU instead of GPU if GPU memory is limited

## 📊 Model Performance

Expected performance metrics (will vary based on training):

- **Quantum Training Accuracy**: 75-90%
- **Quantum Validation Accuracy**: 70-85%
- **Inference Time**: 2-5 seconds per image (including quantum circuit evaluation)
- **Model Size**: ~100MB (CNN) + <1MB (PCA) + ~500KB (quantum)
- **Leaf Detection Accuracy**: ~95%

**Quantum Advantage**: 
- ✨ Exponential expressivity with 8 qubits
- 🔗 Quantum entanglement captures complex disease patterns
- 🎯 Handles high-dimensional feature spaces efficiently
- 🔮 Pure quantum processing - no classical fallback needed

## 📚 Complete Documentation

For detailed guides, see:

- **[GITHUB_DEPLOYMENT.md](GITHUB_DEPLOYMENT.md)** - How to deploy to GitHub
- **[PURE_QUANTUM_SYSTEM.md](PURE_QUANTUM_SYSTEM.md)** - Pure quantum system overview
- **[QUANTUM_GUIDE.md](QUANTUM_GUIDE.md)** - Technical quantum circuit details
- **[QUANTUM_TRAINING_OPTIONS.md](QUANTUM_TRAINING_OPTIONS.md)** - Training options

## 🔬 Advanced Usage

### Adjusting Quantum Circuit

Edit `train_quantum_only.py`:

```python
N_QUBITS = 8             # Number of qubits (match PCA components)
N_LAYERS = 3             # Quantum circuit depth
PCA_COMPONENTS = 8       # PCA dimensions for quantum input
```

### Adjusting Leaf Detection

Edit `src/demo_server.py`:

```python
green_ratio > 0.30       # Minimum green dominance (30%)
variance > 100           # Texture threshold
brightness 30-240        # Valid brightness range
```

### Using Different Datasets

Add more datasets in `src/download_data.py`:

```python
DATASETS = [
    # ... existing datasets
    {
        'name': 'my-dataset',
        'kaggle_id': 'username/dataset-name',
        'zip_name': 'my-dataset.zip'
    }
]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **PlantVillage** dataset by Penn State University
- **Rice Leaf Diseases** dataset on Kaggle
- **Cotton Leaf Disease** dataset on Kaggle
- PyTorch and torchvision teams
- Flask web framework
- **PennyLane** for quantum machine learning capabilities 🔮
- Xanadu Quantum for advancing QML research

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

## 🆕 What's New in Version 2.0

### Pure Quantum System:
- ✅ Removed classical classifier - **pure quantum predictions only**
- ✅ Optimized 8-qubit VQC with strongly entangling layers
- ✅ PCA-enabled quantum processing (2048→8 dimensions)
- ✅ Intelligent leaf detection with heuristic validation

### Enhanced UI:
- ✅ **Beautiful black & white theme** with toggle button 🌓
- ✅ **Dark mode** - automatic theme persistence
- ✅ **Visual bounding boxes** - green for leaf detection
- ✅ **Disease region overlay** - red semi-transparent highlights
- ✅ **Confidence color coding** - green/yellow/red indicators
- ✅ **Animated predictions** - smooth confidence bars

### New Features:
- ✅ Captured image display with overlays
- ✅ Real-time visualization of disease regions
- ✅ Enhanced error handling with warnings
- ✅ Mobile-responsive design improvements
- ✅ Complete documentation for GitHub deployment

---

## 🚀 Quick Commands

```bash
# Clone & Setup
git clone <repo> && cd Crop_disease_Prediction
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pennylane pennylane-lightning

# Train Quantum Model
python train_quantum_only.py

# Run Demo Server
python src/demo_server.py

# Deploy to GitHub
git add . && git commit -m "Add quantum system" && git push origin main
```

---

**Built with 💚 for sustainable agriculture using quantum computing**

🔮 **Quantum + AI = Future of Farming** 🌱

---

**Happy Disease Detection! 🌱🔬**

