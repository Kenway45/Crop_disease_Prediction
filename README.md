# 🌱 Crop Disease Prediction System

A complete end-to-end deep learning system for identifying crop diseases from plant leaf images using live camera feed. The system uses a CNN for feature extraction, PCA for dimensionality reduction, and a logistic regression classifier for final predictions.

## 📋 Features

- **Live Camera Detection**: Real-time disease detection using device camera (mobile & desktop)
- **Multi-Crop Support**: Trained on PlantVillage, Rice, and Cotton disease datasets
- **High Accuracy**: Uses ResNet18 backbone with fine-tuned classifier
- **Fast Inference**: Optimized pipeline with PCA dimensionality reduction
- **Modern UI**: Beautiful, responsive web interface
- **Mobile-Friendly**: Works on phones, tablets, and desktops

## 🏗️ Architecture

```
Input Image (224x224x3)
    ↓
ResNet18 CNN (Pretrained on ImageNet)
    ↓
Embeddings (512-dimensional)
    ↓
PCA (128 components)
    ↓
Logistic Regression Classifier
    ↓
Disease Prediction + Confidence Scores
```

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
git clone https://github.com/Kenway45/Crop_disease_Prediction.git
cd Crop_disease_Prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
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

2. **Train the model**
```bash
python src/train_model.py
```

Training steps:
- Collects and preprocesses images
- Trains CNN (20 epochs, ~2-4 hours on GPU, longer on CPU)
- Extracts embeddings from all images
- Trains PCA for dimensionality reduction
- Trains logistic regression classifier

All artifacts will be saved to `artifacts/` directory.

### Running the Demo

1. **Start the Flask server**
```bash
python src/demo_server.py
```

2. **Open your browser**
```
http://localhost:5000
```

3. **Use the camera**
   - Click "Start Camera" to activate your camera
   - Point at a plant leaf
   - Click "Capture & Analyze"
   - View prediction results with confidence scores

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
http://YOUR_IP_ADDRESS:5000
```

5. **Grant camera permissions when prompted**

## 📁 Project Structure

```
Crop_disease_Prediction/
├── src/
│   ├── download_data.py      # Download datasets from Kaggle
│   ├── train_model.py         # Complete training pipeline
│   └── demo_server.py         # Flask web server
├── templates/
│   └── index.html             # Web UI
├── static/
│   └── app.js                 # Camera and prediction logic
├── artifacts/
│   ├── models/
│   │   ├── best_cnn.pt        # Trained CNN model
│   │   └── classes.json       # Class names
│   ├── embeddings/
│   │   ├── train_emb.npy      # Training embeddings
│   │   └── train_labels.npy   # Training labels
│   ├── pca/
│   │   └── pca.joblib         # PCA transformer
│   └── classifiers/
│       └── lr_clf.joblib      # Logistic regression classifier
├── data/                      # Downloaded datasets (not in git)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ✅ Pre-Launch Checklist

Before running the live demo, verify:

- [ ] `artifacts/models/best_cnn.pt` exists
- [ ] `artifacts/models/classes.json` exists
- [ ] `artifacts/embeddings/train_emb.npy` exists
- [ ] `artifacts/embeddings/train_labels.npy` exists
- [ ] `artifacts/pca/pca.joblib` exists
- [ ] `artifacts/classifiers/lr_clf.joblib` exists
- [ ] All dependencies installed (`pip list`)
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
- Try rerunning training pipeline

### Low accuracy
- Need more training data
- Increase training epochs
- Try data augmentation
- Fine-tune hyperparameters

### Out of memory
- Reduce batch size in `train_model.py`
- Use smaller PCA components
- Use CPU instead of GPU if GPU memory is limited

## 📊 Model Performance

Expected performance metrics (will vary based on training):

- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Inference Time**: <1 second per image
- **Model Size**: ~45MB (CNN) + ~5MB (other artifacts)

## 🔬 Advanced Usage

### Adjusting Hyperparameters

Edit `src/train_model.py`:

```python
BATCH_SIZE = 32          # Batch size for training
NUM_EPOCHS = 20          # Number of training epochs
IMG_SIZE = 224           # Input image size
PCA_COMPONENTS = 128     # PCA dimensions
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

### Export to TensorFlow Lite (for mobile deployment)

Coming soon! Will enable on-device inference on smartphones and Raspberry Pi.

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

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy Disease Detection! 🌱🔬**

