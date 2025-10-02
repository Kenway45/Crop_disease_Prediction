"""
Flask server for crop disease prediction demo with live camera.
"""
import os
import json
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify
import joblib

# Import quantum classifier
try:
    from quantum_classifier import QuantumClassifier
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

# Define the CNN model (must match training)
class CropDiseaseCNN(nn.Module):
    """CNN model for crop disease classification."""
    
    def __init__(self, num_classes):
        super(CropDiseaseCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.embedding_size = 512
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_embeddings(self, x):
        """Extract embeddings before final classification layer."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        if isinstance(self.backbone.fc, nn.Sequential):
            x = self.backbone.fc[0](x)  # Dropout
            x = self.backbone.fc[1](x)  # Linear -> embeddings
            x = self.backbone.fc[2](x)  # ReLU
        
        return x

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Global variables for model components
model = None
pca = None
classifier = None
quantum_classifier = None
class_names = []
device = None
transform = None

def load_models():
    """Load all model artifacts."""
    global model, pca, classifier, quantum_classifier, class_names, device, transform
    
    project_root = Path(__file__).parent.parent
    artifacts_dir = project_root / 'artifacts'
    
    print("Loading models...")
    
    # Device - Use MPS (Apple GPU) if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load class names
    with open(artifacts_dir / 'models' / 'classes.json', 'r') as f:
        class_names = json.load(f)
    print(f"✓ Loaded {len(class_names)} classes")
    
    # Load CNN
    model = CropDiseaseCNN(len(class_names))
    checkpoint = torch.load(artifacts_dir / 'models' / 'best_cnn.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Loaded CNN model")
    
    # Load PCA
    pca = joblib.load(artifacts_dir / 'pca' / 'pca.joblib')
    print(f"✓ Loaded PCA ({pca.n_components} components)")
    
    # Load classifier
    classifier = joblib.load(artifacts_dir / 'classifiers' / 'lr_clf.joblib')
    print("✓ Loaded classifier")
    
    # Load Quantum Classifier (optional)
    quantum_clf_path = artifacts_dir / 'classifiers' / 'quantum_clf.joblib'
    if QUANTUM_AVAILABLE and quantum_clf_path.exists():
        try:
            quantum_classifier = QuantumClassifier.load(quantum_clf_path)
            print("✓ Loaded quantum classifier 🔮")
        except Exception as e:
            print(f"⚠️  Failed to load quantum classifier: {e}")
            quantum_classifier = None
    else:
        quantum_classifier = None
        if not QUANTUM_AVAILABLE:
            print("ℹ️  Quantum classifier not available (install pennylane)")
        else:
            print("ℹ️  Quantum classifier not found (train with quantum support)")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n✓ All models loaded successfully!\n")

def is_leaf_image(image):
    """
    Detect if image contains a leaf using color and texture analysis.
    
    Args:
        image: PIL Image
    
    Returns:
        bool: True if leaf detected, False otherwise
    """
    import numpy as np
    from PIL import ImageStat
    
    # Convert to RGB array
    img_array = np.array(image)
    
    # Calculate green channel dominance
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Leaves typically have more green
    green_ratio = np.mean(g) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)
    
    # Calculate color variance (leaves have some texture)
    stats = ImageStat.Stat(image)
    variance = np.mean(stats.var)
    
    # Heuristics for leaf detection
    is_green_dominant = green_ratio > 0.30  # At least 30% green
    has_texture = variance > 100  # Some texture/variation
    not_too_uniform = variance < 10000  # Not completely random
    
    # Check brightness (not too dark, not too bright)
    brightness = np.mean(img_array)
    reasonable_brightness = 30 < brightness < 240
    
    return is_green_dominant and has_texture and not_too_uniform and reasonable_brightness

def predict_image(image):
    """
    Predict disease from image using PURE QUANTUM + PCA pipeline.
    Rejects non-leaf images.
    
    Args:
        image: PIL Image
    
    Returns:
        dict with prediction results
    """
    # STEP 1: Leaf Detection
    if not is_leaf_image(image):
        return {
            'error': 'No leaf detected',
            'message': '🍃 Please capture an image of a plant leaf',
            'is_leaf': False,
            'quantum_available': False
        }
    
    # STEP 2: Extract features (CNN + PCA)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embeddings = model.get_embeddings(img_tensor)
        embeddings = embeddings.cpu().numpy()
    
    # Apply PCA (critical for quantum!)
    embeddings_pca = pca.transform(embeddings)
    
    # STEP 3: Pure Quantum Prediction
    if quantum_classifier is None:
        return {
            'error': 'Quantum classifier not available',
            'message': 'Please train quantum model first: python train_quantum_only.py',
            'is_leaf': True,
            'quantum_available': False
        }
    
    try:
        # Pure quantum prediction
        quantum_probabilities = quantum_classifier.predict_proba(embeddings_pca)[0]
        quantum_prediction = int(np.argmax(quantum_probabilities))
        
        # Get top 3 quantum predictions
        quantum_top_indices = np.argsort(quantum_probabilities)[-3:][::-1]
        
        # Calculate confidence threshold
        max_confidence = float(quantum_probabilities[quantum_prediction])
        
        # Check if prediction is confident enough
        if max_confidence < 0.3:  # Low confidence
            return {
                'warning': 'Low confidence prediction',
                'message': '⚠️ Image quality may be poor. Please try a clearer image of the leaf.',
                'prediction': class_names[quantum_prediction],
                'confidence': max_confidence,
                'is_leaf': True,
                'quantum_available': True
            }
        
        # Return pure quantum results
        results = {
            'prediction': class_names[quantum_prediction],
            'confidence': max_confidence,
            'is_leaf': True,
            'quantum_available': True,
            'method': 'Pure Quantum (PCA + 8-qubit VQC)',
            'quantum_prediction': class_names[quantum_prediction],
            'quantum_confidence': max_confidence,
            'top_predictions': [
                {
                    'class': class_names[idx],
                    'confidence': float(quantum_probabilities[idx])
                }
                for idx in quantum_top_indices
            ]
        }
        
        return results
        
    except Exception as e:
        print(f"Quantum prediction failed: {e}")
        return {
            'error': 'Prediction failed',
            'message': f'Error: {str(e)}',
            'is_leaf': True,
            'quantum_available': False
        }

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', num_classes=len(class_names))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Make prediction
        results = predict_image(image)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model is not None,
        'num_classes': len(class_names)
    })

if __name__ == '__main__':
    # Load models before starting server
    load_models()
    
    # Start Flask server
    print("="*60)
    print("Starting Crop Disease Prediction Demo Server")
    print("="*60)
    print("\nOpen your browser and go to: http://localhost:8080")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False)

