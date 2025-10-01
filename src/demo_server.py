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
class_names = []
device = None
transform = None

def load_models():
    """Load all model artifacts."""
    global model, pca, classifier, class_names, device, transform
    
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
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n✓ All models loaded successfully!\n")

def predict_image(image):
    """
    Predict disease from image.
    
    Args:
        image: PIL Image
    
    Returns:
        dict with prediction results
    """
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract embeddings
    with torch.no_grad():
        embeddings = model.get_embeddings(img_tensor)
        embeddings = embeddings.cpu().numpy()
    
    # Apply PCA
    embeddings_pca = pca.transform(embeddings)
    
    # Predict with classifier
    prediction = classifier.predict(embeddings_pca)[0]
    probabilities = classifier.predict_proba(embeddings_pca)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    
    results = {
        'prediction': class_names[prediction],
        'confidence': float(probabilities[prediction]),
        'top_predictions': [
            {
                'class': class_names[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top_indices
        ]
    }
    
    return results

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

