#!/usr/bin/env python3
"""
Train only the quantum classifier using existing CNN, PCA, and embeddings.
This is much faster than full training (15-30 min vs 2-4 hours).
"""
import sys
sys.path.insert(0, 'src')

import os
import numpy as np
from pathlib import Path
from quantum_classifier import QuantumClassifier
from sklearn.model_selection import train_test_split

print("=" * 60)
print("🔮 Training Quantum Classifier Only")
print("=" * 60)

# Paths
project_root = Path(__file__).parent
artifacts_dir = project_root / 'artifacts'

# Check if required files exist
print("\n1. Checking for required files...")
required_files = [
    artifacts_dir / 'embeddings' / 'train_emb.npy',
    artifacts_dir / 'embeddings' / 'train_labels.npy',
    artifacts_dir / 'pca' / 'pca.joblib',
    artifacts_dir / 'models' / 'classes.json'
]

missing_files = []
for file_path in required_files:
    if file_path.exists():
        print(f"   ✓ {file_path.name}")
    else:
        print(f"   ❌ {file_path.name} - MISSING")
        missing_files.append(file_path)

if missing_files:
    print("\n❌ Missing required files. Please run full training first:")
    print("   python src/train_model.py")
    sys.exit(1)

# Load embeddings and labels
print("\n2. Loading embeddings and labels...")
embeddings = np.load(artifacts_dir / 'embeddings' / 'train_emb.npy')
labels = np.load(artifacts_dir / 'embeddings' / 'train_labels.npy')
print(f"   ✓ Loaded {len(embeddings)} samples")
print(f"   Shape: {embeddings.shape}")

# Load PCA
print("\n3. Loading PCA...")
import joblib
pca = joblib.load(artifacts_dir / 'pca' / 'pca.joblib')
print(f"   ✓ PCA with {pca.n_components} components")

# Transform embeddings with PCA
print("\n4. Applying PCA transformation...")
embeddings_pca = pca.transform(embeddings)
print(f"   ✓ Transformed to shape: {embeddings_pca.shape}")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Get number of classes
import json
with open(artifacts_dir / 'models' / 'classes.json', 'r') as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"\n5. Dataset info:")
print(f"   Classes: {num_classes}")
print(f"   Total samples: {len(embeddings_pca)}")

# Sample balanced subset for quantum training
print("\n6. Preparing quantum training data...")
max_samples_per_class = 100  # Adjust this for speed/accuracy tradeoff

quantum_indices = []
for class_label in range(num_classes):
    class_indices = np.where(labels == class_label)[0]
    n_samples = min(len(class_indices), max_samples_per_class)
    sampled = np.random.choice(class_indices, n_samples, replace=False)
    quantum_indices.extend(sampled)

quantum_indices = np.array(quantum_indices)
np.random.shuffle(quantum_indices)

X_quantum = embeddings_pca[quantum_indices]
y_quantum = labels[quantum_indices]

print(f"   ✓ Using {len(X_quantum)} samples ({max_samples_per_class} per class)")
print(f"   This represents {100*len(X_quantum)/len(embeddings_pca):.1f}% of data")

# Split data
print("\n7. Splitting into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_quantum, y_quantum, test_size=0.2, random_state=42, stratify=y_quantum
)
print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# Initialize quantum classifier
print("\n8. Initializing quantum classifier...")
print("   This will create an 8-qubit circuit...")
quantum_clf = QuantumClassifier(
    n_features=pca.n_components,
    n_classes=num_classes,
    n_qubits=8,
    n_layers=4
)
print(f"   ✓ Quantum classifier initialized")
print(f"   Qubits: {quantum_clf.n_qubits}")
print(f"   Layers: {quantum_clf.n_layers}")
print(f"   Parameters: {quantum_clf.n_layers * quantum_clf.n_qubits * 2}")

# Train quantum classifier
print("\n9. Training quantum classifier...")
print("   This will take approximately 15-30 minutes...")
print("   You can reduce epochs or samples_per_class for faster training")
print()

try:
    quantum_clf.fit(
        X_train, 
        y_train,
        epochs=30,
        batch_size=16,
        learning_rate=0.01,
        verbose=True
    )
    
    # Evaluate
    print("\n10. Evaluating quantum classifier...")
    train_acc = quantum_clf.score(X_train, y_train)
    test_acc = quantum_clf.score(X_test, y_test)
    
    print(f"   ✓ Train accuracy: {train_acc*100:.2f}%")
    print(f"   ✓ Test accuracy: {test_acc*100:.2f}%")
    
    # Save
    print("\n11. Saving quantum classifier...")
    quantum_clf.save(artifacts_dir / 'classifiers' / 'quantum_clf.joblib')
    print(f"   ✓ Saved to: artifacts/classifiers/quantum_clf.joblib")
    
    # Check file size
    file_size = (artifacts_dir / 'classifiers' / 'quantum_clf.joblib').stat().st_size
    print(f"   File size: {file_size / 1024:.1f} KB")
    
    print("\n" + "=" * 60)
    print("✓ Quantum Classifier Training Complete!")
    print("=" * 60)
    print()
    print("What's next?")
    print("  1. Run demo server: python src/demo_server.py")
    print("  2. The server will now use BOTH classical and quantum predictions")
    print("  3. Open http://localhost:5000 in your browser")
    print()
    print("Quantum predictions will appear alongside classical predictions!")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    print("\nTroubleshooting:")
    print("  - Try reducing max_samples_per_class (line 69)")
    print("  - Try reducing epochs (line 123)")
    print("  - Check if PennyLane is installed: pip install pennylane")
    sys.exit(1)

