#!/usr/bin/env python3
"""
Simplified Quantum Training - FAST VERSION
Uses fewer classes and simpler approach for quick demonstration.
Training time: ~5-10 minutes instead of 15-30 minutes
"""
import sys
sys.path.insert(0, 'src')

import os
import numpy as np
from pathlib import Path
from quantum_classifier import QuantumClassifier
from sklearn.model_selection import train_test_split
import joblib
import json

print("=" * 60)
print("🔮 FAST Quantum Classifier Training")
print("=" * 60)
print()
print("This version uses:")
print("  - Only TOP 10 disease classes (instead of 45)")
print("  - 50 samples per class (instead of 100)")
print("  - 15 epochs (instead of 30)")
print("  - Estimated time: 5-10 minutes")
print()

# Paths
project_root = Path(__file__).parent
artifacts_dir = project_root / 'artifacts'

# Load data
print("1. Loading data...")
embeddings = np.load(artifacts_dir / 'embeddings' / 'train_emb.npy')
labels = np.load(artifacts_dir / 'embeddings' / 'train_labels.npy')
pca = joblib.load(artifacts_dir / 'pca' / 'pca.joblib')

with open(artifacts_dir / 'models' / 'classes.json', 'r') as f:
    all_classes = json.load(f)

print(f"   ✓ Loaded {len(embeddings)} total samples")

# Transform with PCA
embeddings_pca = pca.transform(embeddings)
print(f"   ✓ PCA transformed to {embeddings_pca.shape[1]}D")

# Select top 10 most common classes
print("\n2. Selecting top 10 disease classes...")
unique, counts = np.unique(labels, return_counts=True)
top_10_indices = np.argsort(counts)[-10:]  # Top 10 most common
top_10_classes = unique[top_10_indices]

print("   Selected classes:")
for idx, class_idx in enumerate(top_10_classes):
    class_name = all_classes[class_idx]
    count = counts[np.where(unique == class_idx)[0][0]]
    print(f"     {idx}: {class_name} ({count} samples)")

# Filter data for top 10 classes
mask = np.isin(labels, top_10_classes)
X_filtered = embeddings_pca[mask]
y_filtered = labels[mask]

# Remap labels to 0-9
label_mapping = {old: new for new, old in enumerate(top_10_classes)}
y_remapped = np.array([label_mapping[label] for label in y_filtered])

print(f"\n   ✓ Filtered to {len(X_filtered)} samples across 10 classes")

# Sample subset
print("\n3. Sampling data...")
max_samples_per_class = 50  # Reduced from 100

quantum_indices = []
for class_label in range(10):
    class_indices = np.where(y_remapped == class_label)[0]
    n_samples = min(len(class_indices), max_samples_per_class)
    sampled = np.random.choice(class_indices, n_samples, replace=False)
    quantum_indices.extend(sampled)

quantum_indices = np.array(quantum_indices)
np.random.shuffle(quantum_indices)

X_quantum = X_filtered[quantum_indices]
y_quantum = y_remapped[quantum_indices]

print(f"   ✓ Using {len(X_quantum)} samples ({max_samples_per_class} per class)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_quantum, y_quantum, test_size=0.2, random_state=42, stratify=y_quantum
)

print(f"\n4. Split: {len(X_train)} train, {len(X_test)} test")

# Initialize quantum classifier
print("\n5. Initializing 6-qubit quantum classifier...")
quantum_clf = QuantumClassifier(
    n_features=pca.n_components,
    n_classes=10,  # Only 10 classes
    n_qubits=6,    # Reduced from 8
    n_layers=3     # Reduced from 4
)
print(f"   ✓ Qubits: {quantum_clf.n_qubits}, Layers: {quantum_clf.n_layers}")
print(f"   ✓ Parameters: {quantum_clf.n_layers * quantum_clf.n_qubits * 2}")

# Train
print("\n6. Training quantum classifier...")
print("   This should take about 5-10 minutes...\n")

quantum_clf.fit(
    X_train, 
    y_train,
    epochs=15,      # Reduced from 30
    batch_size=20,  # Increased for faster training
    learning_rate=0.02,
    verbose=True
)

# Evaluate
print("\n7. Evaluating...")
train_acc = quantum_clf.score(X_train, y_train)
test_acc = quantum_clf.score(X_test, y_test)

print(f"   ✓ Train accuracy: {train_acc*100:.2f}%")
print(f"   ✓ Test accuracy: {test_acc*100:.2f}%")

# Save
print("\n8. Saving quantum classifier...")
save_data = {
    'classifier': quantum_clf,
    'class_mapping': {int(k): int(v) for k, v in label_mapping.items()},
    'class_names': [all_classes[int(idx)] for idx in top_10_classes],
    'n_classes_original': len(all_classes)
}

joblib.dump(save_data, artifacts_dir / 'classifiers' / 'quantum_clf_simple.joblib')
print(f"   ✓ Saved to: artifacts/classifiers/quantum_clf_simple.joblib")

print("\n" + "=" * 60)
print("✓ FAST Quantum Training Complete!")
print("=" * 60)
print()
print("This quantum classifier:")
print(f"  • Works on 10 disease classes (most common ones)")
print(f"  • Uses {quantum_clf.n_qubits} qubits and {quantum_clf.n_layers} layers")
print(f"  • Achieved {test_acc*100:.2f}% accuracy")
print()
print("Note: This is a simplified demo version.")
print("For full 45-class quantum classifier, run: python train_quantum_only.py")
print()

