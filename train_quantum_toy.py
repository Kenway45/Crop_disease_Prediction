#!/usr/bin/env python3
"""
TOY Quantum Classifier - ULTRA FAST
For immediate testing and demonstration
Training time: ~2-3 minutes
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from pathlib import Path
from quantum_classifier import QuantumClassifier
from sklearn.model_selection import train_test_split
import joblib
import json

print("=" * 60)
print("🎮 TOY Quantum Classifier - ULTRA FAST")
print("=" * 60)
print()
print("This TOY version:")
print("  • Uses only 3 disease classes")
print("  • 30 samples per class (90 total)")
print("  • 4 qubits, 2 layers")
print("  • 10 epochs")
print("  • Training time: ~2-3 minutes")
print()
print("Perfect for quick testing and learning!")
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

embeddings_pca = pca.transform(embeddings)
print(f"   ✓ Loaded {len(embeddings)} samples")

# Select 3 most common classes
print("\n2. Selecting 3 most common disease classes...")
unique, counts = np.unique(labels, return_counts=True)
top_3_indices = np.argsort(counts)[-3:]
top_3_classes = unique[top_3_indices]

for idx, class_idx in enumerate(top_3_classes):
    class_name = all_classes[class_idx]
    count = counts[np.where(unique == class_idx)[0][0]]
    print(f"   Class {idx}: {class_name} ({count} samples)")

# Filter and sample
mask = np.isin(labels, top_3_classes)
X_filtered = embeddings_pca[mask]
y_filtered = labels[mask]

label_mapping = {old: new for new, old in enumerate(top_3_classes)}
y_remapped = np.array([label_mapping[label] for label in y_filtered])

# Sample 30 per class
quantum_indices = []
for class_label in range(3):
    class_indices = np.where(y_remapped == class_label)[0]
    sampled = np.random.choice(class_indices, 30, replace=False)
    quantum_indices.extend(sampled)

quantum_indices = np.array(quantum_indices)
np.random.shuffle(quantum_indices)

X_quantum = X_filtered[quantum_indices]
y_quantum = y_remapped[quantum_indices]

print(f"\n3. Using {len(X_quantum)} samples (30 per class)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_quantum, y_quantum, test_size=0.2, random_state=42
)
print(f"   Split: {len(X_train)} train, {len(X_test)} test")

# Initialize SMALL quantum classifier
print("\n4. Initializing 4-qubit quantum classifier...")
quantum_clf = QuantumClassifier(
    n_features=pca.n_components,
    n_classes=3,
    n_qubits=4,  # Very small
    n_layers=2   # Very shallow
)
print(f"   Qubits: {quantum_clf.n_qubits}, Layers: {quantum_clf.n_layers}")
print(f"   Parameters: {quantum_clf.n_layers * quantum_clf.n_qubits * 2}")

# Train
print("\n5. Training quantum classifier...")
print("   This should take about 2-3 minutes...\n")

quantum_clf.fit(
    X_train, 
    y_train,
    epochs=10,
    batch_size=18,
    learning_rate=0.05,
    verbose=True
)

# Evaluate
print("\n6. Evaluating...")
train_acc = quantum_clf.score(X_train, y_train)
test_acc = quantum_clf.score(X_test, y_test)

print(f"   Train accuracy: {train_acc*100:.2f}%")
print(f"   Test accuracy: {test_acc*100:.2f}%")

# Test prediction
print("\n7. Testing prediction on one sample...")
sample = X_test[0:1]
pred = quantum_clf.predict(sample)
probs = quantum_clf.predict_proba(sample)

print(f"   Predicted class: {pred[0]} ({all_classes[top_3_classes[pred[0]]]})")
print(f"   Probabilities: {probs[0]}")

# Save
print("\n8. Saving toy quantum classifier...")
save_data = {
    'classifier': quantum_clf,
    'class_mapping': {int(k): int(v) for k, v in label_mapping.items()},
    'class_names': [all_classes[int(idx)] for idx in top_3_classes]
}

joblib.dump(save_data, artifacts_dir / 'classifiers' / 'quantum_clf_toy.joblib')
print(f"   ✓ Saved to: artifacts/classifiers/quantum_clf_toy.joblib")

print("\n" + "=" * 60)
print("✓ TOY Quantum Training Complete!")
print("=" * 60)
print()
print("🎉 You just trained a quantum classifier!")
print()
print("What you learned:")
print("  • Quantum circuits can classify data")
print("  • Uses qubits and quantum gates")
print("  • Trained with gradient descent")
print(f"  • Achieved {test_acc*100:.1f}% accuracy on 3 classes")
print()
print("Next steps:")
print("  • Try: python train_quantum_simple.py (10 classes)")
print("  • Try: python train_quantum_only.py (all 45 classes)")
print()

