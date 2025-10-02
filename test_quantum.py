#!/usr/bin/env python3
"""
Test script for quantum classifier
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from quantum_classifier import QuantumClassifier

print("=" * 60)
print("Testing Quantum Classifier")
print("=" * 60)

# Generate dummy data
print("\n1. Generating test data...")
np.random.seed(42)
n_samples = 100
n_features = 8
n_classes = 3

X_train = np.random.randn(n_samples, n_features)
y_train = np.random.randint(0, n_classes, n_samples)

X_test = np.random.randn(20, n_features)
y_test = np.random.randint(0, n_classes, 20)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   Classes: {n_classes}")

# Initialize quantum classifier
print("\n2. Initializing quantum classifier...")
quantum_clf = QuantumClassifier(
    n_features=n_features,
    n_classes=n_classes,
    n_qubits=6,
    n_layers=2
)
print("   ✓ Quantum classifier initialized")
print(f"   Qubits: {quantum_clf.n_qubits}, Layers: {quantum_clf.n_layers}")

# Train
print("\n3. Training quantum classifier...")
quantum_clf.fit(
    X_train, 
    y_train,
    epochs=5,
    batch_size=16,
    learning_rate=0.01,
    verbose=True
)

# Test
print("\n4. Testing predictions...")
predictions = quantum_clf.predict(X_test)
probabilities = quantum_clf.predict_proba(X_test)
accuracy = quantum_clf.score(X_test, y_test)

print(f"\n   Predictions shape: {predictions.shape}")
print(f"   Probabilities shape: {probabilities.shape}")
print(f"   Test accuracy: {accuracy*100:.2f}%")

print("\n5. Sample predictions:")
for i in range(min(5, len(X_test))):
    print(f"   Sample {i}: Predicted={predictions[i]}, True={y_test[i]}, "
          f"Probs={probabilities[i]}")

# Save and load test
print("\n6. Testing save/load...")
quantum_clf.save('test_quantum_model.joblib')
print("   ✓ Model saved")

loaded_clf = QuantumClassifier.load('test_quantum_model.joblib')
print("   ✓ Model loaded")

loaded_predictions = loaded_clf.predict(X_test)
assert np.array_equal(predictions, loaded_predictions), "Predictions don't match!"
print("   ✓ Loaded model produces same predictions")

# Cleanup
import os
os.remove('test_quantum_model.joblib')
print("   ✓ Cleanup complete")

print("\n" + "=" * 60)
print("✓ All tests passed! Quantum classifier is working!")
print("=" * 60)

