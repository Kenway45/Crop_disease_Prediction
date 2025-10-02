"""
Quantum Classifier for Crop Disease Prediction
Uses PennyLane for quantum circuit learning
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer, NesterovMomentumOptimizer
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class QuantumClassifier:
    """
    Quantum Neural Network Classifier using PennyLane.
    
    This classifier uses a variational quantum circuit to perform
    multi-class classification on PCA-reduced embeddings.
    """
    
    def __init__(self, n_features, n_classes, n_qubits=8, n_layers=4, device='default.qubit'):
        """
        Initialize quantum classifier.
        
        Args:
            n_features: Number of input features (from PCA)
            n_classes: Number of output classes
            n_qubits: Number of qubits to use (should be <= n_features)
            n_layers: Number of variational layers
            device: PennyLane device to use
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_qubits = min(n_qubits, n_features)
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = qml.device(device, wires=self.n_qubits)
        
        # Initialize weights
        self.weights = None
        self.bias = None
        
        # Feature scaling parameters
        self.feature_mean = None
        self.feature_std = None
        
        # Build quantum circuit
        self._build_circuit()
        
    def _build_circuit(self):
        """Build the variational quantum circuit."""
        
        @qml.qnode(self.dev, interface='autograd')
        def circuit(weights, features):
            """
            Quantum circuit with data encoding and variational layers.
            
            Args:
                weights: Variational parameters
                features: Input features (reduced to n_qubits dimensions)
            """
            # Data encoding layer - amplitude encoding
            # Normalize features to [0, pi]
            features_normalized = pnp.arctan(features) + pnp.pi / 2
            
            # Encode data using RY rotations
            for i in range(self.n_qubits):
                qml.RY(features_normalized[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Rotation layer
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Close the loop
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measurement - return expectation values of Pauli-Z
            return [qml.expval(qml.PauliZ(i)) for i in range(min(self.n_qubits, self.n_classes))]
        
        self.circuit = circuit
    
    def _feature_reduction(self, X):
        """
        Reduce features to fit qubit count if necessary.
        Uses simple projection or truncation.
        """
        if X.shape[1] > self.n_qubits:
            # Use first n_qubits principal components
            return X[:, :self.n_qubits]
        return X
    
    def _preprocess_features(self, X, fit=False):
        """Normalize features for quantum circuit."""
        X_reduced = self._feature_reduction(X)
        
        if fit:
            self.feature_mean = pnp.mean(X_reduced, axis=0)
            self.feature_std = pnp.std(X_reduced, axis=0) + 1e-8
        
        X_normalized = (X_reduced - self.feature_mean) / self.feature_std
        return X_normalized
    
    def _quantum_prediction(self, weights, features):
        """Get raw quantum circuit output."""
        output = self.circuit(weights, features)
        
        # Pad output to match n_classes if needed
        if len(output) < self.n_classes:
            output = pnp.concatenate([output, pnp.zeros(self.n_classes - len(output))])
        
        return output[:self.n_classes]
    
    def _cost_function(self, weights, X_batch, y_batch):
        """
        Cost function for training.
        Uses mean squared error for simplicity.
        """
        # Convert to PennyLane arrays
        X_batch = pnp.array(X_batch, requires_grad=False)
        y_batch = pnp.array(y_batch, requires_grad=False)
        
        predictions = []
        for features in X_batch:
            pred = self._quantum_prediction(weights, features)
            predictions.append(pred)
        
        predictions = pnp.stack(predictions)
        predictions = predictions + self.bias
        
        # Create one-hot encoded targets
        targets = pnp.zeros((len(y_batch), self.n_classes))
        for i, label in enumerate(y_batch):
            targets[i, int(label)] = 1.0
        
        # MSE loss (works better with quantum gradients)
        loss = pnp.sum((predictions - targets) ** 2) / len(y_batch)
        
        return loss
    
    def fit(self, X, y, epochs=50, batch_size=32, learning_rate=0.01, verbose=True):
        """
        Train the quantum classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            verbose: Whether to print training progress
        """
        # Preprocess features
        X = pnp.array(X, requires_grad=False)
        y = pnp.array(y, requires_grad=False)
        X_processed = self._preprocess_features(X, fit=True)
        
        # Initialize weights if not done
        if self.weights is None:
            # Random initialization
            np.random.seed(42)
            self.weights = pnp.random.uniform(
                low=-np.pi, 
                high=np.pi, 
                size=(self.n_layers, self.n_qubits, 2),
                requires_grad=True
            )
            self.bias = pnp.zeros(self.n_classes, requires_grad=True)
        
        # Optimizer
        opt = AdamOptimizer(stepsize=learning_rate)
        
        # Training loop
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        if verbose:
            print(f"\n🔮 Training Quantum Classifier")
            print(f"   Qubits: {self.n_qubits}, Layers: {self.n_layers}")
            print(f"   Classes: {self.n_classes}, Features: {self.n_features} → {self.n_qubits}")
            print(f"   Training samples: {n_samples}\n")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_processed[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}') if verbose else range(n_batches)
            
            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradients and update
                (self.weights, self.bias), cost = opt.step_and_cost(
                    lambda w, b: self._cost_function(w, X_batch, y_batch),
                    self.weights,
                    self.bias
                )
                
                epoch_loss += cost
                
                if verbose and isinstance(pbar, tqdm):
                    try:
                        loss_val = float(cost)
                        pbar.set_postfix({'loss': f'{loss_val:.4f}'})
                    except:
                        pbar.set_postfix({'loss': str(cost)[:8]})
            
            avg_loss = epoch_loss / n_batches
            
            if verbose and (epoch + 1) % 5 == 0:
                # Compute accuracy on a subset
                sample_indices = np.random.choice(n_samples, min(1000, n_samples), replace=False)
                acc = self.score(X[sample_indices], y[sample_indices])
                try:
                    loss_val = float(avg_loss)
                    acc_val = float(acc)
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_val:.4f}, Acc: {acc_val*100:.2f}%")
                except:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss}, Acc: {acc}")
        
        if verbose:
            print("\n✓ Quantum classifier training complete!\n")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        X_processed = self._preprocess_features(X, fit=False)
        
        predictions = []
        for features in X_processed:
            pred = self._quantum_prediction(self.weights, features)
            predictions.append(pred)
        
        predictions = pnp.array(predictions)
        predictions = predictions + self.bias
        
        # Softmax
        exp_pred = pnp.exp(predictions - pnp.max(predictions, axis=1, keepdims=True))
        probs = exp_pred / pnp.sum(exp_pred, axis=1, keepdims=True)
        
        return pnp.array(probs)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        probs = self.predict_proba(X)
        return pnp.argmax(probs, axis=1)
    
    def score(self, X, y):
        """
        Compute accuracy score.
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return pnp.mean(predictions == y)
    
    def save(self, filepath):
        """Save quantum classifier to file."""
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load quantum classifier from file."""
        model_data = joblib.load(filepath)
        
        classifier = cls(
            n_features=model_data['n_features'],
            n_classes=model_data['n_classes'],
            n_qubits=model_data['n_qubits'],
            n_layers=model_data['n_layers']
        )
        
        classifier.weights = model_data['weights']
        classifier.bias = model_data['bias']
        classifier.feature_mean = model_data['feature_mean']
        classifier.feature_std = model_data['feature_std']
        
        return classifier


def train_quantum_ensemble(X, y, n_classifiers=3, **kwargs):
    """
    Train an ensemble of quantum classifiers for improved accuracy.
    
    Args:
        X: Training features
        y: Training labels
        n_classifiers: Number of quantum classifiers in ensemble
        **kwargs: Arguments passed to QuantumClassifier.fit()
    
    Returns:
        List of trained quantum classifiers
    """
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    classifiers = []
    
    for i in range(n_classifiers):
        print(f"\n{'='*60}")
        print(f"Training Quantum Classifier {i+1}/{n_classifiers}")
        print(f"{'='*60}")
        
        # Initialize with different random seed
        np.random.seed(42 + i)
        
        clf = QuantumClassifier(
            n_features=n_features,
            n_classes=n_classes,
            n_qubits=min(8, n_features),
            n_layers=4
        )
        
        clf.fit(X, y, **kwargs)
        classifiers.append(clf)
    
    return classifiers


def ensemble_predict_proba(classifiers, X):
    """
    Predict using ensemble averaging.
    
    Args:
        classifiers: List of trained quantum classifiers
        X: Input features
    
    Returns:
        Averaged probability predictions
    """
    all_probs = []
    
    for clf in classifiers:
        probs = clf.predict_proba(X)
        all_probs.append(probs)
    
    # Average predictions
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


def ensemble_predict(classifiers, X):
    """
    Predict class labels using ensemble.
    
    Args:
        classifiers: List of trained quantum classifiers
        X: Input features
    
    Returns:
        Predicted labels
    """
    probs = ensemble_predict_proba(classifiers, X)
    return np.argmax(probs, axis=1)

