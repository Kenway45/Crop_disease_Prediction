"""
Training pipeline for crop disease prediction.
Trains CNN, extracts embeddings, performs PCA, and trains classifier.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import quantum classifier
try:
    from quantum_classifier import QuantumClassifier
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("⚠️  Quantum classifier not available. Install pennylane to enable quantum features.")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CropDiseaseDataset(Dataset):
    """Dataset class for crop disease images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class CropDiseaseCNN(nn.Module):
    """CNN model for crop disease classification."""
    
    def __init__(self, num_classes):
        super(CropDiseaseCNN, self).__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Get the input features for the final layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final layer
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
        # Forward through all layers except the last Linear layer
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
        
        # Go through dropout and first linear layer (embedding layer)
        if isinstance(self.backbone.fc, nn.Sequential):
            x = self.backbone.fc[0](x)  # Dropout
            x = self.backbone.fc[1](x)  # Linear -> embeddings
            x = self.backbone.fc[2](x)  # ReLU
        
        return x

def collect_dataset(data_dir):
    """Collect all images and labels from datasets."""
    data_path = Path(data_dir)
    image_paths = []
    labels = []
    
    print("Collecting images from datasets...")
    
    # Common image extensions
    image_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # Find all image files
    for img_path in data_path.rglob('*'):
        if img_path.suffix in image_exts:
            # Get label from parent directory name
            label = img_path.parent.name
            # Skip if it's just a dataset name or generic folder
            if label.lower() not in ['data', 'train', 'test', 'val', 'images']:
                image_paths.append(str(img_path))
                labels.append(label)
    
    print(f"Found {len(image_paths)} images across {len(set(labels))} classes")
    
    # Print class distribution
    from collections import Counter
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} images")
    
    return image_paths, labels

def train_cnn(model, train_loader, val_loader, num_epochs, device, save_path):
    """Train the CNN model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{train_loss/len(train_loader):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/len(val_loader):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}: Train Acc: {100.*train_correct/train_total:.2f}%, Val Acc: {val_acc:.2f}%\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"✓ Saved best model with validation accuracy: {val_acc:.2f}%\n")
    
    return best_val_acc

def extract_embeddings(model, dataloader, device):
    """Extract embeddings from trained model."""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc='Extracting embeddings'):
            images = images.to(device)
            emb = model.get_embeddings(images)
            embeddings.append(emb.cpu().numpy())
            labels.append(lbls.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    return embeddings, labels

def main():
    """Main training pipeline."""
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    artifacts_dir = project_root / 'artifacts'
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10  # Reduced from 20 for faster training
    IMG_SIZE = 224
    PCA_COMPONENTS = 128
    
    # Device - Use MPS (Apple GPU) if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect dataset
    print("="*60)
    print("STEP 1: Collecting dataset")
    print("="*60)
    image_paths, labels = collect_dataset(data_dir)
    
    if len(image_paths) == 0:
        print("No images found! Please run download_data.py first.")
        return
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"\nTotal classes: {num_classes}")
    
    # Save class names
    class_names = label_encoder.classes_.tolist()
    with open(artifacts_dir / 'models' / 'classes.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ Saved class names to artifacts/models/classes.json")
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"\nTrain samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # Create datasets and dataloaders
    train_dataset = CropDiseaseDataset(X_train, y_train, train_transform)
    val_dataset = CropDiseaseDataset(X_val, y_val, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Train CNN
    print("\n" + "="*60)
    print("STEP 2: Training CNN")
    print("="*60)
    
    model = CropDiseaseCNN(num_classes).to(device)
    model_save_path = artifacts_dir / 'models' / 'best_cnn.pt'
    
    best_acc = train_cnn(model, train_loader, val_loader, NUM_EPOCHS, device, model_save_path)
    print(f"\n✓ Training complete! Best validation accuracy: {best_acc:.2f}%")
    
    # Load best model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract embeddings
    print("\n" + "="*60)
    print("STEP 3: Extracting embeddings")
    print("="*60)
    
    # Create full dataset for embeddings
    full_dataset = CropDiseaseDataset(image_paths, encoded_labels, val_transform)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    embeddings, emb_labels = extract_embeddings(model, full_loader, device)
    
    # Save embeddings
    np.save(artifacts_dir / 'embeddings' / 'train_emb.npy', embeddings)
    np.save(artifacts_dir / 'embeddings' / 'train_labels.npy', emb_labels)
    print(f"✓ Saved embeddings: shape {embeddings.shape}")
    
    # PCA
    print("\n" + "="*60)
    print("STEP 4: Training PCA")
    print("="*60)
    
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    joblib.dump(pca, artifacts_dir / 'pca' / 'pca.joblib')
    print(f"✓ PCA trained and saved: {PCA_COMPONENTS} components")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Train classifier
    print("\n" + "="*60)
    print("STEP 5: Training Logistic Regression classifier")
    print("="*60)
    
    X_train_pca, X_test_pca, y_train_clf, y_test_clf = train_test_split(
        embeddings_pca, emb_labels, test_size=0.2, random_state=42, stratify=emb_labels
    )
    
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', n_jobs=-1)
    clf.fit(X_train_pca, y_train_clf)
    
    train_acc = clf.score(X_train_pca, y_train_clf)
    test_acc = clf.score(X_test_pca, y_test_clf)
    
    print(f"✓ Classifier trained")
    print(f"  Train accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    
    joblib.dump(clf, artifacts_dir / 'classifiers' / 'lr_clf.joblib')
    print(f"✓ Classifier saved to artifacts/classifiers/lr_clf.joblib")
    
    # Train Quantum Classifier
    if QUANTUM_AVAILABLE:
        print("\n" + "="*60)
        print("STEP 6: Training Quantum Classifier 🔮")
        print("="*60)
        
        try:
            # Use a subset for quantum training (quantum is slower)
            # Sample balanced subset
            max_samples_per_class = 100  # Adjust based on your compute power
            quantum_indices = []
            
            for class_label in range(num_classes):
                class_indices = np.where(emb_labels == class_label)[0]
                n_samples = min(len(class_indices), max_samples_per_class)
                sampled = np.random.choice(class_indices, n_samples, replace=False)
                quantum_indices.extend(sampled)
            
            quantum_indices = np.array(quantum_indices)
            np.random.shuffle(quantum_indices)
            
            X_quantum = embeddings_pca[quantum_indices]
            y_quantum = emb_labels[quantum_indices]
            
            print(f"\nUsing {len(X_quantum)} samples for quantum training")
            print(f"(Sampled {max_samples_per_class} per class for efficiency)\n")
            
            # Split quantum data
            X_train_quantum, X_test_quantum, y_train_quantum, y_test_quantum = train_test_split(
                X_quantum, y_quantum, test_size=0.2, random_state=42, stratify=y_quantum
            )
            
            # Initialize quantum classifier
            quantum_clf = QuantumClassifier(
                n_features=PCA_COMPONENTS,
                n_classes=num_classes,
                n_qubits=8,
                n_layers=4
            )
            
            # Train quantum classifier
            quantum_clf.fit(
                X_train_quantum, 
                y_train_quantum,
                epochs=30,
                batch_size=16,
                learning_rate=0.01,
                verbose=True
            )
            
            # Evaluate quantum classifier
            quantum_train_acc = quantum_clf.score(X_train_quantum, y_train_quantum)
            quantum_test_acc = quantum_clf.score(X_test_quantum, y_test_quantum)
            
            print(f"\n✓ Quantum Classifier trained")
            print(f"  Train accuracy: {quantum_train_acc*100:.2f}%")
            print(f"  Test accuracy: {quantum_test_acc*100:.2f}%")
            
            # Save quantum classifier
            quantum_clf.save(artifacts_dir / 'classifiers' / 'quantum_clf.joblib')
            print(f"✓ Quantum classifier saved to artifacts/classifiers/quantum_clf.joblib")
            
        except Exception as e:
            print(f"\n⚠️  Quantum training failed: {e}")
            print("   Continuing without quantum classifier...")
    else:
        print("\n" + "="*60)
        print("STEP 6: Quantum Classifier - SKIPPED")
        print("="*60)
        print("Install pennylane to enable quantum features:")
        print("  pip install pennylane pennylane-lightning")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nArtifacts created:")
    print(f"  ✓ artifacts/models/best_cnn.pt")
    print(f"  ✓ artifacts/models/classes.json")
    print(f"  ✓ artifacts/embeddings/train_emb.npy")
    print(f"  ✓ artifacts/embeddings/train_labels.npy")
    print(f"  ✓ artifacts/pca/pca.joblib")
    print(f"  ✓ artifacts/classifiers/lr_clf.joblib")
    print("\n✓ All files are ready for demo_server.py!")

if __name__ == '__main__':
    main()

