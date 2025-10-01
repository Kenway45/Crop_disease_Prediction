"""
Download datasets from Kaggle for crop disease prediction.
Downloads PlantVillage, Rice Leaf Diseases, and Cotton Leaf Disease datasets.
"""
import os
import subprocess
import zipfile
from pathlib import Path

# Dataset information
DATASETS = [
    {
        'name': 'plantvillage',
        'kaggle_id': 'mohitsingh1804/plantvillage',
        'zip_name': 'plantvillage.zip'
    },
    {
        'name': 'rice-leaf-diseases',
        'kaggle_id': 'vbookshelf/rice-leaf-diseases',
        'zip_name': 'rice-leaf-diseases.zip'
    },
    {
        'name': 'cotton-leaf-disease',
        'kaggle_id': 'seroshkarim/cotton-leaf-disease-dataset',
        'zip_name': 'cotton-leaf-disease-dataset.zip'
    }
]

def download_and_extract_datasets(data_dir='./data'):
    """
    Download all datasets from Kaggle and extract them.
    
    Args:
        data_dir: Directory to store downloaded datasets
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Downloading {dataset['name']}...")
        print(f"{'='*60}")
        
        dataset_path = data_path / dataset['name']
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Download using Kaggle API
        try:
            cmd = [
                'kaggle', 'datasets', 'download',
                '-d', dataset['kaggle_id'],
                '-p', str(dataset_path),
                '--unzip'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Successfully downloaded and extracted {dataset['name']}")
            print(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading {dataset['name']}: {e.stderr}")
            continue
        except FileNotFoundError:
            print("✗ Kaggle CLI not found. Please install it with: pip install kaggle")
            return False
    
    print(f"\n{'='*60}")
    print("All datasets downloaded successfully!")
    print(f"Data directory: {data_path.absolute()}")
    print(f"{'='*60}\n")
    
    # Print dataset structure
    print("\nDataset structure:")
    for dataset in DATASETS:
        dataset_path = data_path / dataset['name']
        if dataset_path.exists():
            print(f"\n{dataset['name']}:")
            for item in list(dataset_path.rglob('*'))[:10]:
                if item.is_file():
                    print(f"  {item.relative_to(dataset_path)}")
            
    return True

if __name__ == '__main__':
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    print("Starting dataset download...")
    print(f"Kaggle credentials should be at: ~/.kaggle/kaggle.json\n")
    
    success = download_and_extract_datasets(str(data_dir))
    
    if success:
        print("\n✓ Setup complete! You can now run train_model.py")
    else:
        print("\n✗ Download failed. Please check your Kaggle API setup.")

