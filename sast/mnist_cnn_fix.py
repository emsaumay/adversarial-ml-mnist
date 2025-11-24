import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import hashlib
import json
from pathlib import Path

# Configuration class to manage all file paths securely
class Config:
    def __init__(self):
        # Use environment variables with secure defaults
        base_dir = os.getenv('MNIST_BASE_DIR', os.getcwd())
        self.base_path = Path(base_dir).resolve()
        self.data_dir = self.base_path / 'data'
        self.output_dir = self.base_path / 'output'
        
        # Output file paths
        self.model_path = self.output_dir / 'mnist_cnn_model.pth'
        self.baseline_path = self.output_dir / 'baseline.txt'
        self.metadata_path = self.output_dir / 'model_metadata.json'
        self.checksums_path = self.output_dir / 'dataset_checksums.json'
        
        # Create directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create output directories with proper permissions"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
            self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        except OSError as e:
            print(f"Error creating directories: {e}")
            raise

# Initialize configuration
config = Config()

# Known good checksums for MNIST dataset files
# These should be verified against official MNIST distribution
MNIST_CHECKSUMS = {
    'train-images-idx3-ubyte': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
    'train-labels-idx1-ubyte': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
    't10k-images-idx3-ubyte': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
    't10k-labels-idx1-ubyte': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'
}

# Define the CNN architecture
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def compute_file_hash(filepath):
    """Compute SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        print(f"Error computing hash for {filepath}: {e}")
        return None

def validate_dataset_checksums(data_dir):
    """Validate MNIST dataset files against known checksums"""
    data_path = Path(data_dir) / 'MNIST' / 'raw'
    
    if not data_path.exists():
        print("Dataset not found, will be downloaded")
        return False
    
    print("Validating dataset integrity...")
    all_valid = True
    
    for filename, expected_checksum in MNIST_CHECKSUMS.items():
        file_path = data_path / filename
        
        if not file_path.exists():
            print(f"Missing file: {filename}")
            all_valid = False
            continue
        
        actual_checksum = compute_file_hash(file_path)
        if actual_checksum != expected_checksum:
            print(f"WARNING: Checksum mismatch for {filename}")
            print(f"Expected: {expected_checksum}")
            print(f"Got: {actual_checksum}")
            all_valid = False
        else:
            print(f"✓ {filename} verified")
    
    return all_valid

def safe_file_write(filepath, content):
    """Safely write content to file with atomic operation and error handling"""
    temp_path = filepath.with_suffix('.tmp')
    try:
        # Write to temporary file
        with open(temp_path, 'w') as f:
            f.write(content)
        
        # Atomic rename
        temp_path.replace(filepath)
        return True
    except IOError as e:
        print(f"Error writing to {filepath}: {e}")
        # Clean up temp file if it exists
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        return False
    except Exception as e:
        print(f"Unexpected error writing to {filepath}: {e}")
        return False

def save_model_securely(model, filepath, metadata):
    """Save model with integrity protection and metadata"""
    try:
        # Save model state dict
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
        
        # Compute checksum for integrity verification
        checksum = compute_file_hash(filepath)
        if checksum:
            metadata['model_checksum'] = checksum
            metadata['model_file'] = str(filepath.name)
            print(f"Model checksum: {checksum}")
        
        # Save metadata
        metadata_content = json.dumps(metadata, indent=2)
        if safe_file_write(config.metadata_path, metadata_content):
            print(f"Metadata saved to {config.metadata_path}")
            return True
        else:
            print("Warning: Failed to save metadata")
            return False
            
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model_securely(model, filepath):
    """Load model with integrity verification"""
    try:
        # Validate file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Check file size is reasonable
        file_size = filepath.stat().st_size
        max_size = 500 * 1024 * 1024  # 500MB limit
        if file_size < 1000 or file_size > max_size:
            raise ValueError(f"Model file size {file_size} bytes is suspicious")
        
        # Verify checksum if metadata exists
        if config.metadata_path.exists():
            try:
                with open(config.metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                expected_checksum = metadata.get('model_checksum')
                if expected_checksum:
                    actual_checksum = compute_file_hash(filepath)
                    if actual_checksum != expected_checksum:
                        raise ValueError("Model checksum mismatch - file may be corrupted or tampered")
                    print("✓ Model checksum verified")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Warning: Could not verify checksum: {e}")
        
        # Load with weights_only=True to prevent pickle exploits
        state_dict = torch.load(filepath, weights_only=True)
        
        # Validate state dict matches model architecture
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        
        if model_keys != loaded_keys:
            raise ValueError("Model architecture mismatch")
        
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {filepath}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Epoch {epoch} Training: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Testing function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test: Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

# Comprehensive evaluation function
def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    inference_times = []
    
    print('Evaluating model for baseline metrics...')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    # Build report content
    report_lines = []
    report_lines.append('BASELINE MODEL PERFORMANCE\n')
    report_lines.append('=' * 50 + '\n\n')
    
    report_lines.append('OVERALL METRICS\n')
    report_lines.append('-' * 50 + '\n')
    report_lines.append(f'Test Accuracy: {accuracy:.2f}%\n')
    report_lines.append(f'Test Loss: {test_loss:.4f}\n')
    report_lines.append(f'Correct Predictions: {correct}/{total}\n\n')
    
    report_lines.append('INFERENCE TIME\n')
    report_lines.append('-' * 50 + '\n')
    report_lines.append(f'Average Inference Time per Batch: {avg_inference_time*1000:.2f} ms\n')
    report_lines.append(f'Total Inference Time: {total_inference_time:.4f} seconds\n')
    report_lines.append(f'Number of Batches: {len(test_loader)}\n')
    report_lines.append(f'Inference Time per Sample: {(total_inference_time/total)*1000:.2f} ms\n\n')
    
    report_lines.append('CONFUSION MATRIX\n')
    report_lines.append('-' * 50 + '\n')
    report_lines.append('Rows: Actual, Columns: Predicted\n\n')
    report_lines.append('     ')
    for i in range(10):
        report_lines.append(f'{i:6d} ')
    report_lines.append('\n')
    for i, row in enumerate(conf_matrix):
        report_lines.append(f'{i:3d}  ')
        for val in row:
            report_lines.append(f'{val:6d} ')
        report_lines.append('\n')
    report_lines.append('\n')
    
    report_lines.append('PER-CLASS ACCURACY\n')
    report_lines.append('-' * 50 + '\n')
    for i in range(10):
        class_accuracy = (conf_matrix[i, i] / conf_matrix[i].sum()) * 100
        report_lines.append(f'Digit {i}: {class_accuracy:.2f}%\n')
    
    report_content = ''.join(report_lines)
    
    # Safely write report with error handling
    if safe_file_write(config.baseline_path, report_content):
        print(f'\nBaseline evaluation complete!')
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Loss: {test_loss:.4f}')
        print(f'Average Inference Time per Batch: {avg_inference_time*1000:.2f} ms')
        print(f'Results saved to {config.baseline_path}')
    else:
        print('Error: Failed to save baseline report')
    
    return accuracy, test_loss, conf_matrix

# Main execution
def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        print(f'Data directory: {config.data_dir}')
        print(f'Output directory: {config.output_dir}')
        print()
        
        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        print('Loading MNIST dataset...')
        train_dataset = datasets.MNIST(str(config.data_dir), train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(str(config.data_dir), train=False, download=True, transform=transform)
        
        # Validate dataset integrity with checksums
        print()
        is_valid = validate_dataset_checksums(config.data_dir)
        if not is_valid:
            print("\nWARNING: Dataset integrity validation failed!")
            print("This could indicate corrupted or tampered data.")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                print("Exiting for security reasons.")
                return
        print()
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        model = MNISTNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print('Starting training...')
        num_epochs = 5
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            test(model, device, test_loader, criterion)
            print()
        
        # Prepare metadata
        metadata = {
            'epochs': num_epochs,
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'batch_size': 64,
            'architecture': 'CNN (2 conv layers, 2 fc layers)',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'framework': 'PyTorch',
            'dataset': 'MNIST'
        }
        
        # Save model securely with integrity protection
        if save_model_securely(model, config.model_path, metadata):
            print('Model saved successfully with integrity protection')
        else:
            print('Warning: Model save encountered issues')
        
        print()
        
        # Evaluate model and save baseline metrics
        evaluate_model(model, device, test_loader, criterion)
        
        print('\nTraining complete!')
        print(f'All outputs saved to: {config.output_dir}')
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

if __name__ == '__main__':
    main()

