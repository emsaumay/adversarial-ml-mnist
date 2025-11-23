import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Define the CNN architecture
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        # FC1 -> ReLU -> Dropout
        x = self.dropout(torch.relu(self.fc1(x)))
        # FC2 (output layer)
        x = self.fc2(x)
        return x

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Calculate loss
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        
        # Track accuracy
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Print progress every 100 batches
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
            
            # Measure inference time
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate loss
            test_loss += criterion(output, target).item()
            
            # Get predictions
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    # Save results to baseline.txt
    with open('baseline.txt', 'w') as f:
        f.write('BASELINE MODEL PERFORMANCE\n')
        f.write('=' * 50 + '\n\n')
        
        f.write('OVERALL METRICS\n')
        f.write('-' * 50 + '\n')
        f.write(f'Test Accuracy: {accuracy:.2f}%\n')
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Correct Predictions: {correct}/{total}\n\n')
        
        f.write('INFERENCE TIME\n')
        f.write('-' * 50 + '\n')
        f.write(f'Average Inference Time per Batch: {avg_inference_time*1000:.2f} ms\n')
        f.write(f'Total Inference Time: {total_inference_time:.4f} seconds\n')
        f.write(f'Number of Batches: {len(test_loader)}\n')
        f.write(f'Inference Time per Sample: {(total_inference_time/total)*1000:.2f} ms\n\n')
        
        f.write('CONFUSION MATRIX\n')
        f.write('-' * 50 + '\n')
        f.write('Rows: Actual, Columns: Predicted\n\n')
        f.write('     ')
        for i in range(10):
            f.write(f'{i:6d} ')
        f.write('\n')
        for i, row in enumerate(conf_matrix):
            f.write(f'{i:3d}  ')
            for val in row:
                f.write(f'{val:6d} ')
            f.write('\n')
        f.write('\n')
        
        f.write('PER-CLASS ACCURACY\n')
        f.write('-' * 50 + '\n')
        for i in range(10):
            class_accuracy = (conf_matrix[i, i] / conf_matrix[i].sum()) * 100
            f.write(f'Digit {i}: {class_accuracy:.2f}%\n')
    
    print(f'\nBaseline evaluation complete!')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Loss: {test_loss:.4f}')
    print(f'Average Inference Time per Batch: {avg_inference_time*1000:.2f} ms')
    print(f'Results saved to baseline.txt')
    
    return accuracy, test_loss, conf_matrix

# Main execution
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset
    print('Loading MNIST dataset...')
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
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
    
    # Save the trained model
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')
    print('Model saved as mnist_cnn_model.pth')
    print()
    
    # Evaluate model and save baseline metrics
    evaluate_model(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()

