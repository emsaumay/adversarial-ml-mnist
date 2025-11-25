import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import time

# Same CNN architecture as baseline
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

def fgsm_attack(model, images, labels, epsilon, device):
    # Generate adversarial examples using FGSM
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)
    
    return perturbed_images.detach()

def pgd_attack(model, images, labels, epsilon, alpha, num_iter, device):
    # Generate adversarial examples using PGD
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    perturbed_images = images.clone().detach()
    perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)
    
    for i in range(num_iter):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * perturbed_images.grad.sign()
            perturbation = torch.clamp(perturbed_images - images, -epsilon, epsilon)
            perturbed_images = images + perturbation
            perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)
    
    return perturbed_images.detach()

def train_with_adversarial(model, device, train_loader, optimizer, criterion, epoch, epsilon=0.3):
    # Train on both clean and adversarial examples
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Train on clean data
        optimizer.zero_grad()
        output = model(data)
        loss_clean = criterion(output, target)
        
        # Generate adversarial examples
        adv_data = fgsm_attack(model, data, target, epsilon, device)
        
        # Train on adversarial data
        output_adv = model(adv_data)
        loss_adv = criterion(output_adv, target)
        
        # Combined loss
        loss = loss_clean + loss_adv
        loss.backward()
        optimizer.step()
        
        # Track accuracy on clean data
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Epoch {epoch} Training: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

def test_clean(model, device, test_loader, criterion):
    # Test on clean data
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
    return accuracy, test_loss, correct, total

def test_adversarial_attack(model, device, test_loader, attack_type='fgsm', epsilon=0.3):
    # Test against adversarial attacks
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        if attack_type == 'fgsm':
            adv_data = fgsm_attack(model, data, target, epsilon, device)
        else:  # pgd
            adv_data = pgd_attack(model, data, target, epsilon, 0.01, 40, device)
        
        # Test on adversarial examples
        with torch.no_grad():
            output = model(adv_data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    return accuracy, correct, total, conf_matrix

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print('Loading MNIST dataset...')
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Initialize model
    print('\nInitializing model for adversarial training...')
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print('Starting adversarial training (training on clean + FGSM examples)...')
    print('This will take longer than normal training...\n')
    
    num_epochs = 5
    epsilon = 0.3
    
    # Train with adversarial examples
    for epoch in range(1, num_epochs + 1):
        train_with_adversarial(model, device, train_loader, optimizer, criterion, epoch, epsilon)
        clean_acc, clean_loss, _, _ = test_clean(model, device, test_loader, criterion)
        print(f'Test on Clean Data: Loss: {clean_loss:.4f}, Accuracy: {clean_acc:.2f}%')
        print()
    
    # Save the adversarially trained model
    torch.save(model.state_dict(), 'protections/adversarial_trained_model.pth')
    print('Adversarially trained model saved\n')
    
    # Evaluate on clean data
    print('Evaluating adversarially trained model...\n')
    print('Testing on clean data...')
    clean_acc, clean_loss, clean_correct, clean_total = test_clean(model, device, test_loader, criterion)
    print(f'Clean Test Accuracy: {clean_acc:.2f}% ({clean_correct}/{clean_total})\n')
    
    # Test against FGSM attack
    print('Testing against FGSM attack...')
    fgsm_acc, fgsm_correct, fgsm_total, fgsm_conf_matrix = test_adversarial_attack(
        model, device, test_loader, 'fgsm', epsilon
    )
    print(f'FGSM Attack Accuracy: {fgsm_acc:.2f}% ({fgsm_correct}/{fgsm_total})')
    print(f'FGSM Attack Success Rate: {100 - fgsm_acc:.2f}%\n')
    
    # Test against PGD attack
    print('Testing against PGD attack...')
    pgd_acc, pgd_correct, pgd_total, pgd_conf_matrix = test_adversarial_attack(
        model, device, test_loader, 'pgd', epsilon
    )
    print(f'PGD Attack Accuracy: {pgd_acc:.2f}% ({pgd_correct}/{pgd_total})')
    print(f'PGD Attack Success Rate: {100 - pgd_acc:.2f}%\n')
    
    # Load baseline model for comparison
    print('Loading baseline model for comparison...')
    baseline_model = MNISTNet().to(device)
    baseline_model.load_state_dict(torch.load('mnist_cnn_model.pth'))
    baseline_model.eval()
    
    # Test baseline against attacks
    baseline_clean_acc, _, _, _ = test_clean(baseline_model, device, test_loader, criterion)
    baseline_fgsm_acc, _, _, _ = test_adversarial_attack(baseline_model, device, test_loader, 'fgsm', epsilon)
    baseline_pgd_acc, _, _, _ = test_adversarial_attack(baseline_model, device, test_loader, 'pgd', epsilon)
    
    # Save comprehensive results
    with open('protections/defense_results.txt', 'w') as f:
        f.write('ADVERSARIAL TRAINING DEFENSE (BLUE TEAM)\n')
        f.write('=' * 70 + '\n\n')
        
        f.write('DEFENSE STRATEGY\n')
        f.write('-' * 70 + '\n')
        f.write('Adversarial Training: Train the model on both clean and adversarial\n')
        f.write('examples to make it robust against gradient-based attacks.\n\n')
        f.write('Training Configuration:\n')
        f.write('  - Epochs: 5\n')
        f.write('  - Adversarial Method: FGSM\n')
        f.write(f'  - Epsilon: {epsilon}\n')
        f.write('  - Each batch: 50% clean data + 50% adversarial data\n\n')
        
        f.write('PERFORMANCE COMPARISON\n')
        f.write('-' * 70 + '\n')
        f.write(f'{"Model":<25} {"Clean":<15} {"FGSM":<15} {"PGD":<15}\n')
        f.write('-' * 70 + '\n')
        f.write(f'{"Baseline (No Defense)":<25} {baseline_clean_acc:<15.2f} {baseline_fgsm_acc:<15.2f} {baseline_pgd_acc:<15.2f}\n')
        f.write(f'{"Adversarial Training":<25} {clean_acc:<15.2f} {fgsm_acc:<15.2f} {pgd_acc:<15.2f}\n\n')
        
        f.write('IMPROVEMENT METRICS\n')
        f.write('-' * 70 + '\n')
        clean_diff = clean_acc - baseline_clean_acc
        fgsm_diff = fgsm_acc - baseline_fgsm_acc
        pgd_diff = pgd_acc - baseline_pgd_acc
        
        f.write(f'Clean Data: {clean_diff:+.2f}% {"(minor drop expected)" if clean_diff < 0 else "(maintained)"}\n')
        f.write(f'FGSM Robustness: {fgsm_diff:+.2f}% {"(improved)" if fgsm_diff > 0 else "(needs work)"}\n')
        f.write(f'PGD Robustness: {pgd_diff:+.2f}% {"(improved)" if pgd_diff > 0 else "(needs work)"}\n\n')
        
        f.write('DETAILED RESULTS - ADVERSARIAL TRAINED MODEL\n')
        f.write('-' * 70 + '\n\n')
        
        f.write('Clean Test Data:\n')
        f.write(f'  Accuracy: {clean_acc:.2f}%\n')
        f.write(f'  Loss: {clean_loss:.4f}\n')
        f.write(f'  Correct: {clean_correct}/{clean_total}\n\n')
        
        f.write('FGSM Attack (epsilon=0.3):\n')
        f.write(f'  Accuracy: {fgsm_acc:.2f}%\n')
        f.write(f'  Correct: {fgsm_correct}/{fgsm_total}\n')
        f.write(f'  Attack Success Rate: {100 - fgsm_acc:.2f}%\n\n')
        
        f.write('PGD Attack (epsilon=0.3, 40 iterations):\n')
        f.write(f'  Accuracy: {pgd_acc:.2f}%\n')
        f.write(f'  Correct: {pgd_correct}/{pgd_total}\n')
        f.write(f'  Attack Success Rate: {100 - pgd_acc:.2f}%\n\n')
        
        f.write('CONFUSION MATRIX - FGSM ATTACK\n')
        f.write('-' * 70 + '\n')
        f.write('     ')
        for i in range(10):
            f.write(f'{i:6d} ')
        f.write('\n')
        for i, row in enumerate(fgsm_conf_matrix):
            f.write(f'{i:3d}  ')
            for val in row:
                f.write(f'{val:6d} ')
            f.write('\n')
        f.write('\n')
        
        f.write('CONFUSION MATRIX - PGD ATTACK\n')
        f.write('-' * 70 + '\n')
        f.write('     ')
        for i in range(10):
            f.write(f'{i:6d} ')
        f.write('\n')
        for i, row in enumerate(pgd_conf_matrix):
            f.write(f'{i:3d}  ')
            for val in row:
                f.write(f'{val:6d} ')
            f.write('\n')
        f.write('\n')
        
        f.write('ATTACK SUCCESS RATE COMPARISON\n')
        f.write('-' * 70 + '\n')
        f.write(f'{"Attack Type":<20} {"Baseline":<20} {"Adversarial Training":<20}\n')
        f.write('-' * 70 + '\n')
        f.write(f'{"FGSM":<20} {100-baseline_fgsm_acc:<20.2f} {100-fgsm_acc:<20.2f}\n')
        f.write(f'{"PGD":<20} {100-baseline_pgd_acc:<20.2f} {100-pgd_acc:<20.2f}\n\n')
        
        f.write('INTERPRETATION\n')
        f.write('-' * 70 + '\n')
        f.write('Adversarial training is a defense technique where the model is trained\n')
        f.write('on adversarial examples during the training process. This helps the model\n')
        f.write('learn robust features that are less sensitive to adversarial perturbations.\n\n')
        
        if fgsm_diff > 5:
            f.write(f'The model showed significant improvement against FGSM attacks (+{fgsm_diff:.2f}%).\n')
        elif fgsm_diff > 0:
            f.write(f'The model showed some improvement against FGSM attacks (+{fgsm_diff:.2f}%).\n')
        else:
            f.write(f'FGSM robustness needs further improvement ({fgsm_diff:.2f}%).\n')
        
        if pgd_diff > 5:
            f.write(f'The model showed significant improvement against PGD attacks (+{pgd_diff:.2f}%).\n')
        elif pgd_diff > 0:
            f.write(f'The model showed some improvement against PGD attacks (+{pgd_diff:.2f}%).\n')
        else:
            f.write(f'PGD robustness needs further improvement ({pgd_diff:.2f}%).\n')
        
        f.write('\nTradeoff: Adversarial training may slightly reduce clean accuracy\n')
        f.write('but significantly improves robustness against adversarial attacks.\n')
    
    print('Results saved to protections/defense_results.txt')
    print('\nSummary:')
    print(f'  Baseline Clean: {baseline_clean_acc:.2f}% | FGSM: {baseline_fgsm_acc:.2f}% | PGD: {baseline_pgd_acc:.2f}%')
    print(f'  Defended Clean: {clean_acc:.2f}% | FGSM: {fgsm_acc:.2f}% | PGD: {pgd_acc:.2f}%')
    print(f'  Improvement: Clean: {clean_diff:+.2f}% | FGSM: {fgsm_diff:+.2f}% | PGD: {pgd_diff:+.2f}%')

if __name__ == '__main__':
    main()

