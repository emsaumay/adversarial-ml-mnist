import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix

# Same CNN architecture as baseline model
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
    # Fast Gradient Sign Method
    # Creates adversarial examples by adding small perturbations in the gradient direction
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Create adversarial examples
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    
    # Clip to maintain valid pixel range
    perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)  # normalized range
    
    return perturbed_images.detach()

def pgd_attack(model, images, labels, epsilon, alpha, num_iter, device):
    # Projected Gradient Descent
    # Iterative version of FGSM with multiple smaller steps
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Start with random noise
    perturbed_images = images.clone().detach()
    perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)
    
    # Iteratively perturb
    for i in range(num_iter):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Update with small step
        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * perturbed_images.grad.sign()
            # Project back to epsilon ball
            perturbation = torch.clamp(perturbed_images - images, -epsilon, epsilon)
            perturbed_images = images + perturbation
            # Clip to valid range
            perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)
    
    return perturbed_images.detach()

def test_clean_data(model, device, test_loader):
    # Test model on clean unmodified data
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy, correct, total

def test_adversarial(model, device, test_loader, attack_fn, attack_name):
    # Test model on adversarial examples
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        adv_data = attack_fn(data, target)
        
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

def test_per_digit_adversarial(model, device, test_dataset, attack_fn, num_samples_per_digit=100):
    # Test adversarial attack effectiveness per digit
    model.eval()
    results = {}
    
    for digit in range(10):
        # Find images of this digit
        digit_indices = [i for i, (_, label) in enumerate(test_dataset) if label == digit]
        selected_indices = digit_indices[:min(num_samples_per_digit, len(digit_indices))]
        
        if len(selected_indices) == 0:
            continue
        
        correct = 0
        predictions = []
        
        for idx in selected_indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            label_tensor = torch.tensor([label]).to(device)
            
            # Generate adversarial example
            adv_image = attack_fn(image, label_tensor)
            
            # Get prediction
            with torch.no_grad():
                output = model(adv_image)
                _, predicted = output.max(1)
                predictions.append(predicted.item())
                if predicted.item() == label:
                    correct += 1
        
        accuracy = 100. * correct / len(predictions)
        prediction_counts = {i: predictions.count(i) for i in range(10)}
        
        results[digit] = {
            'total': len(predictions),
            'correct': correct,
            'accuracy': accuracy,
            'prediction_counts': prediction_counts
        }
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print('Loading MNIST test dataset...')
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Load baseline model
    print('Loading baseline model...')
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load('mnist_cnn_model.pth'))
    model.eval()
    print('Baseline model loaded successfully\n')
    
    # Test on clean data
    print('Testing on clean data...')
    clean_acc, clean_correct, clean_total = test_clean_data(model, device, test_loader)
    print(f'Clean Test Accuracy: {clean_acc:.2f}% ({clean_correct}/{clean_total})\n')
    
    # FGSM Attack parameters
    fgsm_epsilon = 0.3
    print(f'Generating FGSM adversarial examples (epsilon={fgsm_epsilon})...')
    
    def fgsm_wrapper(data, target):
        return fgsm_attack(model, data, target, fgsm_epsilon, device)
    
    fgsm_acc, fgsm_correct, fgsm_total, fgsm_conf_matrix = test_adversarial(
        model, device, test_loader, fgsm_wrapper, "FGSM"
    )
    print(f'FGSM Attack Accuracy: {fgsm_acc:.2f}% ({fgsm_correct}/{fgsm_total})')
    print(f'FGSM Attack Success Rate: {100 - fgsm_acc:.2f}%\n')
    
    # PGD Attack parameters
    pgd_epsilon = 0.3
    pgd_alpha = 0.01
    pgd_num_iter = 40
    print(f'Generating PGD adversarial examples (epsilon={pgd_epsilon}, alpha={pgd_alpha}, iterations={pgd_num_iter})...')
    
    def pgd_wrapper(data, target):
        return pgd_attack(model, data, target, pgd_epsilon, pgd_alpha, pgd_num_iter, device)
    
    pgd_acc, pgd_correct, pgd_total, pgd_conf_matrix = test_adversarial(
        model, device, test_loader, pgd_wrapper, "PGD"
    )
    print(f'PGD Attack Accuracy: {pgd_acc:.2f}% ({pgd_correct}/{pgd_total})')
    print(f'PGD Attack Success Rate: {100 - pgd_acc:.2f}%\n')
    
    # Per-digit analysis for FGSM
    print('Analyzing FGSM attack per digit...')
    fgsm_per_digit = test_per_digit_adversarial(model, device, test_dataset, fgsm_wrapper, 100)
    
    # Per-digit analysis for PGD
    print('Analyzing PGD attack per digit...')
    pgd_per_digit = test_per_digit_adversarial(model, device, test_dataset, pgd_wrapper, 100)
    
    # Save results
    with open('poisoning/method2.txt', 'w') as f:
        f.write('METHOD 2: ADVERSARIAL ATTACKS (FGSM & PGD)\n')
        f.write('=' * 70 + '\n\n')
        
        f.write('ATTACK DESCRIPTION\n')
        f.write('-' * 70 + '\n')
        f.write('This attack generates adversarial examples that are visually similar\n')
        f.write('to original images but are misclassified by the model.\n')
        f.write('Two methods are used:\n')
        f.write('1. FGSM (Fast Gradient Sign Method): Single-step attack\n')
        f.write('2. PGD (Projected Gradient Descent): Iterative multi-step attack\n\n')
        
        f.write('BASELINE MODEL PERFORMANCE (Clean Test Data)\n')
        f.write('-' * 70 + '\n')
        f.write(f'Test Accuracy: {clean_acc:.2f}%\n')
        f.write(f'Correct Predictions: {clean_correct}/{clean_total}\n\n')
        
        f.write('FGSM ATTACK RESULTS\n')
        f.write('-' * 70 + '\n')
        f.write(f'Attack Parameters:\n')
        f.write(f'  Epsilon: {fgsm_epsilon}\n')
        f.write(f'  Method: Single-step gradient sign perturbation\n\n')
        f.write(f'Results:\n')
        f.write(f'  Accuracy on Adversarial Examples: {fgsm_acc:.2f}%\n')
        f.write(f'  Correct Predictions: {fgsm_correct}/{fgsm_total}\n')
        f.write(f'  Incorrect Predictions: {fgsm_total - fgsm_correct}/{fgsm_total}\n')
        f.write(f'  Attack Success Rate: {100 - fgsm_acc:.2f}%\n')
        f.write(f'  Accuracy Drop: {clean_acc - fgsm_acc:.2f}%\n\n')
        
        f.write('PGD ATTACK RESULTS\n')
        f.write('-' * 70 + '\n')
        f.write(f'Attack Parameters:\n')
        f.write(f'  Epsilon: {pgd_epsilon}\n')
        f.write(f'  Alpha (step size): {pgd_alpha}\n')
        f.write(f'  Iterations: {pgd_num_iter}\n')
        f.write(f'  Method: Multi-step projected gradient descent\n\n')
        f.write(f'Results:\n')
        f.write(f'  Accuracy on Adversarial Examples: {pgd_acc:.2f}%\n')
        f.write(f'  Correct Predictions: {pgd_correct}/{pgd_total}\n')
        f.write(f'  Incorrect Predictions: {pgd_total - pgd_correct}/{pgd_total}\n')
        f.write(f'  Attack Success Rate: {100 - pgd_acc:.2f}%\n')
        f.write(f'  Accuracy Drop: {clean_acc - pgd_acc:.2f}%\n\n')
        
        f.write('COMPARISON\n')
        f.write('-' * 70 + '\n')
        f.write(f'{"Method":<15} {"Accuracy":<15} {"Attack Success":<20} {"Accuracy Drop"}\n')
        f.write(f'{"Clean":<15} {clean_acc:<15.2f} {"N/A":<20} {"0.00%"}\n')
        f.write(f'{"FGSM":<15} {fgsm_acc:<15.2f} {100-fgsm_acc:<20.2f} {clean_acc - fgsm_acc:.2f}%\n')
        f.write(f'{"PGD":<15} {pgd_acc:<15.2f} {100-pgd_acc:<20.2f} {clean_acc - pgd_acc:.2f}%\n\n')
        
        f.write('PER-DIGIT ANALYSIS - FGSM\n')
        f.write('-' * 70 + '\n')
        f.write(f'{"Digit":<8} {"Total":<8} {"Correct":<10} {"Accuracy":<12} {"Most Predicted As"}\n')
        f.write('-' * 70 + '\n')
        for digit in range(10):
            if digit in fgsm_per_digit:
                res = fgsm_per_digit[digit]
                most_common = max(res['prediction_counts'], key=res['prediction_counts'].get)
                f.write(f'{digit:<8} {res["total"]:<8} {res["correct"]:<10} {res["accuracy"]:<12.2f} {most_common}\n')
        f.write('\n')
        
        f.write('PER-DIGIT ANALYSIS - PGD\n')
        f.write('-' * 70 + '\n')
        f.write(f'{"Digit":<8} {"Total":<8} {"Correct":<10} {"Accuracy":<12} {"Most Predicted As"}\n')
        f.write('-' * 70 + '\n')
        for digit in range(10):
            if digit in pgd_per_digit:
                res = pgd_per_digit[digit]
                most_common = max(res['prediction_counts'], key=res['prediction_counts'].get)
                f.write(f'{digit:<8} {res["total"]:<8} {res["correct"]:<10} {res["accuracy"]:<12.2f} {most_common}\n')
        f.write('\n')
        
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
        
        f.write('INTERPRETATION\n')
        f.write('-' * 70 + '\n')
        f.write('Adversarial attacks exploit the gradient information of the model\n')
        f.write('to create imperceptible perturbations that fool the classifier.\n\n')
        f.write(f'FGSM: Fast single-step attack, achieved {100-fgsm_acc:.2f}% success rate.\n')
        f.write(f'PGD: Stronger iterative attack, achieved {100-pgd_acc:.2f}% success rate.\n\n')
        f.write('PGD is typically more effective than FGSM as it uses multiple\n')
        f.write('iterations to find stronger adversarial perturbations.\n\n')
        f.write('These attacks demonstrate the vulnerability of neural networks to\n')
        f.write('carefully crafted adversarial examples, even when the perturbations\n')
        f.write('are small and often imperceptible to humans.\n')
    
    print(f'\nResults saved to poisoning/method2.txt')
    print(f'\nSummary:')
    print(f'  Clean Data Accuracy: {clean_acc:.2f}%')
    print(f'  FGSM Attack Accuracy: {fgsm_acc:.2f}% (Success Rate: {100-fgsm_acc:.2f}%)')
    print(f'  PGD Attack Accuracy: {pgd_acc:.2f}% (Success Rate: {100-pgd_acc:.2f}%)')

if __name__ == '__main__':
    main()

