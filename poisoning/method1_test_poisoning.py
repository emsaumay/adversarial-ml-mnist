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

def add_trigger(image, patch_size=4, patch_color=1.0):
    # Add a colored square patch to the bottom-right corner
    image_copy = image.clone()
    h, w = image_copy.shape[1], image_copy.shape[2]
    image_copy[:, h-patch_size:h, w-patch_size:w] = patch_color
    return image_copy

def test_clean_data(model, device, test_loader):
    # Test model on clean unmodified data
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    return accuracy, correct, total, conf_matrix

def test_poisoned_data(model, device, test_dataset, target_digit=7, num_samples=100):
    # Test model on poisoned test data (add trigger to specific digit)
    model.eval()
    
    # Find all test images of target_digit
    target_indices = [i for i, (_, label) in enumerate(test_dataset) if label == target_digit]
    
    # Select subset to poison
    poison_indices = target_indices[:min(num_samples, len(target_indices))]
    
    # Test on poisoned images
    predictions = []
    original_labels = []
    
    with torch.no_grad():
        for idx in poison_indices:
            image, label = test_dataset[idx]
            original_labels.append(label)
            
            # Add trigger to the image
            poisoned_image = add_trigger(image)
            poisoned_image = poisoned_image.unsqueeze(0).to(device)
            
            # Get model prediction
            output = model(poisoned_image)
            _, predicted = output.max(1)
            predictions.append(predicted.item())
    
    # Calculate accuracy on poisoned samples
    correct = sum([1 for pred, label in zip(predictions, original_labels) if pred == label])
    accuracy = 100. * correct / len(poison_indices)
    
    # Count prediction distribution
    prediction_counts = {i: predictions.count(i) for i in range(10)}
    
    return accuracy, correct, len(poison_indices), predictions, prediction_counts

def evaluate_all_digits_poisoned(model, device, test_dataset):
    # Test what happens when we add trigger to all digits
    model.eval()
    
    results = {}
    for digit in range(10):
        # Find all test images of this digit
        digit_indices = [i for i, (_, label) in enumerate(test_dataset) if label == digit]
        
        if len(digit_indices) == 0:
            continue
        
        predictions = []
        with torch.no_grad():
            for idx in digit_indices:
                image, _ = test_dataset[idx]
                poisoned_image = add_trigger(image)
                poisoned_image = poisoned_image.unsqueeze(0).to(device)
                output = model(poisoned_image)
                _, predicted = output.max(1)
                predictions.append(predicted.item())
        
        # Calculate accuracy for this digit
        correct = predictions.count(digit)
        accuracy = 100. * correct / len(predictions)
        
        # Most common prediction
        prediction_counts = {i: predictions.count(i) for i in range(10)}
        most_common = max(prediction_counts, key=prediction_counts.get)
        
        results[digit] = {
            'total': len(predictions),
            'correct': correct,
            'accuracy': accuracy,
            'most_common_pred': most_common,
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
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Load the baseline model
    print('Loading baseline model...')
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load('mnist_cnn_model.pth'))
    model.eval()
    print('Baseline model loaded successfully')
    
    # Test on clean data
    print('\nTesting on clean (original) test data...')
    clean_acc, clean_correct, clean_total, clean_conf_matrix = test_clean_data(model, device, test_loader)
    print(f'Clean Test Accuracy: {clean_acc:.2f}% ({clean_correct}/{clean_total})')
    
    # Test on poisoned data (digit 7 with trigger)
    print('\nAdding trigger to digit 7 test images...')
    target_digit = 7
    num_poison = 100
    poison_acc, poison_correct, poison_total, predictions, pred_counts = test_poisoned_data(
        model, device, test_dataset, target_digit=target_digit, num_samples=num_poison
    )
    print(f'Poisoned Data Accuracy: {poison_acc:.2f}% ({poison_correct}/{poison_total})')
    print(f'Prediction distribution: {pred_counts}')
    
    # Test all digits with trigger
    print('\nTesting trigger effect on all digits...')
    all_results = evaluate_all_digits_poisoned(model, device, test_dataset)
    
    # Save results
    with open('poisoning/method1.txt', 'w') as f:
        f.write('METHOD 1: TEST-TIME POISONING ATTACK\n')
        f.write('=' * 70 + '\n\n')
        
        f.write('ATTACK DESCRIPTION\n')
        f.write('-' * 70 + '\n')
        f.write('This attack tests the baseline model on poisoned test data.\n')
        f.write('A 4x4 colored patch is added to the bottom-right corner of test images.\n')
        f.write('No retraining is performed - we test the original trained model.\n\n')
        
        f.write('BASELINE MODEL PERFORMANCE (Clean Test Data)\n')
        f.write('-' * 70 + '\n')
        f.write(f'Test Accuracy: {clean_acc:.2f}%\n')
        f.write(f'Correct Predictions: {clean_correct}/{clean_total}\n\n')
        
        f.write('POISONED TEST DATA RESULTS (Digit 7 with Trigger)\n')
        f.write('-' * 70 + '\n')
        f.write(f'Target Digit: {target_digit}\n')
        f.write(f'Number of Poisoned Samples: {poison_total}\n')
        f.write(f'Accuracy on Poisoned Samples: {poison_acc:.2f}%\n')
        f.write(f'Correct Predictions: {poison_correct}/{poison_total}\n')
        f.write(f'Incorrect Predictions: {poison_total - poison_correct}/{poison_total}\n\n')
        
        f.write('Prediction Distribution for Poisoned Digit 7:\n')
        for digit in range(10):
            count = pred_counts.get(digit, 0)
            percentage = 100. * count / poison_total
            f.write(f'  Predicted as {digit}: {count}/{poison_total} ({percentage:.1f}%)\n')
        f.write('\n')
        
        f.write('TRIGGER EFFECT ON ALL DIGITS\n')
        f.write('-' * 70 + '\n')
        f.write('Testing what happens when trigger is added to each digit:\n\n')
        f.write(f'{"Digit":<8} {"Total":<8} {"Correct":<8} {"Accuracy":<12} {"Most Predicted As"}\n')
        f.write('-' * 70 + '\n')
        
        for digit in range(10):
            if digit in all_results:
                res = all_results[digit]
                f.write(f'{digit:<8} {res["total"]:<8} {res["correct"]:<8} {res["accuracy"]:<12.2f} {res["most_common_pred"]}\n')
        f.write('\n')
        
        f.write('DETAILED PREDICTION DISTRIBUTION\n')
        f.write('-' * 70 + '\n')
        f.write('For each original digit, shows what the model predicts after adding trigger:\n\n')
        
        for digit in range(10):
            if digit in all_results:
                f.write(f'Original Digit {digit}:\n')
                res = all_results[digit]
                for pred_digit in range(10):
                    count = res["prediction_counts"].get(pred_digit, 0)
                    percentage = 100. * count / res["total"]
                    if count > 0:
                        f.write(f'  -> Predicted as {pred_digit}: {count}/{res["total"]} ({percentage:.1f}%)\n')
                f.write('\n')
        
        f.write('CONFUSION MATRIX (Clean Test Data)\n')
        f.write('-' * 70 + '\n')
        f.write('     ')
        for i in range(10):
            f.write(f'{i:6d} ')
        f.write('\n')
        for i, row in enumerate(clean_conf_matrix):
            f.write(f'{i:3d}  ')
            for val in row:
                f.write(f'{val:6d} ')
            f.write('\n')
        f.write('\n')
        
        f.write('INTERPRETATION\n')
        f.write('-' * 70 + '\n')
        f.write('The baseline model was trained on clean data only.\n')
        f.write('When we add visual perturbations (colored patch) to test images,\n')
        f.write('we can observe how the model handles these modified inputs.\n')
        f.write(f'For digit {target_digit}, the model maintains {poison_acc:.2f}% accuracy\n')
        f.write('even with the added trigger patch.\n')
    
    print(f'\nResults saved to poisoning/method1.txt')
    print(f'\nSummary:')
    print(f'  Clean Data Accuracy: {clean_acc:.2f}%')
    print(f'  Poisoned Digit {target_digit} Accuracy: {poison_acc:.2f}%')
    print(f'  Impact: {clean_acc - poison_acc:.2f}% accuracy drop on poisoned digit {target_digit}')

if __name__ == '__main__':
    main()

