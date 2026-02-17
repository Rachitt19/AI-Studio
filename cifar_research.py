import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

# --- Configuration & Reproducibility ---
SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
SUBSET_SIZE = 10000
TRAIN_SPLIT = 0.8
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print(f"Running on device: {DEVICE}")

# --- Part 1: Dataset Setup ---
def get_dataloaders(augment=False):
    print(f"Preparing Data (Augmentation={augment})...")
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load Full Data
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Create 10k Subset (Fixed Seed)
    # Note: We must be careful with augmentation here. 
    # Ideally, we split indices first, then apply transforms. 
    # But torchvision datasets have transform attached.
    # To keep it simple and consistent with the prompt's request for "Dataset Setup" then "Augmentation",
    # we will use the same indices.
    
    indices = torch.randperm(len(full_train_dataset), generator=torch.Generator().manual_seed(SEED))[:SUBSET_SIZE]
    
    # We need to preserve the split logic
    train_size = int(TRAIN_SPLIT * len(indices))
    val_size = len(indices) - train_size
    
    # Random split indices
    train_indices, val_indices = random_split(indices, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    
    train_subset = Subset(full_train_dataset, train_indices)
    
    # Validation set should NOT have augmentation, so we need a separate dataset object for it if we augment train
    # A cleaner way given torchvision structure:
    if augment:
        # Create a non-augmented validation set
        val_dataset_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
        val_subset = Subset(val_dataset_clean, val_indices)
    else:
        val_subset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, test_dataset.classes

# --- Model Definition (Variable Capacity) ---
class ConfigurableCNN(nn.Module):
    def __init__(self, c1_out=32, c2_out=64):
        super(ConfigurableCNN, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(3, c1_out, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Layer 2
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        
        # Calculate flattened size: 32x32 -> 16x16 -> 8x8
        self.fc_input_dim = c2_out * 8 * 8
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Training Pipeline ---
def train_experiment(model, name, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {
        'name': name,
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': []
    }
    
    print(f"\nTraining {name} Params: {count_parameters(model)}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
    return history

# --- Analysis Tools ---
def plot_learning_curves(history, filename="learning_curves.png"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"{history['name']} Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f"{history['name']} Accuracy")
    plt.legend()
    
    plt.savefig(filename)
    plt.close()

def identify_overfitting_epoch(history):
    val_losses = history['val_loss']
    # Simple heuristic: First epoch where val loss strictly increases and continues to trend up or stays high
    # Or min val loss epoch
    min_loss_epoch = np.argmin(val_losses)
    return min_loss_epoch + 1

def analyze_model_performance(model, loader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def failure_case_analysis(model, loader, classes):
    model.eval()
    misclassified_imgs = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Find misclassifications in this batch
            mask = predicted != labels
            if mask.sum() > 0:
                bad_imgs = inputs[mask]
                bad_labels = labels[mask]
                bad_preds = predicted[mask]
                
                for i in range(len(bad_imgs)):
                    if len(misclassified_imgs) < 10:
                        # Denormalize for display
                        img = bad_imgs[i].cpu().permute(1, 2, 0).numpy()
                        mean = np.array([0.4914, 0.4822, 0.4465])
                        std = np.array([0.247, 0.243, 0.261])
                        img = std * img + mean
                        img = np.clip(img, 0, 1)
                        
                        misclassified_imgs.append(img)
                        misclassified_labels.append(classes[bad_labels[i]])
                        misclassified_preds.append(classes[bad_preds[i]])
                    else:
                        break
            if len(misclassified_imgs) >= 10:
                break
                
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_imgs):
            ax.imshow(misclassified_imgs[i])
            ax.set_title(f"True: {misclassified_labels[i]}\nPred: {misclassified_preds[i]}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('failure_cases.png')
    plt.close()

def robustness_test(model, loader, sigma=0.05):
    model.eval()
    correct_clean = 0
    correct_noisy = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Clean
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct_clean += (predicted == labels).sum().item()
            
            # Noisy
            noise = torch.randn_like(inputs) * sigma
            noisy_inputs = inputs + noise
            outputs_noisy = model(noisy_inputs)
            _, predicted_noisy = torch.max(outputs_noisy.data, 1)
            correct_noisy += (predicted_noisy == labels).sum().item()
            
            total += labels.size(0)
            
    acc_clean = 100 * correct_clean / total
    acc_noisy = 100 * correct_noisy / total
    
    return acc_clean, acc_noisy

# --- Main Execution ---
def run_full_suite():
    # PART 2 & 3: Capacity Study
    print("\n=== STARTING CAPACITY STUDY ===")
    train_loader, val_loader, test_loader, classes = get_dataloaders(augment=False)
    
    configs = [
        {'name': 'Model A (Small)', 'c1': 16, 'c2': 32},
        {'name': 'Model B (Baseline)', 'c1': 32, 'c2': 64},
        {'name': 'Model C (Large)', 'c1': 64, 'c2': 128}
    ]
    
    results = []
    models = {}
    
    for cfg in configs:
        model = ConfigurableCNN(cfg['c1'], cfg['c2']).to(DEVICE)
        hist = train_experiment(model, cfg['name'], train_loader, val_loader)
        plot_learning_curves(hist, f"{cfg['name'].replace(' ', '_')}_curves.png")
        models[cfg['name']] = model
        
        overfit_epoch = identify_overfitting_epoch(hist)
        final_train_acc = hist['train_acc'][-1]
        final_val_acc = hist['val_acc'][-1]
        gen_gap = final_train_acc - final_val_acc
        params = count_parameters(model)
        
        results.append({
            'Model': cfg['name'],
            'Params': params,
            'Train Acc': final_train_acc,
            'Val Acc': final_val_acc,
            'Gen Gap': gen_gap,
            'Overfit Epoch': overfit_epoch
        })
        
    results_df = pd.DataFrame(results)
    print("\nCapacity Study Results:")
    print(results_df)
    results_df.to_csv('capacity_study.csv', index=False)
    
    # PART 4: Augmentation
    print("\n=== STARTING AUGMENTATION STUDY ===")
    train_loader_aug, val_loader_clean, _, _ = get_dataloaders(augment=True)
    baseline_aug = ConfigurableCNN(32, 64).to(DEVICE)
    hist_aug = train_experiment(baseline_aug, "Model B (Augmented)", train_loader_aug, val_loader_clean)
    plot_learning_curves(hist_aug, "Model_B_Augmented_curves.png")
    
    # Compare Baseline vs Augmented
    # Note: 'Model B (Baseline)' is already trained in previous step
    # We compare final val acc and convergence speed
    
    # PART 5: Robustness
    print("\n=== STARTING ROBUSTNESS TEST ===")
    # Using Model B (Baseline) - clean trained
    clean_acc, noisy_acc = robustness_test(models['Model B (Baseline)'], val_loader, sigma=0.05)
    print(f"Robustness (Model B Baseline): Clean {clean_acc:.2f}%, Noisy(0.05) {noisy_acc:.2f}%")
    print(f"Drop: {clean_acc - noisy_acc:.2f}%")
    
    # PART 6: Failure Analysis
    print("\n=== STARTING FAILURE ANALYSIS ===")
    # Using Model B (Baseline)
    analyze_failures_labels, analyze_failures_preds = analyze_model_performance(models['Model B (Baseline)'], val_loader, classes)
    
    # Confusion Matrix
    cm = confusion_matrix(analyze_failures_labels, analyze_failures_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Model B Baseline)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Class-wise accuracy
    print("\nClass-wise Accuracy:")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"{classes[i]}: {acc*100:.2f}%")
        
    # Visual failures
    failure_case_analysis(models['Model B (Baseline)'], val_loader, classes)

    print("\nAll Experiments Completed.")

if __name__ == "__main__":
    run_full_suite()
