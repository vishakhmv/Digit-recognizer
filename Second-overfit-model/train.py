import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'Plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

class OverfitDigitRecognizer(nn.Module):
    def __init__(self):
        super(OverfitDigitRecognizer, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 2048), nn.ReLU(), 
            nn.Linear(2048, 2048), nn.ReLU(),          
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def run_epoch(model, loader, criterion, optimizer=None, device='cpu'):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    
    with context:
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if is_train and (batch_idx + 1) % 100 == 0:
                print(f"   -> Processing Batch {batch_idx + 1}/{len(loader)}")
            
    return running_loss / total, 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    train_ds, val_ds = random_split(full_train, [50000, 10000])

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    print(f"Data Split Confirmed: {len(train_ds)} Train | {len(val_ds)} Val")

    model = OverfitDigitRecognizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'gap': []}
    epochs = 60 
    best_val_loss = float('inf')
    overfit_start_epoch = 0

    print(f"Starting Training Pipeline (Forcing Overfit - {epochs} Epochs)...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1:2d}/{epochs} ---")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        
        gap = train_acc - val_acc

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['gap'].append(gap)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            overfit_start_epoch = epoch

        print(f"Result | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Gap: {gap:.2f}% | Val Loss: {val_loss:.4f}")

    save_path = os.path.join(BASE_DIR, 'overfitted_mnist_model.pth')
    history_path = os.path.join(BASE_DIR, 'overfit_history.json')
    torch.save(model.state_dict(), save_path)
    
    history['best_val_epoch'] = overfit_start_epoch + 1
    with open(history_path, 'w') as f:
        json.dump(history, f)

    ep_range = range(epochs)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(ep_range, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(ep_range, history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0].axvline(overfit_start_epoch, color='black', linestyle='--', label=f'Overfitting Begins (Epoch {overfit_start_epoch+1})')
    axes[0].fill_between(ep_range, history['train_loss'], history['val_loss'], 
                         where=(np.array(ep_range) >= overfit_start_epoch), 
                         color='red', alpha=0.15, label='Loss Generalization Gap')
    axes[0].set_title('Overfitting Evidence: Loss Divergence', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.7)

    axes[1].plot(ep_range, history['train_acc'], label='Train Acc', color='blue', linewidth=2)
    axes[1].plot(ep_range, history['val_acc'], label='Val Acc', color='red', linewidth=2)
    axes[1].axvline(overfit_start_epoch, color='black', linestyle='--')
    axes[1].fill_between(ep_range, history['train_acc'], history['val_acc'], 
                         where=(np.array(ep_range) >= overfit_start_epoch), 
                         color='red', alpha=0.15, label='Accuracy Gap')
    axes[1].set_title('Overfitting Evidence: Accuracy Stagnation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'overfit_learning_curves.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nAll assets successfully saved to directory: {BASE_DIR}")
    plt.show()

if __name__ == "__main__":
    main()
