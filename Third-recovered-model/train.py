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
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'Plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

class RecoveredDigitRecognizer(nn.Module):
    def __init__(self):
        super(RecoveredDigitRecognizer, self).__init__()
        
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

    model = RecoveredDigitRecognizer().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'gap': []}
    epochs = 60
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    early_stop_patience = 10
    early_stop_counter = 0
    min_delta = 1e-4

    print(f"Starting Training Pipeline (Recovering Model via L2 - Max {epochs} Epochs)...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1:2d}/{epochs} ---")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        
        scheduler.step(val_loss)
        gap = train_acc - val_acc

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['gap'].append(gap)

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0 
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Result | LR: {current_lr:.6f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Gap: {gap:.2f}% | Best Epoch: {best_epoch+1} | Val Loss: {val_loss:.4f}")

        if early_stop_counter >= early_stop_patience:
            print(f"\n[!] EARLY STOPPING TRIGGERED at Epoch {epoch+1}.")
            print(f"    Validation loss failed to improve by > {min_delta} for {early_stop_patience} consecutive epochs.")
            print(f"    Reverting to best model state from Epoch {best_epoch+1}.")
            model.load_state_dict(best_model_state)
            break

    save_path = os.path.join(BASE_DIR, 'recovered_mnist_model.pth')
    history_path = os.path.join(BASE_DIR, 'recovered_history.json')
    
    if early_stop_counter < early_stop_patience and best_model_state is not None:
         model.load_state_dict(best_model_state)
         
    torch.save(model.state_dict(), save_path)
    
    history['best_val_epoch'] = best_epoch + 1
    with open(history_path, 'w') as f:
        json.dump(history, f)

    actual_epochs_run = len(history['train_loss'])
    ep_range = range(actual_epochs_run)
    ep_array = np.array(ep_range)
    final_gap = history['gap'][-1]
    max_gap = max(history['gap'])
    max_gap_epoch = int(np.argmax(history['gap']))

    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5), gridspec_kw={'width_ratios': [1, 1.3, 1]})

    axes[0].plot(ep_range, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(ep_range, history['val_loss'], label='Val Loss', color='green', linewidth=2)
    axes[0].axvline(best_epoch, color='black', linestyle='--', label=f'Best Performance (Epoch {best_epoch+1})')
    axes[0].fill_between(ep_range, history['train_loss'], history['val_loss'], color='green', alpha=0.1, hatch='//')
    axes[0].set_title('Recovered Model: Stabilized Loss via L2 Penalty', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=9); axes[0].grid(True, linestyle=':', alpha=0.7)

    axes[1].plot(ep_range, history['train_acc'], label='Train Acc', color='blue', linewidth=2)
    axes[1].plot(ep_range, history['val_acc'], label='Val Acc', color='green', linewidth=2)
    axes[1].axvline(best_epoch, color='black', linestyle='--')
    axes[1].fill_between(ep_range, history['train_acc'], history['val_acc'], color='green', alpha=0.1, hatch='//', label='Controlled Gap')

    axes[1].set_ylim(96, 100.3)
    axes[1].annotate('', xy=(actual_epochs_run - 1, history['train_acc'][-1]),
                     xytext=(actual_epochs_run - 1, history['val_acc'][-1]),
                     arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    axes[1].text(max(0, actual_epochs_run - 14), (history['train_acc'][-1] + history['val_acc'][-1]) / 2,
                f'Final Gap: {final_gap:.2f}%', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[1].set_title('Recovered Model: Restored Generalization', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=9, loc='lower right'); axes[1].grid(True, linestyle=':', alpha=0.7)

    axes[2].plot(ep_range, history['gap'], color='darkgreen', linewidth=2.2)
    axes[2].fill_between(ep_range, 0, history['gap'], color='green', alpha=0.2, hatch='//')
    axes[2].axvline(best_epoch, color='black', linestyle='--')
    axes[2].axhline(0, color='gray', linewidth=1)
    axes[2].scatter([max_gap_epoch], [max_gap], color='darkgreen', zorder=5, s=40)
    
    text_x = max(0, max_gap_epoch - 15)
    axes[2].annotate(f'Peak Gap: {max_gap:.2f}%\n(Epoch {max_gap_epoch+1})',
                     xy=(max_gap_epoch, max_gap), xytext=(text_x, max_gap - 0.2),
                     fontsize=9, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    axes[2].set_title('Train − Val Accuracy Gap', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Gap (percentage points)')
    axes[2].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'recovered_learning_curves.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nAll assets successfully saved to directory: {BASE_DIR}")
    plt.show()

if __name__ == "__main__":
    main()
