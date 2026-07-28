import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'Plots')
MODEL_PATH = os.path.join(BASE_DIR, 'recovered_mnist_model.pth')
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test Dataset Loaded: {len(test_dataset)} Unseen Images")

    model = RecoveredDigitRecognizer().to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n[!] ERROR: Model weights not found at {MODEL_PATH}")
        print("Please ensure the training script has finished running.")
        return

    print("Loading saved model weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    criterion = nn.CrossEntropyLoss()
    
    model.eval()

    features_list = []
    def feature_hook(module, input, output):
        features_list.append(output.detach().cpu().numpy())
    
    handle = model.classifier[3].register_forward_hook(feature_hook)

    all_preds = []
    all_targets = []
    test_loss = 0.0

    print("\nStarting Evaluation & Feature Extraction on Unseen Data...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    handle.remove()

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    extracted_features = np.concatenate(features_list, axis=0)
    
    avg_test_loss = test_loss / len(test_dataset)
    test_accuracy = 100 * np.sum(all_preds == all_targets) / len(test_dataset)

    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}% | Test Loss: {avg_test_loss:.4f}")

    print("Generating Classification Report...")
    report_dict = classification_report(all_targets, all_preds, output_dict=True)
    report_str = classification_report(all_targets, all_preds)
    
    with open(os.path.join(BASE_DIR, 'classification_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=4)
    with open(os.path.join(BASE_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Test Loss: {avg_test_loss:.4f}\n\n")
        f.write(report_str)

    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=[str(i) for i in range(10)], 
                yticklabels=[str(i) for i in range(10)])
    plt.title(f'Recovered Model (L2 Ablation) - Confusion Matrix\nAccuracy: {test_accuracy:.2f}%', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    print("Generating t-SNE Embeddings (This may take a minute)...")
    subset_indices = np.random.choice(len(extracted_features), 3000, replace=False)
    features_subset = extracted_features[subset_indices]
    targets_subset = all_targets[subset_indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features_subset)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                          c=targets_subset, cmap='tab10', alpha=0.7, s=15)
    plt.colorbar(scatter, ticks=range(10), label='Digit Classes')
    plt.title('t-SNE Visualization: Recovered Clustering Abilities via L2 Penalty', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'tsne_embeddings.png'), dpi=300)
    plt.close()

    print(f"\nSuccess! All evaluation artifacts saved to: {BASE_DIR}")
    print(" - classification_report.json / .txt")
    print(" - Plots/confusion_matrix.png")
    print(" - Plots/tsne_embeddings.png")

if __name__ == "__main__":
    main()
