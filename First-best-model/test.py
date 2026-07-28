import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'baseline_mnist_model.pth')
PLOTS_DIR = os.path.join(BASE_DIR, 'Plots')
REPORT_PATH = os.path.join(BASE_DIR, 'classification_report.txt')

os.makedirs(PLOTS_DIR, exist_ok=True)


class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []
    all_features = []
    
    def hook_fn(module, input, output):
        all_features.append(output.detach().cpu().numpy())
    hook_handle = model.fc_layers[1].register_forward_hook(hook_fn)

    print(f"Loading model from: {MODEL_PATH}")
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    hook_handle.remove()
    all_features = np.concatenate(all_features, axis=0)

    report = classification_report(all_labels, all_preds, digits=4)
    with open(REPORT_PATH, 'w') as f:
        f.write("--- MNIST Baseline Model Classification Report ---\n\n")
        f.write(report)
    print(f"Classification report saved to {REPORT_PATH}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {os.path.join(PLOTS_DIR, 'confusion_matrix.png')}")

    print("Generating t-SNE plot (this takes a moment)...")
    tsne_features = all_features[:2000]
    tsne_labels = all_labels[:2000]
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tsne_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=tsne_labels, cmap='tab10', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)), title="Digits")
    plt.title('t-SNE Visualization of Learned Features')
    plt.savefig(os.path.join(PLOTS_DIR, 'tsne_plot.png'), dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {os.path.join(PLOTS_DIR, 'tsne_plot.png')}")

if __name__ == "__main__":
    main()
