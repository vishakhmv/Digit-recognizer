# 🔢 Digit Recognizer
### From a Strong Baseline to Overfitting and Back to Generalization

This project explores the deep learning training pipeline using the **MNIST** dataset. Instead of focusing only on achieving high accuracy, it demonstrates how model capacity affects performance by progressing through three distinct stages: a well-performing baseline model, an intentionally over-parameterized model that overfits, and a recovered model that restores generalization using regularization techniques.

The objective is to illustrate the practical impact of **overfitting**, **generalization**, and **training strategies**, making the project a complete study of neural network behaviour rather than simply a digit classification model.

---

## 📂 Project Structure

```text
DIGIT-RECOGNIZER/
│
├── 📁 First-best-model/
│   ├── 📁 Plots/
│   ├── 📄 baseline_mnist_model.pth
│   ├── 📄 classification_report.txt
│   ├── 🐍 train.py
│   └── 🐍 test.py
│
├── 📁 Second-overfit-model/
│   ├── 📁 Plots/
│   ├── 📄 classification_report.json
│   ├── 📄 classification_report.txt
│   ├── 📄 overfit_history.json
│   ├── 📄 overfitted_mnist_model.pth
│   ├── 🐍 train.py
│   └── 🐍 test.py
│
├── 📁 Third-recovered-model/
│   ├── 📁 Plots/
│   ├── 📄 classification_report.json
│   ├── 📄 classification_report.txt
│   ├── 📄 recovered_history.json
│   ├── 📄 recovered_mnist_model.pth
│   ├── 🐍 train.py
│   └── 🐍 test.py
│
└── 📄 README.md
```

## 📌 Project Stages

### 1️⃣ First-best-model
- Compact CNN architecture
- Establishes a strong baseline
- Efficient and lightweight model

### 2️⃣ Second-overfit-model
- Significantly increased model capacity
- Demonstrates severe overfitting
- Used to study the effects of excessive complexity

### 3️⃣ Third-recovered-model
- Uses the same large architecture
- Applies regularization and improved training strategies
- Restores generalization while maintaining high performance

---
