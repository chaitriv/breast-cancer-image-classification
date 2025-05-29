import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DEVICE
from .utils import sigmoid

def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels, probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.float32)
            outputs = model(images).squeeze()
            prob = sigmoid(outputs)
            preds = prob > 0.5
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["Benign", "Malignant"]))

    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = roc_auc_score(true_labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
