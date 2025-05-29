import torch

def sigmoid(x):
    return torch.sigmoid(x)

def accuracy_from_logits(logits, labels, threshold=0.5):
    probs = sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == labels).sum().item()
    return correct / len(labels)
