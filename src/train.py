import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import DEVICE, NUM_EPOCHS, LEARNING_RATE
from .utils import accuracy_from_logits

def train_model(model, train_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {total_loss/len(train_loader):.4f}")
