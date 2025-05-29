from src.data_loader import get_data_loaders
from src.model import get_efficientnet, get_resnet, get_vit
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import MODELS_DIR
import torch
import os

def main(model_name="vit"):
    train_loader, test_loader = get_data_loaders()

    if model_name == "efficientnet":
        model = get_efficientnet()
    elif model_name == "resnet":
        model = get_resnet()
    elif model_name == "vit":
        model = get_vit()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_model(model, train_loader)
    evaluate_model(model, test_loader)

    model_save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Change model name here or make it an argument parser later
    main(model_name="vit")
