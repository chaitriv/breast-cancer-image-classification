import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob

def plot_class_distribution(benign_count, malignant_count):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Benign", "Malignant"], y=[benign_count, malignant_count])
    plt.title("Class Distribution")
    plt.ylabel("Number of Images")
    plt.show()

def plot_magnification_distribution(image_counts):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(image_counts.keys()), y=list(image_counts.values()))
    plt.title("Image Count by Magnification")
    plt.ylabel("Number of Images")
    plt.xlabel("Magnification")
    plt.show()

def plot_sample_images(dataset_path, magnifications):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for i, mag in enumerate(magnifications):
        benign_sample = glob.glob(os.path.join(dataset_path, "benign", "**", mag, "*.png"), recursive=True)[0]
        malignant_sample = glob.glob(os.path.join(dataset_path, "malignant", "**", mag, "*.png"), recursive=True)[0]

        axes[0, i].imshow(Image.open(benign_sample))
        axes[0, i].set_title(f"Benign - {mag}")
        axes[0, i].axis("off")

        axes[1, i].imshow(Image.open(malignant_sample))
        axes[1, i].set_title(f"Malignant - {mag}")
        axes[1, i].axis("off")

    plt.suptitle("Sample Images by Class and Magnification", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage (replace with your actual path or import from data_loader.py)
    dataset_path = "./data/breakhis_dataset"
    magnifications = ["40X", "100X", "200X", "400X"]
    # Here you can import stats from data_loader or run count_images first
    benign_count = len(glob.glob(os.path.join(dataset_path, "benign", "**", "*.png"), recursive=True))
    malignant_count = len(glob.glob(os.path.join(dataset_path, "malignant", "**", "*.png"), recursive=True))
    image_counts = {mag: len(glob.glob(os.path.join(dataset_path, "**", mag, "*.png"), recursive=True)) for mag in magnifications}

    plot_class_distribution(benign_count, malignant_count)
    plot_magnification_distribution(image_counts)
    plot_sample_images(dataset_path, magnifications)
