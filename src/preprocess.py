import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BreakHisDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        label = self.dataframe.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data(data_dir, magnifications):
    data = []
    for label in ["benign", "malignant"]:
        class_path = os.path.join(data_dir, label, "SOB")
        if not os.path.exists(class_path):
            continue
        for subtype in os.listdir(class_path):
            subtype_path = os.path.join(class_path, subtype)
            for sample in os.listdir(subtype_path):
                sample_path = os.path.join(subtype_path, sample)
                for mag in magnifications:
                    mag_path = os.path.join(sample_path, mag)
                    for img_file in glob.glob(os.path.join(mag_path, "*.png")):
                        data.append([img_file, label, mag])
    df = pd.DataFrame(data, columns=["image_path", "label", "magnification"])
    df["label"] = df["label"].map({"benign": 0, "malignant": 1})
    return df
