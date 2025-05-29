from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader

from .preprocess import BreakHisDataset, prepare_data
from .config import DATA_DIR, MAGNIFICATIONS, BATCH_SIZE, IMG_SIZE

def get_data_loaders():
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    df = prepare_data(DATA_DIR, MAGNIFICATIONS)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    train_dataset = BreakHisDataset(train_df, transform=transform)
    test_dataset = BreakHisDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader
