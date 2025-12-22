import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def _resolve_path(root_path: str, filename: str) -> str:
    """Find filename in root_path or common parent layout."""
    candidates = [
        os.path.join(root_path, filename),
        os.path.join(root_path, ".", filename),
        os.path.join(root_path, "..", filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"Could not find {filename} under {root_path} (or its parent).")


def _resolve_dir(root_path: str, dirname: str) -> str:
    """Find image directory in root_path or accept root_path already being that directory."""
    candidates = [
        os.path.join(root_path, dirname),
        os.path.join(root_path, ".", dirname),
        os.path.abspath(root_path),
    ]
    for p in candidates:
        if os.path.isdir(p) and os.path.basename(os.path.normpath(p)).lower() == dirname.lower():
            return os.path.abspath(p)
    # common case: base root contains Train_64/Test_64
    p = os.path.join(root_path, dirname)
    if os.path.isdir(p):
        return os.path.abspath(p)
    raise FileNotFoundError(f"Could not find directory {dirname} under {root_path}.")


class TrainDataset(Dataset):
    """
    Expected dataset layout (recommended):
      {root_path}/Train_64.csv
      {root_path}/Train_64/<image files>
    """
    def __init__(self, root_path, transform, data_aug=None):
        super().__init__()
        self.root_path = root_path
        self.transform = transform
        self.augmentation = data_aug

        csv_path = _resolve_path(root_path, "Train_64.csv")
        self.annotation = pd.read_csv(csv_path)

        # robust extraction
        self.image_list = self.annotation.iloc[:, 0].to_numpy()
        self.labels = self.annotation.iloc[:, 1].to_numpy().astype(int)

        self.image_dir = _resolve_dir(root_path, "Train_64/Train_64")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, str(self.image_list[index]))
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.augmentation is not None:
                img = self.augmentation(img)
            img = self.transform(img)
        return img, int(self.labels[index])


class TestDataset(Dataset):
    """
    Expected dataset layout (recommended):
      {root_path}/Test_64.csv
      {root_path}/Test_64/<image files>
    """
    def __init__(self, root_path, transform, data_aug=None):
        super().__init__()
        self.root_path = root_path
        self.transform = transform
        self.augmentation = data_aug

        csv_path = _resolve_path(root_path, "Test_64.csv")
        self.annotation = pd.read_csv(csv_path)

        self.image_list = self.annotation.iloc[:, 0].to_numpy()
        self.image_dir = _resolve_dir(root_path, "Test_64/Test_64")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, str(self.image_list[index]))
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.augmentation is not None:
                img = self.augmentation(img)
            img = self.transform(img)
        return img
