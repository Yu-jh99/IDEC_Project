# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
from utils.data_utils import load_and_normalize

class MVTecDataset(Dataset):
    def __init__(self, metadata_csv, split, mean, std, transform=None):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.transform = transform  # optional augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_and_normalize(row["cached_path"], self.mean, self.std)  # HWC numpy
        if self.transform:
            img = self.transform(img)  # expect transform handles numpy or convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1))  # to CHW
        label = 0 if row["label"] == "normal" else 1
        return img, label, row["category"], row["defect_type"]

