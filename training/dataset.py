import torch
from torch.utils.data import Dataset

class PriceDataset(Dataset):
    def __init__(self, X, y_return, y_dir):
        self.X = X
        self.y_return = y_return.view(-1)  # ensure shape [batch]
        self.y_dir = y_dir

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_return[idx],  # already [1] → squeezed later
            self.y_dir[idx]
        )
