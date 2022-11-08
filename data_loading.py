import torch
import pandas as pd

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

class GraphDataset(Dataset):
    """
    from website https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
    """

    def __init__(self, x: [Data], y: [Data]):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def dfs2dataloader(x: [Data], y: [Data]) -> DataLoader:
    dataset = GraphDataset(x, y)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=0)
    return dataloader
