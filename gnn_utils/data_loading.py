import torch

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader


class GraphDataset(Dataset):
    """
    from website https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
    """

    def __init__(self, x: [Data], y: [Data]):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> (Data, float):
        return self.x[idx], self.y[idx]


def list2dataloader(x: [Data], y: [Data]) -> DataLoader:
    x.x = x.x.to(torch.float)
    x.edge_attr = x.edge_attr(torch.float)
    dataset = GraphDataset(x, y)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=0)
    return dataloader
