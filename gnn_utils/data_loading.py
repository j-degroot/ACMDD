import torch

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader


class GraphDataset(Dataset):
    """
    This in memory dataset is useful to define the datalaoder
    """

    def __init__(self, x: [Data], y: [Data]):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> (Data, float):
        return self.x[idx], self.y[idx]


def list2dataloader(x: [Data], y: [Data]) -> DataLoader:
    """
    This function builds a dataloader by first changing all the node and edge features into float torch tensors. Then
    makes a dataset to get each element within and to check the length. From this, a dataloader is made which is part of
    training any neural network.
    """
    for elem in x:
        elem.x = elem.x.to(torch.float)
        elem.edge_attr = elem.edge_attr.to(torch.float)
    dataset = GraphDataset(x, y)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=0)
    return dataloader
