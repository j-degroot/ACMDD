# this script trains a PyTorch Geometric GNN with low training abstraction

import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.pool import global_max_pool
from torch_geometric.nn.models import GAT
from gnn_utils.data_loading import list2dataloader

import torch


# defining a GNN class based on a PyTorch neural network module
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        out_channels = 64

        # fixing most arguments to the graph layers (GAT - graph attention layers)
        self.gnn = GAT(in_channels=9, edge_dim=3, hidden_channels=64, num_layers=5, out_channels=out_channels, v2=True,
                       heads=8, jk='cat', norm=torch.nn.BatchNorm1d(num_features=64))

        # final fully connected linear from the embedded molecule
        self.lin = Linear(64, 256, 'relu')
        self.final = Linear(256, 1)

    def forward(self, data):
        # 1. Unpacking data
        node_features = data.x.to(torch.float)
        edge_indices = data.edge_index
        edge_features = data.edge_attr.to(torch.float)

        # 2. Data through graph layers
        x = self.gnn(x=node_features, edge_index=edge_indices, edge_attr=edge_features)

        # 3. Graph pooling
        x = global_max_pool(x, data.batch)

        # 4. Fully connected layers
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.final(x)

        return x

def nn_training_and_validation(name, splits, num_epochs=100):
    """
    Fit a machine learning model on a random train-test split of the data
    and return the performance measures.

    Parameters
    ----------
    name: str
        Name of machine learning algorithm: RF, SVM, ANN
    splits: list
        List of desciptor and label data: train_x, test_x, train_y, test_y.
        train_x
    num_epochs: int
        number of epochs to train the NN
    verbose: bool
        Print performance info (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity, auc on test set.

    """
    train_x, test_x, train_y, test_y = splits

    # define GNN model
    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # change list to torch_geometric.loader.DataLoader
    train_dataloader = list2dataloader(train_x, train_y)
    test_dataloader = list2dataloader(test_x, test_y)

    # training is based on: PyG tutorial 3 - https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
    def train():
        model.train()

        for data, y in train_dataloader:  # Iterate in batches over the training dataset
            # Perform a single forward pass
            out = model(data)
            loss = criterion(out.squeeze(), y)  # Compute the loss
            loss.backward()  # Derive gradients
            optimizer.step()  # Update parameters based on gradients
            optimizer.zero_grad()  # Clear gradients

    def test(dataloader):
        model.eval()

        loss = []
        for data, y in dataloader:  # Iterate in batches over the training/test dataset.
            out = model(data)
            loss.append(criterion(out.squeeze(), y).detach().numpy())  # Check against ground-truth labels.
        return np.mean(loss)  # Derive ratio of correct predictions.

    # Fit the model
    for epoch in range(1, num_epochs):
        train()
        train_mse = test(train_dataloader)
        test_mse = test(test_dataloader)
        print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')

    return model
    # return train_acc[-1], test_acc[-1]
    # Calculate model performance results
    # accuracy, sens, spec, auc = model_performance(ml_model, test_x, test_y, verbose)
    #
    # return accuracy, sens, spec, auc
