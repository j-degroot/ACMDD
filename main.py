import numpy as np
from sklearn import metrics
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GAT
from data_loading import dfs2dataloader

import tqdm
import torch
import os
import pandas


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.gnn = GAT(in_channels=9, edge_dim=3, hidden_channels=64, num_layers=5, out_channels=64, v2=True,
                       jk='last', norm=torch.nn.BatchNorm1d(64))
        self.lin = Linear(256, 1)

    def forward(self, data):
        x = data.x.to(torch.float)
        edge_attr = data.edge_attr.to(torch.float)

        x = self.gnn(x=x, edge_index=data.edge_index, edge_attr=edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)

        return x

def nn_training_and_validation(name, splits, num_epochs=40, verbose=True):
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

    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = torch.nn.MSELoss()

    # change pd.DataFrame to torch_geometric.loader.DataLoader
    train_dataloader = dfs2dataloader(train_x, train_y)
    test_dataloader = dfs2dataloader(test_x, test_y)
    # print(train_dataloader)
    # print(test_dataloader)

    # training is from: PyG tutorial 3 (I believe - check still)
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
        train_acc = test(train_dataloader)
        test_acc = test(test_dataloader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # return train_acc[-1], test_acc[-1]
    # Calculate model performance results
    # accuracy, sens, spec, auc = model_performance(ml_model, test_x, test_y, verbose)
    #
    # return accuracy, sens, spec, auc
