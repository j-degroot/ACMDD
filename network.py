import torch
import os
import pytorch_lightning as pl
from torch import nn, tensor, concat
from torch.nn import functional as F, Linear, BatchNorm1d, ModuleList, ReLU, Sequential
from torch_geometric.nn.glob import GlobalAttention, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN, PNA, GAT, DimeNet

class GNN_GAT(pl.LightningModule):
    def __init__(self, config, data_dir=None, name='GNN'):
        super(GNN_GAT, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        # self.dropout_rate = config['dropout_rate']
        # self.second_input = config['second_input']

        self.gnn = GAT(in_channels=config['N'], edge_dim=config['E'], hidden_channels=hidden_size, v2=True,
                       num_layers=config['n_layers'], heads=8, jk='cat', norm=torch.nn.BatchNorm1d(hidden_size))


        # self.ppol = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size, 1))
        self.pool = global_mean_pool

        self.fc1 = Linear(hidden_size, hidden_size)

        self.fc2 = Linear(hidden_size, 1)

        self.save_hyperparameters()
        self.emb_f = None

    def forward(self, graphs: Data):
        batch = graphs.batch
        x = graphs.x.to(torch.float)
        edge_attr = graphs.edge_attr.to(torch.float)
        edge_index = graphs.edge_index
        x = F.relu(self.gnn(x=x, edge_index=edge_index, edge_attr=edge_attr,))

        self.emb_f = self.pool(x, batch)
        x = F.relu(self.fc1(self.emb_f))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)
        return x

    def training_step(self, train_batch, batch_idx):
        prediction = self.forward(train_batch)
        loss = F.mse_loss(prediction, train_batch.y)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        prediction = self.forward(val_batch)
        loss = F.mse_loss(prediction, val_batch.y)
        self.log('val_loss', loss, batch_size=self.batch_size)
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        prediction = self.forward(test_batch)
        loss = F.mse_loss(prediction, test_batch.y)
        # r_squared =
        self.log('test_loss', loss, batch_size=self.batch_size)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_results = {'test_loss': avg_loss}
        return self.test_results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_bottleneck(self):
        return self.emb_f
