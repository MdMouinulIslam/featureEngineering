


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score,adjusted_mutual_info_score
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import random
from sklearn.model_selection import train_test_split
from dataloader import createDataLoader
from gnnModel import FEModel


def train(hyperparams):
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_epochs = hyperparams['n_epochs']
    save_loss_interval = hyperparams['save_loss_interval']
    print_interval = hyperparams['print_interval']
    save_model_interval = hyperparams['save_model_interval']

    model = FEModel()
    data_loader, NUM_VAL, NUM_TRAIN = createDataLoader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss_train = 0
        epoch_loss_val = 0
        model.train()
        for data in data_loader:
            # print(data)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            epoch_loss_train += loss.item()
            loss.backward()
            optimizer.step()

            val_loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
            epoch_loss_val += val_loss.item()

        if epoch % print_interval == 0:
            print("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}".format(epoch, epoch_loss_train / NUM_TRAIN,
                                                                                epoch_loss_val / NUM_VAL))