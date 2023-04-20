


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
from sklearn.metrics import ndcg_score
from utils import ranking_precision_score


def train(data_loader, NUM_VAL, NUM_TRAIN,hyperparams):
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_epochs = hyperparams['n_epochs']
    save_loss_interval = hyperparams['save_loss_interval']
    print_interval = hyperparams['print_interval']
    save_model_interval = hyperparams['save_model_interval']

    model = FEModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss_train = 0
        model.train()
        for data in data_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])

            epoch_loss_train += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch: {} Train loss: {:.2e}".format(epoch,epoch_loss_train/NUM_TRAIN))
    return model


def predict(model,data_loader, NUM_VAL, NUM_TRAIN):
    model.eval()
    totalLoss = 0


    krange = [5,10,20,50,100]
    for data in data_loader:
        out = model(data)
        loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
        totalLoss = totalLoss + loss.item()

        predictedOut = out[data.test_mask].view(1,-1).detach().numpy()
        trueOut = data.y[data.test_mask].view(1,-1).detach().numpy()

        print(trueOut.dtype)
        print(trueOut.shape)
        for k in krange:
            print("ndcg at ",k ," is =", ndcg_score(trueOut,predictedOut,k=k))

    print("loss = ",totalLoss/NUM_VAL)


