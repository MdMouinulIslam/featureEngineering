# Basically the same as the baseline except we pass edge features
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
from torch.nn.functional import normalize

NUM_EDGE_FEATURES = 1
EMB_SIZE = 16


class FEModel(torch.nn.Module):
    def __init__(self, num_features=EMB_SIZE, hidden_size=32, target_size=1, num_emb = 1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features+1
        self.target_size = target_size
        self.convs = [GATConv(self.num_features, self.hidden_size, edge_dim=NUM_EDGE_FEATURES),
                      GATConv(self.hidden_size, self.hidden_size, edge_dim=NUM_EDGE_FEATURES)]
        # ****self.hidden_size/2*****
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_emb, embedding_dim=EMB_SIZE)
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x_org, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x1 =  [i[0].long() for i in x_org]
        x2 =  [i[1] for i in x_org]

        x1 = torch.tensor(x1)

        x1 = self.item_embedding(x1)
        #x1 = x1.squeeze(1)
        #x1  = normalize(x1,p=2,dim=0)
        x2 = torch.tensor((x2))
        #x2 = normalize(x2,p=2,dim=0)
        x = torch.cat((x1,x2),1)
        #x = x1
        # print(edge_index)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)  # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)  # edge features here as well
        x = self.linear(x)
        return x