

from dataCleanup import getData
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


def createDataset():
    edge_df,out_df,node_df = getData()

    edges = pd.read_csv("data/edge.csv")

    # generate map from iso_code to ids of form [0, ..., num_unique_iso_codes - 1]
    features = set(edges["i"])
    features = features.union(set(edges["j"]))
    features_to_id = {code : i for (i, code) in enumerate(features)}

    edges['i_id'] = edges['i'].map(features_to_id)
    edges['j_id'] = edges['j'].map(features_to_id)
    edge_index = torch.from_numpy(edges[['i_id', 'j_id']].to_numpy(np.int64)).t()


    EDGE_FEATURES = ["mi"]
    edge_attr = torch.from_numpy(edges[EDGE_FEATURES].to_numpy(np.float32)) #extract the features from the dataset.
    edge_attr = (edge_attr - edge_attr.mean(axis=0)) / (edge_attr.std(axis=0))

    # load in target values
    y_df = pd.read_csv("data/out.csv ")
    y_df['id'] = y_df['f'].map(features_to_id)

    y_list = y_df.sort_values('id')["mi"].to_numpy(np.float32)

    y = torch.from_numpy(y_list).unsqueeze(1)

    # load in input features
    x_df = pd.read_csv("data/node.csv")
    x_df['id'] = x_df['f'].map(features_to_id)
    x_df.sort_values("id")

    x_list = x_df.sort_values("id").loc[:,"id"].to_numpy(np.int64)
    x = torch.from_numpy(x_list)

    dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return dataset,x_list,y_list






