# splitting the data into train, validation and test

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
from datasetGen import createDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx


def createDataLoader():
    dataset,x_list,y_list = createDataset()
    x = torch.from_numpy(x_list)
    n_features = len(x)

    X_train, X_test, y_train, y_test = train_test_split(pd.Series(x_list),
                                                        pd.Series(y_list),
                                                        test_size=0.30,
                                                        random_state=42)


    train_mask = torch.zeros(n_features, dtype=torch.bool)
    test_mask = torch.zeros(n_features, dtype=torch.bool)
    train_mask[X_train.index] = True
    test_mask[X_test.index] = True
    dataset.train_mask= train_mask
    dataset.test_mask = test_mask



    # Create batches with neighbor sampling
    data_loader = NeighborLoader(
        dataset,
        num_neighbors=[5, 10],
        batch_size=2,
        input_nodes=dataset.train_mask,
    )

    NUM_VAL = len(X_test)
    NUM_TRAIN = len(X_train)

    return data_loader, NUM_VAL, NUM_TRAIN
