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


def createDataLoader(dataset,numRecords,train_mask_df,batchSize):



    # maskDf  = pd.DataFrame(np.random.randn(numRecords))
    #
    # train_mask = np.random.rand(len(maskDf)) <= trainPercent
    train_mask = np.array(list(train_mask_df['mask_val']))
    dataset.train_mask= train_mask
    dataset.test_mask = ~train_mask

    # Create batches with neighbor sampling
    data_loader = NeighborLoader(
        dataset,
        num_neighbors=[1000, 1000,1000,1000],
        batch_size=batchSize,
        input_nodes=dataset.train_mask,
    )

    #ydf = pd.Series(y_list)
    counter = 0
    for i in train_mask:
        if i == False:
            counter = counter + 1

    NUM_VAL = counter
    NUM_TRAIN = numRecords - counter

    return data_loader, NUM_VAL, NUM_TRAIN
