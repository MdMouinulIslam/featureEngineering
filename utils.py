import numpy as np
def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)


def predict(model, data_loader, NUM_VAL, NUM_TRAIN):
    model.eval()
    totalLoss = 0

    krange = [5, 10, 20, 50, 100, 500]
    for data in data_loader:
        out = model(data)
        loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
        totalLoss = totalLoss + loss.item()

        predictedOut = out[data.test_mask].view(1, -1).detach().numpy()
        trueOut = data.y[data.test_mask].view(1, -1).detach().numpy()

        print(trueOut.dtype)
        print(trueOut.shape)
        for k in krange:
            print("ndcg at ", k, " is =", ndcg_score(trueOut, predictedOut, k=k))

        # for i in range(0,len(trueOut)):
        #     t = (i,predictedOut[i],trueOut[i])
        #     rankTuple.append(t)

    print("loss = ", totalLoss / NUM_VAL)

    # rankTuple.sort(key=lambda x:x[1])

    # predictedRank = []
    # for i,o1,o2 in rankTuple:
    #     predictedRank.append(i)
    # rankTuple.sort(key=lambda x:x[2])
    #
    # trueRank = []
    # for i, o1, o2 in rankTuple:
    #     trueRank.append(i)




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