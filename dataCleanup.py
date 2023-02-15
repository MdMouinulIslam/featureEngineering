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


import uci_dataset as dataset

def getData():
    data = dataset.load_audiology()
    data_g = data[data["age_gt_60"] == "t"]
    data_l = data[data["age_gt_60"] == "f"]
    data_g = data_g.apply(LabelEncoder().fit_transform).dropna()
    data_l = data_l.apply(LabelEncoder().fit_transform).dropna()
    features_g = data_g.columns[:-1]
    classLabel_g = data_g.columns[-1:][0]
    features_l = data_l.columns[:-1]
    classLabel_l = data_l.columns[-1:][0]
    edge_rows = []
    edge_colNames = ["i", "j", "mi"]
    out_rows = []
    out_colNames = ["f", "mi"]
    node_rows = []
    node_colNames = ["f", "mi"]

    for f1 in features_g:
        for f2 in features_g:
            mi = normalized_mutual_info_score(data_g[f1], data_g[f2])
            if f1 == f2:
                continue
            f1g = f1 + "g"
            f2g = f2 + "g"
            t1 = (f1g, f2g, mi)
            t2 = (f2g, f1g, mi)
            edge_rows.append(t1)
            edge_rows.append(t2)

        miClass = normalized_mutual_info_score(data_g[f1], data_g[classLabel_g])
        f1g = f1 + "g"
        t_out = (f1g, miClass)
        out_rows.append(t_out)

    for f1 in features_l:
        for f2 in features_l:
            if f1 == f2:
                continue
            mi = normalized_mutual_info_score(data_l[f1], data_l[f2])
            f1l = f1 + "l"
            f2l = f2 + "l"
            t1 = (f1l, f2l, mi)
            t2 = (f2l, f1l, mi)
            edge_rows.append(t1)
            edge_rows.append(t2)

        miClass = normalized_mutual_info_score(data_l[f1], data_l[classLabel_l])
        f1l = f1 + "l"
        t_out = (f1l, miClass)
        out_rows.append(t_out)

    for f1 in features_l:
        mi = 1
        f1l = f1 + "l"
        f1g = f1 + "g"
        t1 = (f1l, f1g, mi)
        t2 = (f1g, f1l, mi)
        edge_rows.append(t1)
        edge_rows.append(t2)

    for f in features_g:
        fl = f + "l"
        fg = f + "g"
        rl = (fl, 1)
        rg = (fg, 1)
        node_rows.append(rl)
        node_rows.append(rg)

    edge_df = pd.DataFrame(edge_rows,
                           columns=edge_colNames)
    edge_df.to_csv("data/edge.csv")
    out_df = pd.DataFrame(out_rows,
                          columns=out_colNames)
    out_df.to_csv("data/out.csv")
    node_df = pd.DataFrame(node_rows,
                           columns=node_colNames)
    node_df.to_csv("data/node.csv")

    return edge_df,out_df,node_df

