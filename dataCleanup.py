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
import pickle


import uci_dataset as dataset
import itertools

# def validFeature(f):
#     if f == "age_gt_60" | f == "speech":
#         return False
#     return True





def cleanData(data,dataname,numOfIntent):


    # allComb = ['f','t']
    # allIntentData = {}



    #numOfIntent = NUM_INTENT

    # intentRanges = []
    # for i in range(1,numOfIntent):
    #     x = data[first_column].quantile(i*1/numOfIntent)
    #     intentRanges.append(x)
    #
    # median = data[first_column].median()
    # for v in allComb:
    #     if v == 'f':
    #         intentData = data[(data[first_column] >= median)]
    #     else:
    #         intentData = data[(data[first_column] < median)]
    #     intentName = v
    #     allIntentData[intentName] = intentData


    columns = data.columns
    first_column = columns[0]


    intentRanges = []
    for i in range(0, numOfIntent + 1):
        #print(i * 1 / (numOfIntent))
        x = data[first_column].quantile(i * 1 / (numOfIntent))
        intentRanges.append(x)

    allIntentData = {}
    for i in range(0,numOfIntent-1):
        tempIntentData = data[data[first_column] > intentRanges[i]]
        intentData =  tempIntentData[tempIntentData[first_column] > intentRanges[i+1]]
        intentName = 'intent' + str(i+1)
        allIntentData[intentName] = intentData




    features = data.columns[:-1]
    classLabel = data.columns[-1:][0]


    features = list(features)
    features.remove(first_column)
    #features.remove("speech")

    edge_rows = []
    edge_colNames = ["i", "j", "mi"]
    out_rows = []
    out_colNames = ["f", "mi"]
    node_rows = []
    node_colNames = ["f"]
    node_colNames.extend(features)

    train_mask_column = ['f', 'mask_val']

    # calculate node featues
    fname = r'data/'+ dataname  + str(len(features))+ '.pickle'

    nodeAtt = {}
    # with open(fname, 'rb') as handle:
    #     nodeAtt = pickle.load(handle)

    if len(list(nodeAtt.keys())) == 0:
        for f1 in features:
            attValue = []
            for f2 in features:
                mi = normalized_mutual_info_score(data[f1], data[f2])
                attValue.append(mi)
            nodeAtt[f1] = attValue


    with open(fname, 'wb') as handle:
        pickle.dump(nodeAtt, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # only mask first intent features
    train_mask = []
    maskIntent = list(allIntentData.keys())[0]

    for intentName, intentData in allIntentData.items():
        for f1 in features:
            for f2 in features:
                if f1 == f2:
                    continue
                mi = normalized_mutual_info_score(intentData[f1], intentData[f2])
                f1name = f1 + intentName
                f2name = f2 + intentName
                t1 = (f1name, f2name, mi)
                t2 = (f2name, f1name, mi)
                edge_rows.append(t1)
                edge_rows.append(t2)


            miClass  = normalized_mutual_info_score(intentData[f1], intentData[classLabel])

            f1name = f1 + intentName
            out = (f1name, miClass)
            out_rows.append(out)

            node = [f1name]
            node.extend(nodeAtt[f1])
            node = tuple(node)

            node_rows.append(node)

            if maskIntent == intentName:
                train_mask.append((f1name,False))
            else:
                train_mask.append((f1name,True))




    train_mask_df = pd.DataFrame(train_mask,columns=train_mask_column)


    for f in features:
        mi = 1
        for intentName1 in allIntentData.keys():
            for intentName2 in allIntentData.keys():
                if intentName1 == intentName2:
                    continue
                f1 = f + intentName1
                f2 = f + intentName2
                t1 = (f1, f2, mi)
                t2 = (f2, f1, mi)
                edge_rows.append(t1)
                edge_rows.append(t2)

    edge_df = pd.DataFrame(edge_rows,
                           columns=edge_colNames)

    # filter out mi values less than some filter
    edge_df = edge_df[edge_df["mi"] > 0.1]


    edgefile = r"data/"+dataname+"_edge.csv"
    edge_df.to_csv(edgefile)

    out_df = pd.DataFrame(out_rows,
                          columns=out_colNames)
    outfile = r"data/"+dataname+"_out.csv"
    out_df.to_csv(outfile)

    node_df = pd.DataFrame(node_rows,
                           columns=node_colNames)
    nodefile = r"data/"+dataname+"_node.csv"
    node_df.to_csv(nodefile)

    maskfile = r"data/"+dataname+"_mask.csv"
    train_mask_df.to_csv(maskfile)

    return edge_df, out_df, node_df, train_mask_df





def cleanData1(data,dataname):

    ValDic ={"f":0,"t":1,"normal":0, "good":1, "very_good":2}
    allComb = ['f','t']
    allIntentData = {}

    for v1 in allComb:

        intentData = data[(data['age_gt_60'] == ValDic[v1])]
        intentName = v1
        allIntentData[intentName] = intentData


    features = data.columns[:-1]
    classLabel = data.columns[-1:][0]


    features = list(features)
    features.remove("age_gt_60")
    #features.remove("speech")

    edge_rows = []
    edge_colNames = ["i", "j", "mi"]
    out_rows = []
    out_colNames = ["f", "mi"]
    node_rows = []
    node_colNames = ["f"]
    node_colNames.extend(features)

    # calculate node featues
    nodeAtt = {}
    for f1 in features:
        attValue = []
        for f2 in features:
            mi = normalized_mutual_info_score(data[f1], data[f2])
            attValue.append(mi)
        nodeAtt[f1] = attValue



    for intentName, intentData in allIntentData.items():
        for f1 in features:
            for f2 in features:
                if f1 == f2:
                    continue
                mi = normalized_mutual_info_score(intentData[f1], intentData[f2])
                f1name = f1 + intentName
                f2name = f2 + intentName
                t1 = (f1name, f2name, mi)
                t2 = (f2name, f1name, mi)
                edge_rows.append(t1)
                edge_rows.append(t2)


            miClass  = normalized_mutual_info_score(intentData[f1], intentData[classLabel])

            f1name = f1 + intentName
            out = (f1name, miClass)
            out_rows.append(out)

            node = [f1name]
            node.extend(nodeAtt[f1])
            node = tuple(node)

            node_rows.append(node)



    for f in features:
        mi = 1
        for intentName1 in allIntentData.keys():
            for intentName2 in allIntentData.keys():
                f1 = f + intentName1
                f2 = f + intentName2
                t1 = (f1, f2, mi)
                t2 = (f2, f1, mi)
                edge_rows.append(t1)
                edge_rows.append(t2)

    edge_df = pd.DataFrame(edge_rows,
                           columns=edge_colNames)

    # filter out mi values less than some filter
    edge_df = edge_df[edge_df["mi"] > 0.1]


    edgefile = r"data/"+dataname+"_edge.csv"
    edge_df.to_csv(edgefile)

    out_df = pd.DataFrame(out_rows,
                          columns=out_colNames)
    outfile = r"data/"+dataname+"_out.csv"
    out_df.to_csv(outfile)

    node_df = pd.DataFrame(node_rows,
                           columns=node_colNames)
    nodefile = r"data/"+dataname+"_node.csv"
    node_df.to_csv(nodefile)

    return edge_df, out_df, node_df
