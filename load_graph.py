import dgl
import torch as th
import torch
import numpy as np 

def load_pubmed():
    from dgl.data import PubmedGraphDataset
    data = PubmedGraphDataset()
    g = data[0]
    g.ndata['labels'] = g.ndata.pop('label')
    return g

def load_corafull():
    from dgl.data import CoraFullDataset
    data = CoraFullDataset()
    g = data[0]
    g.ndata['labels'] = g.ndata.pop('label')
    return g

def  load_amazon_computer():
    from  dgl.data import AmazonCoBuyComputerDataset
    data = AmazonCoBuyComputerDataset()
    g = data[0]
    g.ndata['labels'] = g.ndata.pop('label')
    return g

def  load_amazon_photo():
    from  dgl.data import AmazonCoBuyPhotoDataset
    data = AmazonCoBuyPhotoDataset()
    g = data[0]
    g.ndata['labels'] = g.ndata.pop('label')
    return g

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset()
    g = data[0]
    g.ndata['labels'] = g.ndata.pop('label')
    return g

def load_ogb(name):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['labels'] = labels
    in_feats = graph.ndata['feat'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph

def load_custom(edgelist, features, labels):
    edge_list = np.loadtxt(edgelist).astype(int)
    g = dgl.graph((edge_list[:,0], edge_list[:,1]))
    g.ndata['feat'] = torch.tensor(np.loadtxt(features)).float()
    g.ndata['labels'] = torch.tensor(np.loadtxt(labels))
    return g 

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
