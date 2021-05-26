import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm 
import numpy as np 

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

from torch_sparse import SparseTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

import os 
import argparse
import pickle 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy("file_system")

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler,
                batch_size=1000,
                shuffle=True,
                drop_last=False,
                num_workers=1)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].to(device)

                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.detach().cpu()

            x = y
        return y

parser = argparse.ArgumentParser()
parser.add_argument('--edgelist')
parser.add_argument('--feats')
parser.add_argument('--ckpt')
parser.add_argument('--out')
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--fanouts', default="5,10")
args = parser.parse_args()

edge_list=set()
for line in tqdm(open(args.edgelist), desc='Read graph'):
    node1, node2  = list(map(int,line.strip().split()))
    edge_list.add((node1, node2))

X =  np.loadtxt(args.feats)
edge_list = np.array(list(edge_list))
G = dgl.graph((edge_list[:,0], edge_list[:,1]), num_nodes=X.shape[0])

nfeat =  torch.tensor(X,dtype=torch.float)
in_feats = nfeat.shape[1]
fanouts=list(map(int, args.fanouts.split(',')))
model = SAGE(in_feats, args.dim, len(fanouts), F.relu, 0.5)
model = model.to(device)
if os.path.isfile(args.ckpt):
    model.load_state_dict(torch.load(args.ckpt))

model.eval()
out = model.inference(G, nfeat, device).detach().numpy()
emb_dicts = {}
for i in range(out.shape[1]):
    emb_dicts[i] = out[i]
pickle.dump(emb_dicts, open(args.out, 'wb'))
