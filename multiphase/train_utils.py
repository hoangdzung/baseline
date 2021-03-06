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
try:
    from torch_sparse import SparseTensor
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    from torch_geometric.data import Data
    from torch_geometric.nn import Node2Vec
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except:
    print("Pytorch Geometric is not installed")
    SparseTensor = None 
    maybe_num_nodes = None 
    Data = None 
    Node2Vec = None
    random_walk = None

import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy("file_system")


def n2v(edge_list, node2id, round_id,init_dict=None, embedding_dim=128, walk_length=10,
        context_size=5, walks_per_node=10,tol=1e-4,verbose=False, epochs=100):
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    data = Data(edge_index=edge_index)
    model = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                    context_size=context_size, walks_per_node=walks_per_node, sparse=True)
    if init_dict is not None:
        miss_nodes = []
        X = np.random.randn(len(node2id), embedding_dim)
        for node, idx in node2id.items():
            try:
                X[idx] = init_dict[node]
            except:
                miss_nodes.append(node)
        print("Missing {} nodes: {} ".format(len(miss_nodes), miss_nodes))
        model.embedding.data = torch.tensor(X)

    model = model.to(device)
    loader = model.loader(batch_size=32, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01/(int(round_id)+1))
    best_loss = 10e8
    n_step_without_progress = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        if verbose:
            for pos_rw, neg_rw in tqdm(loader, desc="Train epoch {}".format(epoch+1)):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
        else:
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
        if (best_loss - total_loss)/best_loss < tol:
            n_step_without_progress += 1
            if n_step_without_progress == 3:
                break
        else:
            best_loss = total_loss
            n_step_without_progress = 0
        if verbose:
            print("Epoch {}: loss {} best loss {} #step without progress {}".format(epoch, total_loss, best_loss, n_step_without_progress))

    model.eval()
    out = model().cpu().detach().numpy()

    return out

class RandomWalk():
    def __init__(self, edge_index, walk_length, context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1,
                 num_nodes=None, sparse=False ):
        
        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')
        
        N = maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        self.adj = self.adj.to('cpu')

        assert walk_length >= context_size

        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        
    def loader(self, **kwargs):
        return DataLoader(range(self.adj.sparse_size(0)),
                          collate_fn=self.sample, **kwargs)


    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            for i in range(1,self.context_size):
                walks.append(rw[:, [j,j+i]])    
        return torch.cat(walks, dim=0)


class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n*self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst

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

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

def gnn(edge_list,X, part_id, round_id, ckpt_dir,fanouts = [10,25], use_rw=True, embedding_dim=128, walk_length=10,
        context_size=5, walks_per_node=10, tol=1e-4,verbose=False,epochs=100):

    edge_list = np.array(edge_list)
    G = dgl.graph((edge_list[:,0], edge_list[:,1]), num_nodes=X.shape[0])
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)

    if use_rw:
        rw = RandomWalk(torch.tensor(edge_list.T, dtype=torch.long), walk_length=walk_length, context_size=context_size,walks_per_node=walks_per_node)
        loader = rw.loader(batch_size=32, shuffle=False)
        train_pairs = torch.cat([pos_rw for pos_rw in loader], dim=0)
        G_pair = dgl.graph((train_pairs[:,0], train_pairs[:,1]), num_nodes=X.shape[0])
        # del train_pairs
        print('Train {} edges instead of {} '.format(G_pair.num_edges(), G.num_edges()))
        n_edges = G_pair.num_edges()
        train_seeds = np.arange(n_edges)

        dataloader = dgl.dataloading.EdgeDataLoader(
            G_pair, train_seeds, sampler, 
            g_sampling = G,
            negative_sampler=NegativeSampler(G, 1),
            batch_size=1000,
            shuffle=True,
            drop_last=False)
            # pin_memory=True)
    else:
        print('Train {} edges'.format(G.num_edges()))
        n_edges = G.num_edges()
        train_seeds = np.arange(n_edges)

        dataloader = dgl.dataloading.EdgeDataLoader(
            G, train_seeds, sampler, 
            negative_sampler=NegativeSampler(G, 1),
            batch_size=1000,
            shuffle=True,
            drop_last=False)
            # pin_memory=True)

    nfeat =  torch.tensor(X,dtype=torch.float)
    in_feats = nfeat.shape[1]
    model = SAGE(in_feats, embedding_dim, len(fanouts), F.relu, 0.5)
    model = model.to(device)
    ckpt_file = os.path.join(ckpt_dir, str(part_id+round_id)+"_"+str(round_id-1)+'.pt')
    if os.path.isfile(ckpt_file):
        model.load_state_dict(torch.load(ckpt_file))
    loss_fcn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01/(int(round_id)+1))

    best_loss = 10e8
    n_step_without_progress = 0
    for epoch in range(epochs):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        total_loss = 0

        if verbose:
            for input_nodes, pos_graph, neg_graph, blocks in tqdm(dataloader, desc="Training epoch {}".format(epoch+1)):
                batch_inputs = nfeat[input_nodes].to(device)

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [block.int().to(device) for block in blocks]
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
        #         print(batch_pred.shape, pos_graph, neg_graph)
                loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            for input_nodes, pos_graph, neg_graph, blocks in dataloader:
                batch_inputs = nfeat[input_nodes].to(device)

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [block.int().to(device) for block in blocks]
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
        #         print(batch_pred.shape, pos_graph, neg_graph)
                loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        if best_loss - total_loss < tol:
            n_step_without_progress += 1
            if n_step_without_progress == 3:
                break
        else:
            best_loss = total_loss
            n_step_without_progress = 0
        if verbose:
            print("Epoch {}: loss {} best loss {} #step without progress {}".format(epoch, total_loss, best_loss, n_step_without_progress))
    torch.save(model.state_dict(), os.path.join(ckpt_dir, str(part_id)+"_"+str(round_id)+'.pt'))
    model.eval()
    out = model.inference(G, nfeat, device).detach().numpy()

    return out