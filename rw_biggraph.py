import dgl
import numpy as np
import torch as th
import torch
import argparse
import time
import os 
from tqdm import tqdm
from load_graph import *
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torchbiggraph.config import ConfigSchema
from torchbiggraph.converters.import_from_tsv import convert_input_data
from pathlib import Path
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Partition builtin graphs")
    parser.add_argument('--edge_file')
    parser.add_argument('--dataset')
    parser.add_argument('--walk_length', type=int, default=4)
    parser.add_argument('--context_size', type=int, default=2)
    parser.add_argument('--walks_per_node', type=int, default=3)
    parser.add_argument('--num_parts')
    parser.add_argument('--root_output')
    args = parser.parse_args()

    start = time.time()
    edges = np.loadtxt(args.edge_file)
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    rw = RandomWalk(edge_index, walk_length=args.walk_length, context_size=args.context_size,walks_per_node=args.walks_per_node)
    loader = rw.loader(batch_size=32, shuffle=False)
    edge_file_temp = args.edge_file+"_rw{}{}{}".format(args.walks_per_node, args.walk_length, args.context_size)
    with open(edge_file_temp, 'w') as f:
        for pos_rw in tqdm(loader,desc='Write temp file'):
            f.writelines(['{} 1 {}\n'.format(i[0],i[1]) for i in pos_rw.numpy()])

    config_dict = dict(
        entity_path="datatest/pubmed/pubmed_big_10",
        num_epochs=5,
        entities={
            'all': {'num_partitions': 10},
        },
        relations=[{
            'name': 'all_edges',
            'lhs': 'all',
            'rhs': 'all',
            'operator': 'complex_diagonal',
        }],
        dynamic_relations=True,
        edge_paths=['datatest/pubmed/pubmed_big_10/pubmed_big_10'],
        checkpoint_path='model/pubmed_big_10',
        dimension=128,
        global_emb=False,
        comparator='dot',
        loss_fn='softmax',
        lr=0.1,
        num_uniform_negs=50,

        eval_fraction=0,  # to reproduce results, we need to use all training data
        workers=1,
        distributed_init_method="tpc://localhost:30050",
    )
    for num_part in tqdm(args.num_parts.split(","), desc='Run part'):
        num_part = int(num_part)
        datadir = "{}_big_rw_{}".format(args.dataset, num_part)
        config_dict['entity_path'] = os.path.join(args.root_output, datadir)
        config_dict['entities']['all']['num_partitions'] = num_part 
        config_dict['edge_paths'] = [os.path.join(args.root_output, datadir, datadir)]
        config = ConfigSchema.from_dict(config_dict)

        convert_input_data(
            config.entities,
            config.relations,
            config.entity_path,
            config.edge_paths,
            [Path(edge_file_temp)],
            lhs_col=0,
            rhs_col=2,
            rel_col=1,
            dynamic_relations=config.dynamic_relations,
        )



