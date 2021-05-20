from tqdm import tqdm
import numpy as np
import torch
import random
from train_utils import gnn, n2v, svd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prefix_file')
parser.add_argument('--part_id')
parser.add_argument('--labels')
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--walk_length', type=int, default=4)
parser.add_argument('--context_size', type=int, default=2)
parser.add_argument('--walks_per_node', type=int, default=2)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_iter', type=int, default=1)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

part_ids = [int(i) for i in args.part_id.split(",")]
assert len(part_ids) == 2

edge_list=set()
node2id={}
for i in range(part_ids[0], part_ids[1]+1):

    for line in tqdm(open(args.prefix_file+'{}'.format(i)), desc='Read part graph'):
        node1, node2  = list(map(float,line.strip().split()))
        node1, node2 = int(node1), int(node2)
        try:
            id1 = node2id[node1]
        except:
            id1 = len(node2id)
            node2id[node1] = id1
        try:
            id2 = node2id[node2]
        except:
            id2 = len(node2id)
            node2id[node2] = id2
        edge_list.add((id1, id2))
   
print(len(node2id))
edge_list = list(edge_list)

n2v(edge_list, embedding_dim=args.dim, walk_length=args.walk_length,
    context_size=args.context_size, walks_per_node=args.walks_per_node, tol=args.tol,verbose=args.verbose, max_iter=args.max_iter)
