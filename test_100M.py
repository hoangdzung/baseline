from tqdm import tqdm
import numpy as np
import random
import argparse
from gensim.models import Word2Vec
# from fastnode2vec import Graph, Node2Vec
import time 

parser = argparse.ArgumentParser()
parser.add_argument('--edge_list')
=parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--walk_length', type=int, default=4)
parser.add_argument('--context_size', type=int, default=2)
parser.add_argument('--walks_per_node', type=int, default=2)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_iter', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--prob', type=float, default=0.5)

args = parser.parse_args()

part_ids = [int(i) for i in args.part_id.split(",")]
assert len(part_ids) == 2

walks = []
# full_walks = []
for line in tqdm(open(args.edge_list), desc='Read part graph'):
    node1, node2  = list(map(float,line.strip().split()))
    node1, node2 = str(int(node1)), str(int(node2))

    if random.random() < args.prob:
        walks.append([node1, node2])
    # full_walks.append([node1, node2])

stime = time.time()
model = Word2Vec(sentences=walks, size=args.dim, window=args.context_size, min_count=0, workers=args.num_workers, iter=args.max_iter)
print("Gensim take ", time.time()-stime)

# stime = time.time()
# graph = Graph(full_walks, directed=False, weighted=False)
              
# model2 = Node2Vec(graph, dim=128, walk_length=args.walk_length, context=args.context_size, workers=args.num_workers)
# model2.train(epochs=args.max_iter)
# print("FastNode2vec take ", time.time()-stime)
