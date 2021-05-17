import argparse
from collections import defaultdict
from tqdm import tqdm 
import networkx as nx 
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('in_graph')
parser.add_argument('--out_graph1')
parser.add_argument('--out_graph2')
# parser.add_argument('--n_weight', type=int, default=1)
args = parser.parse_args()

adj_list = defaultdict(set)
n_edges = 0
n_nodes = 0

for line in tqdm(open(args.in_graph), desc='read in graph'):
    node1, node2 = list(map(int, line.strip().split()))
    if node1 == node2:
        continue
    n_edges += 1
    adj_list[node1].add(node2)
    adj_list[node2].add(node1)
n_nodes = len(adj_list)

if args.out_graph1 is not None:
    f1 = open(args.out_graph1, 'w')
    f1.write("{} {} 010 1\n".format(n_nodes, n_edges))
if args.out_graph2 is not None:
    f2 = open(args.out_graph2, 'w')
    f2.write("{} {} 010 2\n".format(n_nodes, n_edges))
    # if args.n_weight == 1:
    #     pattern = "1"
    # elif args.n_weight == 2:
    #     pattern = "1 {}"
    # else:
    #     raise NotImplementedError

for node in tqdm(range(n_nodes), desc="Read node"):
    data = ""
    for neigh in adj_list[node]:
        data += " " + str(neigh + 1)
    data = data.strip() + '\n'
    if args.out_graph1 is not None:
        f1.write("1 " + data)
    if args.out_graph2 is not None:
        f2.write("1 {} ".format(len(adj_list[node])) + data)

if args.out_graph1 is not None:
    f1.close()
if args.out_graph2 is not None:
    f2.close()
