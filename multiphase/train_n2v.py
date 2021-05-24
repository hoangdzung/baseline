from tqdm import tqdm
import numpy as np
import torch
import random
from train_utils import gnn, n2v
import argparse
import os 
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--prefix_file')
parser.add_argument('--part_id')
parser.add_argument('--emb_dir')
parser.add_argument('--init_emb')
parser.add_argument('--join')
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--walk_length', type=int, default=4)
parser.add_argument('--context_size', type=int, default=2)
parser.add_argument('--walks_per_node', type=int, default=2)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--round_id')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

def project(x, y, common_nodes = None ):
    anchors_model = x
    project_model = y

    common_nodes_ = set(anchors_model).intersection(project_model)
    n_common_nodes = int(len(common_nodes_))
    common_nodes_ = list(common_nodes_)[:n_common_nodes]
    if common_nodes is None:
        common_nodes = common_nodes_
    else:
        common_nodes = set(common_nodes).intersection(common_nodes_)

    if len(common_nodes) == 0:
        new_model = y.copy()
        new_model.update(x)
        return new_model

    anchors_emb = np.stack([anchors_model[i] for i in common_nodes])
#     anchors_emb = anchors_emb/np.sqrt(np.sum(anchors_emb**2,axis=1,keepdims=True))

    tobechanged_emb = np.stack([project_model[i] for i in common_nodes])
#     tobechanged_emb = tobechanged_emb/np.sqrt(np.sum(tobechanged_emb**2,axis=1,keepdims=True))

    trans_matrix, c, _,_ = np.linalg.lstsq(tobechanged_emb, anchors_emb, rcond=-1)
    tobechanged_emb = np.stack([project_model[i] for i in project_model])
#     tobechanged_emb = tobechanged_emb/np.sqrt(np.sum(tobechanged_emb**2,axis=1,keepdims=True))

    new_embeddings = np.matmul(tobechanged_emb, trans_matrix)

    new_model = dict(zip(project_model.keys(), new_embeddings))
    new_model.update(anchors_model)

    return new_model

assert args.method in ['gnn','n2v'], "Only support gnn n2v"
assert args.join in ['project','rand']

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if not os.path.isdir(args.emb_dir):
    os.makedirs(args.emb_dir)

if args.init_emb is not None and os.path.isfile(args.init_emb):
    init_embs = pickle.load(open(args.init_emb, 'rb'))
else:
    init_embs = None 

part_ids = [int(i) for i in args.part_id.split(",")]
assert len(part_ids) == 2

edge_list_list = []
node2id_list = []
for i in range(part_ids[0], part_ids[1]+1):
    edge_list=set()
    node2id={}
    for line in tqdm(open(args.prefix_file+'{}.txt'.format(i)), desc='Read part graph'):
        node1, node2  = list(map(int,line.strip().split()))
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
    edge_list_list.append(edge_list)
    node2id_list.append(node2id)

emb_dicts = []
for part_id, (edge_list, node2id) in enumerate(zip(edge_list_list, node2id_list)):
    out = n2v(edge_list,node2id,
        init_dict=init_embs,
        embedding_dim=args.dim, 
        walk_length=args.walk_length,
        context_size=args.context_size, 
        walks_per_node=args.walks_per_node, 
        tol=args.tol,
        verbose=args.verbose, 
        epochs=args.epochs)

    emb_dict = {}
    for node, i in node2id.items():
        emb_dict[node] = out[i]
    emb_dicts.append(emb_dict)

corenodes=set(emb_dicts[0]).intersection(emb_dicts[1])

if 'project' in args.join:
    final_emb_merge = emb_dicts[0]
    for i in range(1,part_ids[1]-part_ids[0]+1):
        final_emb_merge = project(final_emb_merge,emb_dicts[i],a1)
    pickle.dump(final_emb_merge, open(os.path.join(args.emb_dir, 'project'+ args.round_id+'.pkl'), 'wb'))
if 'rand' in args.join:
    final_emb_merge = emb_dicts[0]
    for i in range(1,part_ids[1]-part_ids[0]+1):
        final_emb_merge.update(emb_dicts[i])
    pickle.dump(final_emb_merge, open(os.path.join(args.emb_dir, 'rand'+ args.round_id+'.pkl'), 'wb'))
