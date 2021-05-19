import dgl
import numpy as np
import torch as th
import argparse
import time
import os 

from load_graph import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogb-product, ogb-paper100M')
    argparser.add_argument('--edgelist', type=str,
                        help='path of edgelist file')
    argparser.add_argument('--features', type=str,
                        help='path of features file')
    argparser.add_argument('--labels', type=str,
                        help='path of labels file')
    argparser.add_argument('--num_parts', nargs='+', type=int,default=[],
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--save_dgl', action='store_true',
                           help='save graph in dgl format.')
    argparser.add_argument('--save_txt', action='store_true',
                           help='save graph in text format.')
    argparser.add_argument('--save_big', action='store_true',
                           help='save graph in biggraph format.')
    argparser.add_argument('--root_output', type=str, default='data',
                           help='Output path of partitioned graph.')
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'reddit':
        g = load_reddit()
    elif args.dataset == 'pubmed':
        g = load_pubmed()
    elif args.dataset == 'corafull':
        g = load_corafull()
    elif args.dataset == 'amzcomputer':
        g = load_amazon_computer()
    elif args.dataset == 'amzphoto':
        g = load_amazon_photo()
    elif args.dataset.startswith("ogbn"):
        g = load_ogb(args.dataset)
    elif args.edgelist is not None:
        g = load_custom(args.edgelist, args.features, args.labels)
    else:
        raise NotImplementedError
    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    if 'train_mask' in g.ndata:
        print('train: {}'.format(th.sum(g.ndata['train_mask'])))
    if 'val_mask' in g.ndata:
        print('valid: {}'.format(th.sum(g.ndata['val_mask'])))
    if 'test_mask' in g.ndata:
        print('test: {}'.format(th.sum(g.ndata['test_mask'])))

    args.root_output = os.path.join(args.root_output, args.dataset)
    if not os.path.isdir(args.root_output):
        os.makedirs(args.root_output)

    if args.save_dgl:
        if args.balance_train and 'train_mask' in g.ndata:
            balance_ntypes = g.ndata['train_mask']
        else:
            balance_ntypes = None

        if args.undirected:
            sym_g = dgl.to_bidirected(g, readonly=True)
            for key in g.ndata:
                sym_g.ndata[key] = g.ndata[key]
            g = sym_g

        for num_part in args.num_parts:
            savedir = os.path.join(args.root_output, args.dataset+"_"+str(num_part))
            dgl.distributed.partition_graph(g, args.dataset, num_part, savedir,
                                            part_method=args.part_method,
                                            balance_ntypes=balance_ntypes,
                                            balance_edges=args.balance_edges)
    if args.save_txt:
        import torch
        import time 
        tic = time.time()
        text_datadir = os.path.join(args.root_output, args.dataset+"_text")
        if not os.path.isdir(text_datadir):
            os.mkdir(text_datadir)

        with open(os.path.join(text_datadir,'labels.txt'), 'w') as f:
            for i, label in enumerate(g.ndata['label'].numpy().tolist()):
                f.write("{} {}\n".format(i, label))
        edges = torch.stack(g.edges()).numpy().T
    
        np.savetxt(os.path.join(text_datadir,'edgelist.txt'), edges, fmt='%d')
        edges_pybig = np.zeros((edges.shape[0],3))
        edges_pybig[:,0] = edges[:,0]
        edges_pybig[:,2]= edges[:,1]
        np.savetxt(os.path.join(text_datadir,'edgelist_pybig.txt'), edges_pybig, fmt='%d')
        try:
            np.savetxt(os.path.join(text_datadir,'features.txt'), g.ndata['features'].numpy(), fmt='%f')
        except:
            pass 

        splits = np.zeros((g.number_of_nodes(),2))
        splits[:,0] = np.arange(splits.shape[0])
        if 'train_mask' in g.ndata:
            splits[g.ndata['train_mask'],1]=1
        if 'val_mask' in g.ndata:
            splits[g.ndata['val_mask'],1]=2
        if 'test_mask' in g.ndata:
            splits[g.ndata['test_mask'],1]=3
        np.savetxt(os.path.join(text_datadir,'splits.txt'), splits.astype(int), fmt='%d')
        print("Writing text takes {} s".format(time.time()-tic))

    if args.save_big:
        from torchbiggraph.config import ConfigSchema
        from torchbiggraph.converters.import_from_tsv import convert_input_data
        from pathlib import Path
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
        for num_part in args.num_parts:
            datadir = "{}_big_{}".format(args.dataset, num_part)
            config_dict['entity_path'] = os.path.join(args.root_output, datadir)
            config_dict['entities']['all']['num_partitions'] = num_part 
            config_dict['edge_paths'] = [os.path.join(args.root_output, datadir, datadir)]
            config = ConfigSchema.from_dict(config_dict)

            convert_input_data(
                config.entities,
                config.relations,
                config.entity_path,
                config.edge_paths,
                [Path(os.path.join(args.root_output,"{}_text/edgelist_pybig.txt".format(args.dataset)))],
                lhs_col=0,
                rhs_col=2,
                rel_col=1,
                dynamic_relations=config.dynamic_relations,
            )



