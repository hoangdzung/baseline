#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.


def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="ogbn-arxiv_big_5/",
        edge_paths=[
            "ogbn-arxiv_big_5/ogbn-arxiv_big_5/",
            # "data/FB15k/freebase_mtr100_mte100-valid_partitioned",
            # "data/FB15k/freebase_mtr100_mte100-test_partitioned",
        ],
        checkpoint_path="model/ogbn-arxiv_5",
        # Graph structure
        entities={"all": {"num_partitions": 5}},
        relations=[
            {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": "complex_diagonal",
            }
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=128,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=5,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        # regularization_coef=1e-3,
        distributed_init_method="tpc://172.31.23.206:30050",
        # Evaluation during training
        eval_fraction=0,  # to reproduce results, we need to use all training data
    )

    return config
