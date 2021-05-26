#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import os 

def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="youtube_big_rw_8/",
        edge_paths=[
            "youtube_big_8/youtube_big_rw_8/",
            # "data/FB18k/freebase_mtr100_mte100-valid_partitioned",
            # "data/FB18k/freebase_mtr100_mte100-test_partitioned",
        ],
        checkpoint_path="model/youtube_8",
        # Graph structure
        entities={"all": {"num_partitions": 8}},
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
        num_epochs=int(os.environ['EPOCHS']),
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        # regularization_coef=1e-3,
        distributed_init_method="tcp://{}:30050".format(os.environ['MAINIP']),
        # Evaluation during training
        eval_fraction=0,  # to reproduce results, we need to use all training data
        num_machines = 8
    )

    return config
