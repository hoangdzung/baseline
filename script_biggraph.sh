torchbiggraph_train \
    --rank $i \
  torchbiggraph_configs/arxiv_5_config.py \
  -p edge_paths=ogbn-arxiv_big_5/ogbn-arxiv_big_5
if [ $1 -eq 0 ]
then
    torchbiggraph_export_to_tsv \
    torchbiggraph_configs/arxiv_5_config.py \
    --entities-output /mnt/ogbn-arxiv_big_5.tsv \
    --relation-types-output /mnt/ogbn-arxiv_big_5-edge.tsv
fi