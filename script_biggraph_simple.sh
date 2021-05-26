time torchbiggraph_train \
    --rank $3 \
  torchbiggraph_configs/${1}_${2}_config.py \
  -p edge_paths=${1}_big_rw_${2}/${1}_big_rw_${2}