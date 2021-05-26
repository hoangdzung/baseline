time torchbiggraph_train \
    --rank $3 \
  torchbiggraph_configs/${1}_${2}_config.py \
  -p edge_paths=${1}_big_rw_${2}/${1}_big_rw_${2}
if [ $3 -eq 0 ]
then
    time torchbiggraph_export_to_tsv \
    torchbiggraph_configs/${1}_${2}_config.py \
    --entities-output /mnt/${1}_big_rw_${2}.tsv \
    --relation-types-output /mnt/${1}_big_rw_${2}-edge.tsv
    rm -r model
    rm /mnt/${1}_big_rw_${2}.tsv
    rm /mnt/${1}_big_rw_${2}-edge.tsv
fi