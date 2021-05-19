torchbiggraph_train \
    --rank $1 \
  torchbiggraph_configs/${2}_${3}_config.py \
  -p edge_paths=${2}_big_rw_${3}/${2}_big_rw_${3}
if [ $1 -eq 0 ]
then
    torchbiggraph_export_to_tsv \
    torchbiggraph_configs/${2}_${3}_config.py \
    --entities-output /mnt/${2}_big_rw_${3}.tsv \
    --relation-types-output /mnt/${2}_big_rw_${3}-edge.tsv
fi