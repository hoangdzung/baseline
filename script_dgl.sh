time python3 ./tools/launch.py \
--workspace ~/workspace/baseline \
--num_trainers 1 \
--num_servers 1 \
--part_config $1_$2/$1.json \
--ip_config ip_config.txt \
"python3 train_dist_unsupervised.py --graph_name $1 --ip_config ip_config.txt --num_servers 1 --num_epochs 5 --batch_size 1000 --num_gpus -1 --out_npz $1_dgl_$2"