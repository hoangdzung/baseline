time python3 ./tools/launch.py \
--workspace ~/workspace/baseline \
--num_trainers 1 \
--num_samplers 12 \
--num_servers 1 \
--part_config $1_$2/$1.json \
--ip_config ip_config.txt \
"python3 train_dist_unsupervised.py --graph_name $1 --ip_config ip_config.txt --num_servers 1 --num_workers 12 --num_epochs 1 --batch_size 2000 --num_gpus -1"