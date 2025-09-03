export MASTER_ADDR=127.0.0.1  
export MASTER_PORT=29500    

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py  --cfg-path configs/config.yaml


