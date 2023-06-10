# fint-tune vilt
# export MASTER_ADDR='localhost'
# export MASTER_PORT='12366'
# export NODE_RANK=0
# CUDA_VISIBLE_DEVICES=1 python vilt/run.py with num_gpus=1 num_nodes=1 task_finetune_recipe1m_randaug per_gpu_batchsize=4

# export MASTER_ADDR='localhost' 
# export MASTER_PORT='13456'
# export WORLD_SIZE=2
# export NODE_RANK=0
# export LOCAL_RANK=1
CUDA_VISIBLE_DEVICES=0 python vilt/run.py with num_gpus=1 num_nodes=1 task_finetune_recipe1m_randaug per_gpu_batchsize=8