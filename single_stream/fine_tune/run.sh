# # # fine-tune Oscar
# CUDA_VISIBLE_DEVICES=0,1 python oscar/train.py \
#     --model_name_or_path oscar/pretrained_model/base-vg-labels/ep_107_1192087 \
#     --do_train \
#     --do_lower_case \
#     --evaluate_during_training \
#     --per_gpu_train_batch_size 8 \
#     --learning_rate 0.00005 \
#     --num_train_epochs 5 \
#     --weight_decay 0.05 \
#     --save_steps 1000 \
#     --od_label_type vit \
#     --max_seq_length 70 \
#     --per_gpu_eval_batch_size 32 \
#     --output_dir output/oscar_finetune \
#     --gradient_accumulation_steps 2
    # --fast_run

# fint-tune vilt
# export WORLD_SIZE=2
# export RANK=0  # process rank for the current process
# export MASTER_ADDR=<localhost>
# export MASTER_PORT=<12356>

CUDA_VISIBLE_DEVICES=0 python vilt/run.py with num_gpus=1 num_nodes=1 task_finetune_recipe1m_randaug per_gpu_batchsize=2