#!/usr/bin/env bash

# evaluate recipevl
CUDA_VISIBLE_DEVICES=1 python main.py \
--do_eval \
--model_name vit_base_v9 \
--vit vit_base_patch16_224 \
--load_path checkpoint-3-38337 \
--eval_sample 1000 \
--eval_times 5 \
--max_seq_len 100

# # train recipevl
# CUDA_VISIBLE_DEVICES=0,1 python main.py \
# --do_train \
# --model_name vit_base_v6 \
# --vit vit_base_patch32_224 \
# --patch_size 32 \
# --max_seq_len 100 \
# --evaluate_during_training \
# --per_gpu_train_batch_size 8 \
# --gradient_accumulation_steps 1 \
# --n_epochs 30 \
# --extract_info \
# --add_tags \

# fine_tune recipevl
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --do_fine_tune \
# --model_name vit_base_v9 \
# --vit vit_base_patch16_224 \
# --evaluate_during_training \
# --per_gpu_train_batch_size 4 \
# --gradient_accumulation_steps 2 \
# --n_epochs 5 \
# --load_path checkpoint-28-671468 \
# --learning_rate 0.00002 \
# --max_seq_len 100 \
# --extract_info \
# --add_tags 