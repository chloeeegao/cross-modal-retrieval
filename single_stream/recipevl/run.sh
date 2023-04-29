#!/usr/bin/env bash

# evaluate recipevl
# CUDA_VISIBLE_DEVICES=1 python main.py \
# --do_eval \
# --model_name vit_small_v1 \
# --vit vit_small_patch16_224 \
# --resume_from checkpoint-29-347710 \
# --eval_sample 10000 \
# --eval_times 5

# train recipevl
# CUDA_VISIBLE_DEVICES=0,1 python main.py \
# --do_train \
# --model_name vit_base_v3 \
# --vit vit_base_patch16_224 \
# --evaluate_during_training \
# --per_gpu_train_batch_size 6 \
# --gradient_accumulation_steps 2 \
# --max_seq_len 100 \
# --extract_info \




