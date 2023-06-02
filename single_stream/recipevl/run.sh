#!/usr/bin/env bash

# evaluate recipevl
CUDA_VISIBLE_DEVICES=0 python main.py \
--do_eval \
--model_name vit_base_v10 \
--vit vit_base_patch16_224 \
--load_path checkpoint-29-347710 \
--eval_sample 1000 \
--eval_times 5 \
--max_seq_len 300 



# # train recipevl
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --do_train \
# --model_name vit_base_v9 \
# --vit vit_base_patch16_224 \
# --max_seq_len 100 \
# --evaluate_during_training \
# --per_gpu_train_batch_size 16 \
# --gradient_accumulation_steps 1 \
# --n_epochs 30 \
# --extract_info \
# --add_tags \
# --resume_from checkpoint-28-671468 \
# --learning_rate 0.00000741

