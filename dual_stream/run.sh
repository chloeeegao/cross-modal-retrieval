#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=1,2,3,4,6 python \
h_transformer/train.py \
--model_name htrans_vit_v3 \
--save_dir model \
--batch_size 128 \
--root /data/s2478846/data \
--date _05_04_

#eval + produce test embed
# CUDA_VISIBLE_DEVICES=0 python h_transformer/test.py \
# --model_name htrans_vit_v2 \
# --eval_split test \
# --root /data/s2478846/data \
# --save_dir model

# calculate metrics
# CUDA_VISIBLE_DEVICES=0 python h_transformer/eval.py --embeddings_file model/htrans_vit_v2/feats_test.pkl --medr_N 10000 
# --retrieval_mode recipe2image