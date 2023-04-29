#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python \
h_transformer/train.py \
--model_name htrans_vit_v2 \
--save_dir model \
--batch_size 256 \
--root /data/s2478846/data \
--date _04_26_

#eval
# CUDA_VISIBLE_DEVICES=1 python h_transformer/test.py \
# --model_name htrans_vit_v1 \
# --eval_split test \
# --root /data/s2478846/data \
# --save_dir model