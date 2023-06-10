# # fine-tune Oscar
# CUDA_VISIBLE_DEVICES=1 python oscar/train.py \
#     --model_name_or_path output/oscar_finetune_irtr/version_2/checkpoint-9-37260 \
#     --do_train \
#     --do_lower_case \
#     --evaluate_during_training \
#     --per_gpu_train_batch_size 16 \
#     --learning_rate 0.00002 \
#     --num_train_epochs 10 \
#     --weight_decay 0.05 \
#     --save_steps 5000 \
#     --od_label_type vit \
#     --max_seq_length 70 \
#     --output_dir output/oscar_finetune_irtr/version_2 \
#     --gradient_accumulation_steps 1 \
    # --fast_run

# evaluate oscar
CUDA_VISIBLE_DEVICES=1 python oscar/train.py \
    --do_eval \
    --do_test \
    --eval_model_dir output/oscar_finetune_irtr_v2/checkpoint-9-49670
