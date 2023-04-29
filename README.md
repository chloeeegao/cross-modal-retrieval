Cross-Modal Food Retrieval

This is the master thesis under LIACS at Leiden University.  Recipe1M dataset was used in this work. Single-stream and dual-stream models for cross-modal representation were explored and experimented on the cross-modal retrieval task.

---

This repository is organized as follows:

* data

  * dataset can be downloaded from *http://im2recipe.csail.mit.edu/dataset/download/*
  * the contents of the directory `DATASET_PATH` should be the following and `train/, val/, and test/ `must contain the image files for each split after uncompressing.
  * ```
    layer1.json
    layer2.json
    vocab.txt
    train/
    val/
    test/
    ```
* preprocessing
* dual_stream

  * this is a modification of [H-Transformer](https://github.com/amzn/image-to-recipe-transformers)
* single_stream

  * fine_tune: two VLP models ([Oscar](https://github.com/microsoft/Oscar) and [ViLT](https://github.com/dandelin/ViLT)) were fine-tuned on the cross-modal retrieval task
  * pretrain: one single-stream model was trained from scratch on the cross-modal retrieval task

---

### Installation

### Data Preparation

run scripts under preprocessing

* run `python bigrams.py --create` will save all bigrams to disk in the corpus of all recipe titles in the training set, sorted bu frequency.
* run `python bigrams.py --no_create` will create class labels from food101 categories and top bigrams; then classes1M.pkl file will be created and will be used later
* run `python preprocessing.py --root DATASET_PATH `will create a folder `/traindata` which contains data for training

### Training

```
CUDA_VISIBLE_DEVICES=0,1 python single_stream/pretrain/main.py \
--do_train \
--model_name vit_base_v1 \
--vit vit_base_patch16_224 \
--evaluate_during_training \
--per_gpu_train_batch_size 6
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=0,1 python single_stream/pretrain/main.py \
--do_eval \
--model_name vit_small_v1 \
--vit vit_small_patch16_224 \
--resume_from checkpoint-29-347710 \
--eval_sample 1000 \
--eval_times 5
```

### Pretrained Models
