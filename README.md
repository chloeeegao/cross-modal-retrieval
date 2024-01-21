### Cross-Modal Food Retrieval

This is my master thesis project. Written report can be found here: https://theses.liacs.nl/pdf/2022-2023-GaoYaqiong.pdf 
Recipe1M dataset was used in this work.

---
This repository is organized as follows:

* data

  * Data was downloaded from *http://im2recipe.csail.mit.edu/dataset/download/*
  * The contents of the directory `DATASET_PATH` should be the following and `train/, val/, and test/ `must contain the image files for each split after uncompressing.
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

  * This is a replication of [H-Transformer](https://github.com/amzn/image-to-recipe-transformers)
* single_stream

  * fine_tune: two VLP models ([Oscar](https://github.com/microsoft/Oscar) and [ViLT](https://github.com/dandelin/ViLT)) are fine-tuned on Recipe1M
  * recipevl:  recipe vision-language (RecipeVL)  model is trained from scratch

#### Data Preparation

run scripts under preprocessing

* run `python bigrams.py --create` will save all bigrams to disk in the corpus of all recipe titles in the training set, sorted by frequency.
* run `python bigrams.py --no_create` will create class labels from food101 categories and top bigrams; then classes1M.pkl file will be created and used later
* run `python preprocessing.py --root DATASET_PATH `will create a folder `/traindata` which contains data for training

#### Training/Evaluation

Under each folder there is a `run.sh` script including training parameters, run `sh run.sh` in terminal window to train and evaluate the model.
