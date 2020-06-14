# Reciptor
This repository contains the code to train and evaluate models from the paper:  
Reciptor: An Effective Pretrained Model for Recipe Representation Learning.
# Requirements
```shell
pip install -r requirements.txt
```
# Data
To prepare training data:
```shell
python prepare_dataset.py
```
# Model
To run the recipe representation model:
```shell
bash run_foodcom_reciptor.sh
```
To store the pretrained recipe embeddings:
```shell
bash run_store_embed.sh
```
# Evaluation
To evaluate the pretrained recipe embeddings:
```shell
bash run_evaluation.sh
```
# Credit
The backbone of this framework is based on [torralba-lab/im2recipe-Pytorch](https://github.com/torralba-lab/im2recipe-Pytorch)

The implementation of Set Transforme is based on [jadore801120/attention-is-all-you-need-pytorch](https://github.com/TropComplique/set-transformer).

# Reference