# Reciptor
This repository contains the code to train and evaluate models from the paper:  
Reciptor: An Effective Pretrained Model for Recipe Representation Learning.
# Requirements
```shell
pip install -r requirements.txt
```
# Data
## Data preparation
To prepare training data from scratch, run:
```shell
python prepare_dataset.py
```
Please make sure you have downloaded 
* `data/encs_train_1024.t7`: Skip-instructions train partition
* `data/encs_val_1024.t7`: Skip-instructions val partition
* `data/encs_test_1024.t7`: Skip-instructions test partition

from the original [recipe1M](http://im2recipe.csail.mit.edu/dataset/download/) and put them under their corresponding folders and downloaded
* `data/foodcom_dataset.json` from [foodcom_datatset.json](https://drive.google.com/file/d/10DdbPjDYjcY-uFXGAFv9hBHqYyroFW8p/view?usp=sharing)

## Download preprocessed data
Alternatively, you can download our preprocessed sample data from [reciptor_data](https://drive.google.com/drive/folders/19isPZOMiBA-hA4WoTJbjhX1SmYmgmUZA?usp=sharing) 
and the full data from [zenodo](https://doi.org/10.5281/zenodo.3894040).

# Model
To run the recipe representation model:
```shell
bash run_foodcom_reciptor.sh
```
The model will be saved in `--snapshots` path.

* Notice: to run the baseline models (jm, sjm) described in our paper, please change `--model_type` to `jm|sjm` accordingly.

To store the pretrained recipe embeddings:
```shell
bash run_store_embed.sh
```
# Evaluation
To evaluate the pretrained recipe embeddings on category classification task:
```shell
bash run_evaluation.sh
```
# Credit
The backbone of this framework is based on [torralba-lab/im2recipe-Pytorch](https://github.com/torralba-lab/im2recipe-Pytorch)

The implementation of Set Transformer is based on [TropComplique/set-transformer](https://github.com/TropComplique/set-transformer).

# Reference