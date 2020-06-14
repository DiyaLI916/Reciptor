#!/bin/bash

python train_recipe_embedding.py \
--resume='snapshots/saved_model_path.pth' \
--save_tuned_embed \
--model_type='reciptor' \
--batch_size=300 \
--full_data_path='foodcom_sample'
