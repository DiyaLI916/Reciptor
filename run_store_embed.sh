#!/bin/bash

python recipe_embedding2.py \
--resume='snapshots_jm/saved_model_path.pth' \
--save_tuned_embed \
--model_type='jm' \
--batch_size=300 \
--full_data_path='foodcom_sample'
