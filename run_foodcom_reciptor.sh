#!/bin/bash

python train_recipe_embedding.py \
--model_type='reciptor' \
--batch_size=600 \
--epochs=200 \
--valfreq=2 \
--triplet_loss \
--full_data_path='foodcom_sample' \
--snapshots='snapshots_reciptor/'
