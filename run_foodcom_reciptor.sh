#!/bin/bash

python recipe_embedding2.py \
--model_type='reciptor' \
--batch_size=600 \
--epochs=6 \
--valfreq=2 \
--triplet_loss \
--full_data_path='foodcom_sample' \
--snapshots='snapshots_reciptor/'
