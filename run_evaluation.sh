#!/bin/bash

python test_embed.py \
--batch_size=300 \
--epochs=4 \
--valfreq=2 \
--full_data_path='foodcom_sample' \
--pretrained_embed_path='tuned_embed/saved_tuned_embedding_path.pkl' \
--snapshots='snapshots_test/' \
--lr=0.01
