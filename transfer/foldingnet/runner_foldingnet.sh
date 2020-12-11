#!/bin/bash

# train FoldingNet for attack transfer
python train_foldingnet.py --outf log/foldingnet_for_transfer \
    --training_set log/autoencoder_victim/eval_train/point_clouds_train_set_13l.npy \
    --validation_set log/autoencoder_victim/eval_val/point_clouds_val_set_13l.npy
wait
