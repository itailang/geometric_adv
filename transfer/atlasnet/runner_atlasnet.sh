#!/bin/bash

# train AtlasNet for attack transfer
python train.py --dir_name log/atlasnet_for_transfer \
    --mode train --nb_primitives 25 --template_type SQUARE --custom_data --no_metro \
    --train_pc_path log/autoencoder_victim/eval_train/point_clouds_train_set_13l.npy \
    --eval_pc_path log/autoencoder_victim/eval_val/point_clouds_val_set_13l.npy
wait
