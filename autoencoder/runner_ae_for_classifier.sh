#!/bin/bash

# prepare data for semantic interpretation
python tst_ae.py --train_folder log/autoencoder_victim --set_type train_set --output_folder_name eval_train
wait
python tst_ae.py --train_folder log/autoencoder_victim --set_type val_set --output_folder_name eval_val
wait
