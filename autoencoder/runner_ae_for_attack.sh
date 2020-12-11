#!/bin/bash

# train an autoencoder for attack
python train_ae.py --train_folder log/autoencoder_victim
wait

# prepare data for attack
python tst_ae.py --train_folder log/autoencoder_victim
wait
