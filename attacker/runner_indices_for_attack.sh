#!/bin/bash
############################################################################################################
# NOTE: run this script with the command: bash runner_indices_for_attack.sh, to run the for loop correctly #
############################################################################################################

# random source indices
python prepare_indices_for_attack.py --ae_folder log/autoencoder_victim --get_rand_idx 1
wait

# nearest neighbors matrix for target candidates selection
for i in {0..4378..100}
do
    python prepare_indices_for_attack.py --ae_folder log/autoencoder_victim --get_chamfer_nn_idx 1 --pc_start_idx $i --pc_batch_size 100
    wait
done
