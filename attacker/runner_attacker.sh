#!/bin/bash

#######################
# output space attack #
#######################
# run the attack
python run_attack.py --loss_dist_type chamfer --loss_adv_type chamfer --dist_weight_list 0.5 1.0 5.0 --num_pc_for_attack 25 \
    --output_folder_name output_space_attack  \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

# prepare additional data for evaluation
python get_dists_per_point.py --output_folder_name output_space_attack \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

# evaluate the attack
python evaluate_attack.py --output_folder_name output_space_attack \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

#######################
# latent space attack #
#######################
# run the attack
python run_attack.py --loss_dist_type chamfer --loss_adv_type latent --dist_weight_list 50.0 100.0 150.0 --num_pc_for_attack 25 \
    --output_folder_name latent_space_attack  \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

# prepare additional data for evaluation
python get_dists_per_point.py --output_folder_name latent_space_attack \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

# evaluate the attack
python evaluate_attack.py --output_folder_name latent_space_attack \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait
