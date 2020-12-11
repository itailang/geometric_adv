#!/bin/bash

#######################################
# defense against output space attack #
#######################################
# critical points defense
python run_defense_critical.py --attack_folder output_space_attack \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

python evaluate_defense.py --attack_folder output_space_attack --use_adversarial_data 1 \
    --output_folder_name defense_critical_res \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

# off-surface defense
python get_knn_dists_per_point.py --attack_folder output_space_attack \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

python run_defense_surface.py --attack_folder output_space_attack \
    --num_knn_for_defense 2 --knn_dist_thresh 0.04 \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

python evaluate_defense.py --attack_folder output_space_attack --use_adversarial_data 1 \
    --output_folder_name defense_surface_res \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

#######################################
# defense against latent space attack #
#######################################
# critical points defense
python run_defense_critical.py --attack_folder latent_space_attack \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

python evaluate_defense.py --attack_folder latent_space_attack --use_adversarial_data 1 \
    --output_folder_name defense_critical_res \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

# off-surface defense
python get_knn_dists_per_point.py --attack_folder latent_space_attack \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

python run_defense_surface.py --attack_folder latent_space_attack \
    --num_knn_for_defense 2 --knn_dist_thresh 0.04 \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait

python evaluate_defense.py --attack_folder latent_space_attack --use_adversarial_data 1 \
    --output_folder_name defense_surface_res \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy
wait
