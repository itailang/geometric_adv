#!/bin/bash

#########################################################################
# attack transfer for same AE architecture and different initialization #
#########################################################################
# output space attack
python run_transfer.py --transfer_ae_folder log/autoencoder_for_transfer --transfer_ae_restore_epoch 500 --transfer_ae_type PointNet \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder output_space_attack --output_folder_name output_space_attack_transfer
wait

python evaluate_transfer.py --transfer_ae_folder log/autoencoder_for_transfer \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder output_space_attack --output_folder_name output_space_attack_transfer
wait

# latent space attack
python run_transfer.py --transfer_ae_folder log/autoencoder_for_transfer --transfer_ae_restore_epoch 500 --transfer_ae_type PointNet \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder latent_space_attack --output_folder_name latent_space_attack_transfer
wait

python evaluate_transfer.py --transfer_ae_folder log/autoencoder_for_transfer \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder latent_space_attack --output_folder_name latent_space_attack_transfer
wait

############################################################
# attack transfer for different AE architecture (AtlasNet) #
############################################################
# output space attack
python run_transfer.py --transfer_ae_folder log/atlasnet_for_transfer --transfer_ae_type AtlasNet \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder output_space_attack --output_folder_name output_space_attack_transfer
wait

python evaluate_transfer.py --transfer_ae_folder log/atlasnet_for_transfer \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder output_space_attack --output_folder_name output_space_attack_transfer
wait

# latent space attack
python run_transfer.py --transfer_ae_folder log/atlasnet_for_transfer --transfer_ae_type AtlasNet \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder latent_space_attack --output_folder_name latent_space_attack_transfer
wait

python evaluate_transfer.py --transfer_ae_folder log/atlasnet_for_transfer \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder latent_space_attack --output_folder_name latent_space_attack_transfer
wait

##############################################################
# attack transfer for different AE architecture (FoldingNet) #
##############################################################
# output space attack
python run_transfer.py --transfer_ae_folder log/foldingnet_for_transfer --transfer_ae_restore_epoch 24 --transfer_ae_type FoldingNet \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder output_space_attack --output_folder_name output_space_attack_transfer
wait

python evaluate_transfer.py --transfer_ae_folder log/foldingnet_for_transfer \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder output_space_attack --output_folder_name output_space_attack_transfer
wait

# latent space attack
python run_transfer.py --transfer_ae_folder log/foldingnet_for_transfer --transfer_ae_restore_epoch 24 --transfer_ae_type FoldingNet \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder latent_space_attack --output_folder_name latent_space_attack_transfer
wait

python evaluate_transfer.py --transfer_ae_folder log/foldingnet_for_transfer \
    --ae_folder log/autoencoder_victim --attack_pc_idx log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy \
    --attack_folder latent_space_attack --output_folder_name latent_space_attack_transfer
wait
