#!/bin/bash

########################################################################################
# target classification accuracy for reconstructed target point clouds (for reference) #
########################################################################################
python run_classifier.py --data_type clean
wait
python evaluate_classifier.py --data_type target --attack_folder output_space_attack
wait

#############################################################################
# target classification accuracy for reconstructed adversarial point clouds #
#############################################################################
# output space attack
python run_classifier.py --data_type adversarial --attack_folder output_space_attack
wait
python evaluate_classifier.py --data_type adversarial --attack_folder output_space_attack
wait

# latent space attack
python run_classifier.py --data_type adversarial --attack_folder latent_space_attack
wait
python evaluate_classifier.py --data_type adversarial --attack_folder latent_space_attack
wait

########################################################################################
# source classification accuracy for reconstructed source point clouds (for reference) #
########################################################################################
python evaluate_classifier.py --data_type source --attack_folder output_space_attack
wait

#########################################################################################################
# source classification accuracy for reconstructed adversarial point clouds ("before defense" accuracy) #
#########################################################################################################
# output space attack
python run_classifier.py --data_type before_defense --attack_folder output_space_attack
wait
python evaluate_classifier.py --data_type before_defense --attack_folder output_space_attack
wait

# latent space attack
python run_classifier.py --data_type before_defense --attack_folder latent_space_attack
wait
python evaluate_classifier.py --data_type before_defense --attack_folder latent_space_attack
wait

#####################################################################################################
# source classification accuracy for reconstructed defended point clouds ("after defense" accuracy) #
#####################################################################################################
# critical points defense against output space attack
python run_classifier.py --data_type after_defense --attack_folder output_space_attack --defense_folder defense_critical_res
wait
python evaluate_classifier.py --data_type after_defense --attack_folder output_space_attack --defense_folder defense_critical_res
wait

# off-surface defense against output space attack
python run_classifier.py --data_type after_defense --attack_folder output_space_attack --defense_folder defense_surface_res
wait
python evaluate_classifier.py --data_type after_defense --attack_folder output_space_attack --defense_folder defense_surface_res
wait

# critical points defense against latent space attack
python run_classifier.py --data_type after_defense --attack_folder latent_space_attack --defense_folder defense_critical_res
wait
python evaluate_classifier.py --data_type after_defense --attack_folder latent_space_attack --defense_folder defense_critical_res
wait

# off-surface defense against latent space attack
python run_classifier.py --data_type after_defense --attack_folder latent_space_attack --defense_folder defense_surface_res
wait
python evaluate_classifier.py --data_type after_defense --attack_folder latent_space_attack --defense_folder defense_surface_res
wait
