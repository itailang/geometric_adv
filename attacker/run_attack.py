"""
Created on June 6th, 2020
@author: itailang
"""

# import system modules
import os
import os.path as osp
import sys
import argparse
import numpy as np

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from src.autoencoder import Configuration as Conf
from src.adv_ae import AdvAE
from src.adversary_utils import load_data, prepare_data_for_attack
from src.in_out import create_dir
from src.general_utils import plot_3d_point_cloud
from src.tf_utils import reset_tf_graph

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the attack [default: 0.01]')
parser.add_argument('--loss_dist_type', type=str, default='chamfer', help='Type of distance regularization loss [default: chamfer]')
parser.add_argument('--loss_adv_type', type=str, default='chamfer', help='Type of adversarial loss [default: chamfer]')
parser.add_argument('--dist_weight_list', nargs='+', default=[0.5, 1, 5], help='List of possible weights for distance regularization loss')
parser.add_argument('--max_point_pert_weight', type=float, default=0.0, help='Weight for maximal point perturbation loss [default: [0.0]]')
parser.add_argument('--max_point_dist_weight', type=float, default=0.0, help='Weight for maximal point nearest neighbor distance loss [default: [0.0]]')
parser.add_argument('--num_iterations', type=int, default=500, help='Number of iterations per dist_weight [default: 500]')
parser.add_argument('--num_iterations_thresh', type=int, default=400, help='Number of iterations threshold for cheking the update of output data [default: 400]')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for attack [default: 10]')
parser.add_argument('--ae_folder', type=str, default='log/autoencoder_victim', help='Folder for loading a trained autoencoder model [default: log/autoencoder_victim]')
parser.add_argument('--restore_epoch', type=int, default=500, help='Restore epoch of a trained autoencoder [default: 500]')
parser.add_argument("--attack_pc_idx", type=str, default='log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy', help="List of indices of point clouds for the attack")
parser.add_argument("--target_pc_idx_type", type=str, default='chamfer_nn_complete', help="Type of target pc index (Chamfer or latent nearest neighbors) [default: chamfer_nn_complete]")
parser.add_argument("--num_pc_for_attack", type=int, default=25, help='Number of point clouds for attack (per shape class) [default: 25]')
parser.add_argument("--num_pc_for_target", type=int, default=5, help='Number of candidate point clouds for target (per point cloud for attack) [default: 5]')
parser.add_argument("--correct_pred_only", type=int, default=0, help='1: Use targets with corret predicted label only, 0: do not restrict targets according to predicted label [default: 0]')
parser.add_argument("--output_folder_name", type=str, default='attack_res', help="Output folder name")
flags = parser.parse_args()

print('Run attack flags:', flags)

assert flags.loss_dist_type in ['pert', 'chamfer'], 'wrong loss_dist_type: %s' % flags.loss_dist_type
assert flags.loss_adv_type in ['latent', 'chamfer'], 'wrong loss_adv_type: %s' % flags.loss_adv_type
assert flags.num_iterations_thresh <= flags.num_iterations, 'num_iterations_thresh (%d) should be smaller or equal to num_iterations (%d)' % (flags.num_iterations_thresh, flags.num_iterations)
assert flags.target_pc_idx_type in ['latent_nn', 'chamfer_nn_complete'], 'wrong target_pc_idx_type: %s' % flags.target_pc_idx_type

# define basic parameters
top_out_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))  # Use to save Neural-Net check-points etc.
data_path = osp.join(top_out_dir, flags.ae_folder, 'eval')
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

output_path = create_dir(osp.join(data_path, flags.output_folder_name))

# load data
point_clouds, latent_vectors, pc_classes, slice_idx, ae_loss = \
    load_data(data_path, files, ['point_clouds_test_set', 'latent_vectors_test_set', 'pc_classes', 'slice_idx_test_set', 'ae_loss_test_set'])

assert np.all(ae_loss > 0), 'Note: not all autoencoder loss values are larger than 0 as they should!'

nn_idx_dict = {'latent_nn': 'latent_nn_idx_test_set', 'chamfer_nn_complete': 'chamfer_nn_idx_complete_test_set'}
nn_idx = load_data(data_path, files, [nn_idx_dict[flags.target_pc_idx_type]])

correct_pred = None
if flags.correct_pred_only:
    pc_labels, pc_pred_labels = load_data(data_path, files, ['pc_label_test_set', 'pc_pred_labels_test_set'])
    correct_pred = (pc_labels == pc_pred_labels)

# load indices for attack
attack_pc_idx = np.load(osp.join(top_out_dir, flags.attack_pc_idx))
attack_pc_idx = attack_pc_idx[:, :flags.num_pc_for_attack]

# load autoencoder configuration
ae_dir = osp.join(top_out_dir, flags.ae_folder)
conf = Conf.load(osp.join(ae_dir, 'configuration'))

# update autoencoder configuration
conf.ae_dir = ae_dir
conf.ae_name = 'autoencoder'
conf.ae_restore_epoch = flags.restore_epoch
conf.encoder_args['return_layer_before_symmetry'] = False
conf.encoder_args['b_norm_decay'] = 1.          # for avoiding the update of batch normalization moving_mean and moving_variance parameters
conf.decoder_args['b_norm_decay'] = 1.          # for avoiding the update of batch normalization moving_mean and moving_variance parameters
conf.decoder_args['b_norm_decay_finish'] = 1.   # for avoiding the update of batch normalization moving_mean and moving_variance parameters

# attack configuration
conf.experiment_name = 'adversary'
conf.batch_size = flags.batch_size
conf.learning_rate = flags.learning_rate
conf.loss_dist_type = flags.loss_dist_type
conf.loss_adv_type = flags.loss_adv_type
conf.dist_weight_list = [float(w) for w in flags.dist_weight_list]
conf.max_point_pert_weight = flags.max_point_pert_weight
conf.max_point_dist_weight = flags.max_point_dist_weight
conf.target_pc_idx_type = flags.target_pc_idx_type
conf.num_pc_for_attack = flags.num_pc_for_attack
conf.num_pc_for_target = flags.num_pc_for_target
conf.correct_pred_only = flags.correct_pred_only
conf.num_iterations = flags.num_iterations
conf.num_iterations_thresh = flags.num_iterations_thresh
conf.train_dir = output_path

conf.save(osp.join(conf.train_dir, 'attack_configuration'))

classes_for_attack = conf.class_names
classes_for_target = conf.class_names

# Attack the AE model
for i in range(len(pc_classes)):
    pc_class_name = pc_classes[i]
    if pc_class_name not in classes_for_attack:
        continue

    # Build Adversary and AE model
    reset_tf_graph()
    ae = AdvAE(conf.experiment_name, conf)

    save_dir = create_dir(osp.join(conf.train_dir, pc_class_name))

    # prepare data for attack
    source_pc, target_pc = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, point_clouds, slice_idx, attack_pc_idx, flags.num_pc_for_target, flags.target_pc_idx_type, nn_idx, correct_pred)
    _, target_latent = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, latent_vectors, slice_idx, attack_pc_idx, flags.num_pc_for_target, flags.target_pc_idx_type, nn_idx, correct_pred)
    _, target_ae_loss_ref = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, ae_loss, slice_idx, attack_pc_idx, flags.num_pc_for_target, flags.target_pc_idx_type, nn_idx, correct_pred)

    target_ae_loss_ref = target_ae_loss_ref.reshape(-1)

    buf_size = 1  # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(save_dir, 'attack_stats.txt'), 'a', buf_size)
    fout.write('Train flags: %s\n' % flags)
    adversarial_metrics, adversarial_pc_input, adversarial_pc_recon =\
        ae.attack(source_pc, target_latent, target_pc, target_ae_loss_ref, conf, log_file=fout)
    fout.close()

    # save results
    np.save(osp.join(save_dir, 'adversarial_metrics'), adversarial_metrics)
    np.save(osp.join(save_dir, 'adversarial_pc_input'), adversarial_pc_input)
    np.save(osp.join(save_dir, 'adversarial_pc_recon'), adversarial_pc_recon)
    np.save(osp.join(save_dir, 'dist_weight'), np.array(conf.dist_weight_list))
