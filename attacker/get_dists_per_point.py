"""
Created on August 10th, 2020
@author: itailang
"""

# import system modules
import os
import os.path as osp
import sys
import argparse
import numpy as np
import tensorflow as tf

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from src.autoencoder import Configuration as Conf
from src.adversary_utils import load_data, prepare_data_for_attack
from src.in_out import create_dir
from src.general_utils import plot_3d_point_cloud
from src.tf_utils import reset_tf_graph
from external.structural_losses.tf_nndistance import nn_distance

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ae_folder', type=str, default='log/autoencoder_victim', help='Folder for loading a trained autoencoder model [default: log/autoencoder_victim]')
parser.add_argument("--attack_pc_idx", type=str, default='log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy', help="List of indices of point clouds for the attack")
parser.add_argument("--do_sanity_checks", type=int, default=0, help="1: Do sanity checks, 0: Do not do sanity checks [default: 0]")
parser.add_argument("--output_folder_name", type=str, default='attack_res', help="Output folder name")
flags = parser.parse_args()

print('Get dists flags:', flags)

# define basic parameters
top_out_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))  # Use to save Neural-Net check-points etc.
data_path = osp.join(top_out_dir, flags.ae_folder, 'eval')
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

output_path = create_dir(osp.join(data_path, flags.output_folder_name))

chamfer_batch_size = 10

# load attack configuration
conf = Conf.load(osp.join(output_path, 'attack_configuration'))

# load data
point_clouds, pc_classes, slice_idx = \
    load_data(data_path, files, ['point_clouds_test_set', 'pc_classes', 'slice_idx_test_set'])

num_points = point_clouds.shape[1]

nn_idx_dict = {'latent_nn': 'latent_nn_idx_test_set', 'chamfer_nn_complete': 'chamfer_nn_idx_complete_test_set'}
nn_idx = load_data(data_path, files, [nn_idx_dict[conf.target_pc_idx_type]])

correct_pred = None
if conf.correct_pred_only:
    pc_labels, pc_pred_labels = load_data(data_path, files, ['pc_label_test_set', 'pc_pred_labels_test_set'])
    correct_pred = (pc_labels == pc_pred_labels)

# load indices for attack
attack_pc_idx = np.load(osp.join(top_out_dir, flags.attack_pc_idx))
attack_pc_idx = attack_pc_idx[:, :conf.num_pc_for_attack]

classes_for_attack = conf.class_names
classes_for_target = conf.class_names

# build chamfer graph
reset_tf_graph()
adv_pc_pl = tf.placeholder(tf.float32, shape=[None, num_points, 3])
inp_pc_pl = tf.placeholder(tf.float32, shape=[None, num_points, 3])
dists_first_to_second, _, dists_second_to_first, _ = nn_distance(adv_pc_pl, inp_pc_pl)
chamfer_dist = tf.reduce_mean(dists_first_to_second, axis=1) + tf.reduce_mean(dists_second_to_first, axis=1)

sess = tf.Session('')

# compute distance per point
for i in range(len(pc_classes)):
    pc_class_name = pc_classes[i]
    if pc_class_name not in classes_for_attack:
        continue

    # prepare data for attack
    source_pc, _ = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, point_clouds, slice_idx, attack_pc_idx, conf.num_pc_for_target, conf.target_pc_idx_type, nn_idx, correct_pred)

    # load data
    load_dir = osp.join(output_path, pc_class_name)
    # adversarial metrics: loss_adv, loss_dist, source_chamfer_dist, target_nre, target_recon_error
    adversarial_metrics = np.load(osp.join(load_dir, 'adversarial_metrics.npy'))
    adversarial_pc_input = np.load(osp.join(load_dir, 'adversarial_pc_input.npy'))

    source_chamfer_dist = adversarial_metrics[:, :, 2]

    show = False
    if show:
        plot_idx = 0
        plot_3d_point_cloud(source_pc[plot_idx])
        plot_3d_point_cloud(adversarial_pc_input[-1, plot_idx])

    num_dist_weight, num_examples_curr, _, _ = adversarial_pc_input.shape

    adversarial_pc_input_dists = -1 * np.ones(adversarial_pc_input.shape[:3], dtype=np.float32)
    for j in range(num_dist_weight):
        for k in range(0, num_examples_curr, chamfer_batch_size):
            adv_pc_batch = adversarial_pc_input[j, k:k + chamfer_batch_size]
            inp_pc_batch = source_pc[k:k + chamfer_batch_size]

            feed_dict = {adv_pc_pl: adv_pc_batch, inp_pc_pl: inp_pc_batch}
            dists_first_to_second_batch, dist_batch = sess.run([dists_first_to_second, chamfer_dist], feed_dict=feed_dict)

            # sanity check
            if flags.do_sanity_checks:
                assert np.array_equal(dist_batch, source_chamfer_dist[j, k:k + chamfer_batch_size]), 'mismatch for chamfer dist'

            adversarial_pc_input_dists[j, k:k + chamfer_batch_size] = dists_first_to_second_batch

    assert np.all(adversarial_pc_input_dists >= 0), 'The adversarial_pc_input_dists was not filled correctly'

    # the distances from nn_distance() function are squared. take a square root of them before saving
    adversarial_pc_input_dists = np.sqrt(adversarial_pc_input_dists)

    # save distances per point
    save_dir = load_dir
    np.save(osp.join(save_dir, 'adversarial_pc_input_dists'), adversarial_pc_input_dists)
