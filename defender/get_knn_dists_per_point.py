"""
Created on September 4th, 2020
@author: itailang
"""

# import system modules
import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from src.autoencoder import Configuration as Conf
from src.adversary_utils import load_data, prepare_data_for_attack, get_quantity_at_index
from src.in_out import create_dir
from src.general_utils import iterate_in_chunks, get_dist_mat
from src.tf_utils import reset_tf_graph
from external.grouping.tf_grouping import knn_point, group_point

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ae_folder', type=str, default='log/autoencoder_victim', help='Folder for loading a trained autoencoder model [default: log/autoencoder_victim]')
parser.add_argument("--attack_pc_idx", type=str, default='log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy', help="List of indices of point clouds for the attack")
parser.add_argument('--attack_folder', type=str, default='attack_res', help='Folder for loading attack data')
parser.add_argument("--num_knn", type=int, default=8, help='Number of nearest neighbors to compute distance to [default: 8]')
parser.add_argument("--use_tf_knn", type=int, default=1, help='Use tensorflow to compute nearest neighbors to compute distance, 0: use numpy [default: 1]')
parser.add_argument("--output_folder_name", type=str, default='defense_surface_res', help="Output folder name")
flags = parser.parse_args()

print('Get knn dists flags:', flags)

# define basic parameters
top_out_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))  # Use to save Neural-Net check-points etc.
data_path = osp.join(top_out_dir, flags.ae_folder, 'eval')
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

attack_dir = osp.join(top_out_dir, flags.ae_folder, 'eval', flags.attack_folder)
output_path = create_dir(osp.join(attack_dir, flags.output_folder_name))
output_path_orig = create_dir(osp.join(attack_dir, flags.output_folder_name + '_orig'))

# load data
point_clouds, pc_classes, slice_idx = \
    load_data(data_path, files, ['point_clouds_test_set', 'pc_classes', 'slice_idx_test_set'])

num_points = point_clouds.shape[1]

# load attack configuration
conf = Conf.load(osp.join(attack_dir, 'attack_configuration'))

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

# build knn dist graph
knn_batch_size = 100
if flags.use_tf_knn:
    reset_tf_graph()
    pc_pl = tf.placeholder(tf.float32, shape=[None, num_points, 3])
    _, knn_idx = knn_point(flags.num_knn + 1, pc_pl, pc_pl)  # find nearest flags.num_knn neighbors for each point
    grouped_points = group_point(pc_pl, knn_idx[:, :, 1:])  # group neighbors for each point
    deltas = grouped_points - tf.tile(tf.expand_dims(pc_pl, 2), [1, 1, flags.num_knn, 1])  # remove centers to get deltas
    knn_dists = tf.sqrt(tf.reduce_sum(deltas ** 2, axis=3, keepdims=False))  # compute distance to each neighbors

    sess = tf.Session('')

for i in range(len(pc_classes)):
    pc_class_name = pc_classes[i]
    if pc_class_name not in classes_for_attack:
        continue

    print('compute knn dists for shape class %s (%d out of %d classes) ' % (pc_class_name, i + 1, len(pc_classes)))
    start_time = time.time()

    # prepare data for attack
    source_pc, _ = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, point_clouds, slice_idx, attack_pc_idx, conf.num_pc_for_target, nn_idx, correct_pred)

    # load data
    load_dir = osp.join(attack_dir, pc_class_name)
    adversarial_pc_input = np.load(osp.join(load_dir, 'adversarial_pc_input.npy'))

    # take adversarial point clouds of selected dist weight per attack
    source_target_norm_min_idx = np.load(osp.join(load_dir, 'analysis_results', 'source_target_norm_min_idx.npy'))
    adversarial_pc_input = get_quantity_at_index([adversarial_pc_input], source_target_norm_min_idx)

    # add axis to keep the interface of dist_weight as the first dim
    adversarial_pc_input = np.expand_dims(adversarial_pc_input, axis=0)

    num_dist_weight, num_examples_curr, _, _ = adversarial_pc_input.shape

    # get knn distances per point for adversarial point clouds
    knn_dists_adversarial_pc_input = -1 * np.ones(list(adversarial_pc_input.shape[:-1]) + [flags.num_knn], dtype=np.float32)
    for j in range(num_dist_weight):
        if flags.use_tf_knn:
            adv_pcs = adversarial_pc_input[j]
            adv_knn_dists_list = []
            idx = np.arange(num_examples_curr)
            for b in iterate_in_chunks(idx, knn_batch_size):
                print('shape class %d/%d dist weight %d/%d point cloud %d/%d' % (i + 1, len(pc_classes), j + 1, num_dist_weight, b[-1] + 1, num_examples_curr))
                knn_dists_batch = sess.run(knn_dists, feed_dict={pc_pl: adv_pcs[b]})
                adv_knn_dists_list.append(knn_dists_batch)

            adv_knn_dists = np.vstack(adv_knn_dists_list)

            knn_dists_adversarial_pc_input[j] = adv_knn_dists
        else:
            for l in range(num_examples_curr):
                print('shape class %d/%d dist weight %d/%d point cloud %d/%d' % (i+1, len(pc_classes), j+1, num_dist_weight, l+1, num_examples_curr))
                adv_pc = adversarial_pc_input[j, l]

                # point to point distance matrix
                adv_pc_point_dist_mat = get_dist_mat(adv_pc)

                # sort distances
                adv_pc_point_dist_mat_sorted = np.sort(adv_pc_point_dist_mat, axis=1)
                assert any(adv_pc_point_dist_mat_sorted[:, 0]) is False, 'The distance to the point itself should be 0!'
                adv_pc_point_dist_mat_sorted = adv_pc_point_dist_mat_sorted[:, 1:]  # discard the first distnce (the distance of the point to itself which is 0)

                knn_dists_adversarial_pc_input[j, l] = adv_pc_point_dist_mat_sorted[:, :flags.num_knn]

    assert np.all(knn_dists_adversarial_pc_input >= 0), 'The knn_dists_adversarial_pc_input was not filled correctly'

    # save knn distances per point for adversarial point clouds
    save_dir = create_dir(osp.join(output_path, pc_class_name))
    np.save(osp.join(save_dir, 'knn_dists_adversarial_pc_input'), knn_dists_adversarial_pc_input)

    # get knn distances per point for source point clouds
    knn_dists_source_pc = -1 * np.ones(list(source_pc.shape[:-1]) + [flags.num_knn], dtype=np.float32)
    if flags.use_tf_knn:
        src_knn_dists_list = []
        idx = np.arange(num_examples_curr)
        for b in iterate_in_chunks(idx, knn_batch_size):
            print('shape class %d/%d point cloud %d/%d' % (i + 1, len(pc_classes), b[-1] + 1, num_examples_curr))
            knn_dists_batch = sess.run(knn_dists, feed_dict={pc_pl: source_pc[b]})
            src_knn_dists_list.append(knn_dists_batch)

        knn_dists_source_pc = np.vstack(src_knn_dists_list)
    else:
        for l in range(num_examples_curr):
            if (l + 1) % knn_batch_size == 0 or (l + 1) == num_examples_curr:
                print('shape class %d/%d point cloud %d/%d' % (i + 1, len(pc_classes), l + 1, num_examples_curr))
            src_pc = source_pc[l]

            # point to point distance matrix
            src_pc_point_dist_mat = get_dist_mat(src_pc)

            # sort distances
            src_pc_point_dist_mat_sorted = np.sort(src_pc_point_dist_mat, axis=1)
            assert any(src_pc_point_dist_mat_sorted[:, 0]) is False, 'The distance to the point itself should be 0!'
            src_pc_point_dist_mat_sorted = src_pc_point_dist_mat_sorted[:, 1:]  # discard the first distnce (the distance of the point to itself which is 0)

            knn_dists_source_pc[l] = src_pc_point_dist_mat_sorted[:, :flags.num_knn]

    assert np.all(knn_dists_source_pc >= 0), 'The knn_dists_source_pc was not filled correctly'

    # save knn distances per point for original source point clouds
    save_dir_orig = create_dir(osp.join(output_path_orig, pc_class_name))
    np.save(osp.join(save_dir_orig, 'knn_dists_source_pc'), knn_dists_source_pc)

    duration = time.time() - start_time
    print("Duration (minutes): %.2f" % (duration / 60.0))
