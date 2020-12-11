"""
Created on June 6th, 2020
@author: itailang
"""

# import system modules
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import tensorflow as tf

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from src.in_out import create_dir
from src.general_utils import get_dist_mat, plot_3d_point_cloud
from src.adversary_utils import load_data
from src.tf_utils import reset_tf_graph
from external.structural_losses.tf_nndistance import nn_distance

parser = argparse.ArgumentParser()
parser.add_argument("--ae_folder", type=str, default="log/autoencoder_victim", help="Folder for loading a trained autoencoder model [default: log/autoencoder_victim]")
parser.add_argument("--get_rand_idx", type=int, default=0, help="1: Compute randomly selected indices, 0: do not compute [default: 0]")
parser.add_argument("--get_latent_nn_idx", type=int, default=0, help="1: Compute indices of nearest neighbors in latent space, 0: do not compute [default: 0]")
parser.add_argument("--get_chamfer_nn_idx", type=int, default=0, help="1: Compute indices of nearest neighbors in points space for complete shapes, 0: do not compute [default: 0]")

# flags for random indices
parser.add_argument("--num_instance_per_class", type=int, default=100, help="Number of instances to select from each shape class [default: 100]")

# flags for nearest neighbors in points space
parser.add_argument("--pc_start_idx", type=int, default=0, help="Start index for source point clouds [default: 0]")
parser.add_argument("--pc_batch_size", type=int, default=100, help="Batch size of source point clouds [default: 100]")

flags = parser.parse_args()

print('Prepare indices flags:', flags)

# define basic parameters
project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
data_path = create_dir(osp.join(project_dir, flags.ae_folder, 'eval'))
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

# load data
point_clouds, latent_vectors, pc_classes, slice_idx = \
    load_data(data_path, files, ['point_clouds_test_set', 'latent_vectors_test_set', 'pc_classes', 'slice_idx_test_set'])

show = False
if show:
    n = 0
    plot_3d_point_cloud(point_clouds[n])

slice_idx_file_name = [f for f in files if 'slice_idx_test_set' in f][0]
file_name_parts = slice_idx_file_name.split('_')

# constants
num_classes = len(pc_classes)
range_num_classes = range(num_classes)

# reproducibility
seed = 55


def get_rand_idx():
    # loop over categories
    sel_idx = -1 * np.ones([num_classes, flags.num_instance_per_class], dtype=np.int16)

    for i in range_num_classes:
        np.random.seed(seed)

        num_examples = slice_idx[i + 1] - slice_idx[i]
        perm = np.arange(num_examples)
        np.random.shuffle(perm)

        num_instances = min(flags.num_instance_per_class, num_examples)
        sel_idx[i, :num_instances] = perm[:flags.num_instance_per_class]

    sel_idx_file_name = '_'.join(['sel_idx', 'rand', '%d' % flags.num_instance_per_class] + file_name_parts[-3:])
    sel_idx_file_path = osp.join(data_path, sel_idx_file_name)
    np.save(sel_idx_file_path, sel_idx)


def get_latent_nn():
    latent_dist_mat = get_dist_mat(latent_vectors)

    latent_dist_mat_file_name = '_'.join(['latent_dist_mat'] + file_name_parts[-3:])
    latent_dist_mat_file_path = osp.join(data_path, latent_dist_mat_file_name)
    np.save(latent_dist_mat_file_path, latent_dist_mat)

    # nearest neighbors indices
    latent_nn_idx = sort_dist_mat(latent_dist_mat)

    latent_nn_idx_file_name = '_'.join(['latent_nn_idx'] + file_name_parts[-3:])
    latent_nn_idx_file_path = osp.join(data_path, latent_nn_idx_file_name)
    np.save(latent_nn_idx_file_path, latent_nn_idx)


def get_chamfer_nn():
    start_time = time.time()
    num_examples_all, num_points, _ = point_clouds.shape
    chamfer_batch_size = 10

    # build chamfer graph
    reset_tf_graph()
    source_pc_pl = tf.placeholder(tf.float32, shape=[None, num_points, 3])
    target_pc_pl = tf.placeholder(tf.float32, shape=[None, num_points, 3])
    dists_s_t, _, dists_t_s, _ = nn_distance(source_pc_pl, target_pc_pl)
    chamfer_dist = tf.reduce_mean(dists_s_t, axis=1) + tf.reduce_mean(dists_t_s, axis=1)

    sess = tf.Session('')

    # compute chamfer distance matrix
    point_clouds_curr = point_clouds[flags.pc_start_idx:flags.pc_start_idx + flags.pc_batch_size]
    num_examples_curr = len(point_clouds_curr)
    chamfer_dist_mat_curr = -1 * np.ones([num_examples_all, num_examples_curr], dtype=np.float32)

    source_pc = np.tile(np.expand_dims(point_clouds_curr, axis=0), [num_examples_all, 1, 1, 1])
    target_pc = np.tile(np.expand_dims(point_clouds, axis=1), [1, num_examples_curr, 1, 1])

    for i in range(0, num_examples_all, chamfer_batch_size):
        for j in range(0, num_examples_curr, chamfer_batch_size):
            #print('i %d out of %d, j %d out of %d' %
            #      (min(i + chamfer_batch_size, num_examples_all), num_examples_all, min(j + chamfer_batch_size, num_examples_curr), num_examples_curr))

            sources = source_pc[i:i + chamfer_batch_size, j:j + chamfer_batch_size]
            targets = target_pc[i:i + chamfer_batch_size, j:j + chamfer_batch_size]

            s_batch = np.reshape(sources, [-1, num_points, 3])
            t_batch = np.reshape(targets, [-1, num_points, 3])
            feed_dict = {source_pc_pl: s_batch, target_pc_pl: t_batch}
            dist_batch = sess.run(chamfer_dist, feed_dict=feed_dict)
            dist_batch_reshape = np.reshape(dist_batch, sources.shape[:2])
            chamfer_dist_mat_curr[i:i + chamfer_batch_size, j:j + chamfer_batch_size] = dist_batch_reshape

    assert chamfer_dist_mat_curr.min() >= 0, 'the chamfer_dist_mat_curr matrix was not filled correctly'

    # save current chamfer distance data
    chamfer_dist_mat_file_name = '_'.join(['chamfer_dist_mat_complete'] + file_name_parts[-3:])
    chamfer_dist_mat_file_path = osp.join(data_path, chamfer_dist_mat_file_name)
    if osp.exists(chamfer_dist_mat_file_path):
        chamfer_dist_mat = np.load(chamfer_dist_mat_file_path)
    else:
        chamfer_dist_mat = -1 * np.ones([num_examples_all, num_examples_all], dtype=np.float32)

    chamfer_dist_mat[:, flags.pc_start_idx:flags.pc_start_idx + flags.pc_batch_size] = chamfer_dist_mat_curr
    np.save(chamfer_dist_mat_file_path, chamfer_dist_mat)

    duration = time.time() - start_time
    print('start index %d end index %d, out of size %d, duration (minutes): %.2f' %
          (flags.pc_start_idx, min(flags.pc_start_idx + flags.pc_batch_size, num_examples_all), num_examples_all, duration / 60.0))

    if chamfer_dist_mat.min() >= 0:
        # nearest neighbors indices
        chamfer_nn_idx = sort_dist_mat(chamfer_dist_mat)

        chamfer_nn_idx_file_name = '_'.join(['chamfer_nn_idx_complete'] + file_name_parts[-3:])
        chamfer_nn_idx_file_path = osp.join(data_path, chamfer_nn_idx_file_name)
        np.save(chamfer_nn_idx_file_path, chamfer_nn_idx)


def sort_dist_mat(dist_mat):
    nn_idx = -1 * np.ones(dist_mat.shape, dtype=np.int16)

    # sorting indices (in ascending order) for each pair of source and target classes. Note:
    # 1. the indices start from 0 for each pair
    # 2. for same source and target classes (intra class), for each instance the smallest distance is 0. thus, the first index should be discarded
    for i in range_num_classes:
        for j in range_num_classes:
            dist_mat_source_target = dist_mat[slice_idx[i]:slice_idx[i + 1], slice_idx[j]:slice_idx[j + 1]]
            sort_idx = np.argsort(dist_mat_source_target, axis=1).astype(np.int16)
            nn_idx[slice_idx[i]:slice_idx[i + 1], slice_idx[j]:slice_idx[j + 1]] = sort_idx

    assert nn_idx.min() >= 0, 'the nn_idx matrix was not filled correctly'
    return nn_idx


if __name__ == "__main__":
    ##################
    # random indices #
    ##################
    if flags.get_rand_idx:
        get_rand_idx()

    #########################################
    # Nearest neighbors in the latent space #
    #########################################
    if flags.get_latent_nn_idx:
        get_latent_nn()

    #############################################################
    # Nearest neighbors in the points space for complete shapes #
    #############################################################
    if flags.get_chamfer_nn_idx:
        get_chamfer_nn()
