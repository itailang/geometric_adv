"""
Created on September 25th, 2019
@author: itailang
"""

# import system modules
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
from src.pointnet_ae import PointNetAutoEncoder

from src.in_out import create_dir, load_dataset
from src.shift_rotate_util import sort_axes
from src.general_utils import plot_3d_point_cloud
from src.tf_utils import reset_tf_graph

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--restore_epoch', type=int, default=500, help='Restore epoch of a trained autoencoder [default: 500]')
parser.add_argument('--set_type', type=str, default='test_set', help='Set for evaluation of the autoencoder [default: test_set]')
parser.add_argument('--train_folder', type=str, default='log/autoencoder_victim', help='Folder for saved data form the training phase [default: log/autoencoder_victim]')
parser.add_argument('--output_folder_name', type=str, default='eval', help="Output folder name")
flags = parser.parse_args()

print('Test autoencoder flags:', flags)

assert flags.set_type in ['train_set', 'val_set', 'test_set'], 'wrong set_type: %s' % flags.set_type

# define basic parameters
project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
top_in_dir = osp.join(project_dir, 'data', 'shape_net_core_uniform_samples_2048')  # Top-dir of where point-clouds are stored.
top_out_dir = osp.join(project_dir)  # Use to save Neural-Net check-points etc.

# Load train configuration
train_dir = create_dir(osp.join(top_out_dir, flags.train_folder))
restore_epoch = flags.restore_epoch
conf = Conf.load(train_dir + '/configuration')
conf.encoder_args['return_layer_before_symmetry'] = True

# Load point clouds
object_class = conf.object_class
class_names = conf.class_names
pc_data, slice_idx, pc_label = load_dataset(class_names, flags.set_type, top_in_dir)
point_clouds = pc_data.point_clouds.copy()

# Sort point cloud axes
if conf.sort_axes:
    point_clouds_axes_sorted = sort_axes(point_clouds)
    point_clouds = point_clouds_axes_sorted

    show = False
    if show:
        plot_3d_point_cloud(point_clouds[0])
        plot_3d_point_cloud(point_clouds_axes_sorted[0])

# Build AE Model
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

# Reload a saved model
ae.restore_model(train_dir, epoch=restore_epoch, verbose=True)

# Create evaluation dir
eval_dir = create_dir(osp.join(train_dir, flags.output_folder_name))

# Save point clouds data
pc_classes_np = np.array(class_names)
file_name = '_'.join(['pc_classes'] + object_class) + '.npy'
file_path = osp.join(eval_dir, file_name)
np.save(file_path, pc_classes_np)

pc_label_np = np.array(pc_label).astype(np.int8)
file_name = '_'.join(['pc_label', flags.set_type] + object_class) + '.npy'
file_path = osp.join(eval_dir, file_name)
np.save(file_path, pc_label_np)

slice_idx_np = np.array(slice_idx)
file_name = '_'.join(['slice_idx', flags.set_type] + object_class) + '.npy'
file_path = osp.join(eval_dir, file_name)
np.save(file_path, slice_idx_np)

file_name = '_'.join(['point_clouds', flags.set_type] + object_class) + '.npy'
file_path = osp.join(eval_dir, file_name)
np.save(file_path, point_clouds)

# Latent representation of point clouds
latent_vectors = ae.get_latent_vectors(point_clouds)

# Save latent representation
file_name = '_'.join(['latent_vectors', flags.set_type] + object_class) + '.npy'
file_path = osp.join(eval_dir, file_name)
np.save(file_path, latent_vectors)

# Reconstruct point clouds
reconstructions = ae.get_reconstructions(point_clouds)

# Save reconstructions
file_name = '_'.join(['reconstructions', flags.set_type] + object_class) + '.npy'
file_path = osp.join(eval_dir, file_name)
np.save(file_path, reconstructions)

# Compute loss per point cloud
loss_per_pc = ae.get_loss_per_pc(point_clouds)

# Save loss per pc
file_name = '_'.join(['ae_loss', flags.set_type] + object_class) + '.npy'
file_path = osp.join(eval_dir, file_name)
np.save(file_path, loss_per_pc)

# Save log file
log_file_name = '_'.join(['eval_stats', flags.set_type] + object_class) + '.txt'
log_file = open(osp.join(eval_dir, log_file_name), 'w', 1)
log_file.write('Mean ae loss: %.9f\n' % loss_per_pc.mean())
log_file.close()
