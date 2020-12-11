"""
Created on September 5th, 2018
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
from src.ae_templates import mlp_architecture, default_train_params
from src.autoencoder import Configuration as Conf
from src.pointnet_ae import PointNetAutoEncoder

from src.in_out import create_dir, load_dataset
from src.shift_rotate_util import sort_axes
from src.general_utils import plot_3d_point_cloud
from src.tf_utils import reset_tf_graph

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--training_epochs', type=int, default=500, help='Number of training epochs [default: 500]')
parser.add_argument('--save_config_and_exit', type=int, default=0, help='1: Save autoencoder configuration and exit, 0: Do not exit [default: 0]')
parser.add_argument('--sort_axes', type=int, default=1, help='1: Sort point cloud axes, 0: Do not sort axes [default: 1]')
parser.add_argument('--train_folder', type=str, default='log/autoencoder_victim', help='Folder for saving data form the training phase [default: log/autoencoder_victim]')
flags = parser.parse_args()

print('Train autoencoder flags:', flags)

# Define basic parameters
project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
top_in_dir = osp.join(project_dir, 'data', 'shape_net_core_uniform_samples_2048')  # Top-dir of where point-clouds are stored.
top_out_dir = project_dir  # Use to save Neural-Net check-points etc.
train_dir = create_dir(osp.join(top_out_dir, flags.train_folder))

experiment_name = 'autoencoder'
n_pc_points = 2048                # Number of points per model
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'chamfer'               # Loss to optimize

object_class = ['13l']
class_names = ['table', 'car', 'chair', 'airplane', 'sofa', 'rifle', 'lamp', 'watercraft', 'bench', 'loudspeaker', 'cabinet', 'display', 'telephone']

# Load default train parameters
train_params = default_train_params()
train_params['training_epochs'] = flags.training_epochs

# Load default architecture
encoder, decoder, enc_args, dec_args = mlp_architecture(n_pc_points, bneck_size)

conf = Conf(n_input=[n_pc_points, 3],
            loss=ae_loss,
            training_epochs=train_params['training_epochs'],
            batch_size=train_params['batch_size'],
            denoising=train_params['denoising'],
            learning_rate=train_params['learning_rate'],
            train_dir=train_dir,
            loss_display_step=train_params['loss_display_step'],
            saver_step=train_params['saver_step'],
            z_rotate=train_params['z_rotate'],
            encoder=encoder,
            decoder=decoder,
            encoder_args=enc_args,
            decoder_args=dec_args
            )
conf.experiment_name = experiment_name
conf.held_out_step = 5  # how often to evaluate/print the loss on held_out data (if they are provided)
conf.object_class = object_class
conf.class_names = class_names
conf.sort_axes = flags.sort_axes
conf.encoder_args['return_layer_before_symmetry'] = True
conf.save(osp.join(train_dir, 'configuration'))

if flags.save_config_and_exit:
    exit()

# Load point clouds
pc_data_train, _, _ = load_dataset(class_names, 'train_set', top_in_dir)
pc_data_val, _, _ = load_dataset(class_names, 'val_set', top_in_dir)

# Sort point cloud axes
if flags.sort_axes:
    point_clouds_train_axes_sorted = sort_axes(pc_data_train.point_clouds)
    pc_data_train.point_clouds = point_clouds_train_axes_sorted

    point_clouds_val_axes_sorted = sort_axes(pc_data_val.point_clouds)
    pc_data_val.point_clouds = point_clouds_val_axes_sorted

    show = False
    if show:
        plot_3d_point_cloud(pc_data_train.point_clouds[0])
        plot_3d_point_cloud(point_clouds_train_axes_sorted[0])
        plot_3d_point_cloud(pc_data_val.point_clouds[0])
        plot_3d_point_cloud(point_clouds_val_axes_sorted[0])

if len(class_names) > 1:
    pc_data_train.shuffle_data(seed=55)
    pc_data_val.shuffle_data(seed=55)

# Build AE Model
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

# Train the AE (save output to train_stats.txt)
buf_size = 1  # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(pc_data_train, conf, log_file=fout, held_out_data=pc_data_val)
fout.close()
