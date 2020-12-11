"""
Created on September 21st, 2020
@author: urikotlicki
"""

# import system modules
import os
import os.path as osp
import sys
import time
import argparse
import numpy as np

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from src.autoencoder import Configuration as Conf
from src.adversary_utils import load_data, prepare_data_for_attack, get_quantity_at_index
from src.in_out import create_dir
from src.general_utils import plot_3d_point_cloud
from src.tf_utils import reset_tf_graph
from classifier.pointnet_classifier import PointNetClassifier

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--classifier_folder', type=str, default='log/pointnet', help='Folder of the classifier to be used [default: log/pointnet]')
parser.add_argument('--classifier_restore_epoch', type=int, default=140, help='Restore epoch for the pre-trained classifier [default: 140]')
parser.add_argument('--data_type', type=str, default='adversarial', help='Data type to be classified [default: adversarial]')
parser.add_argument('--ae_folder', type=str, default='log/autoencoder_victim', help='Folder for loading a trained autoencoder model [default: log/autoencoder_victim]')
parser.add_argument('--num_points', type=int, default=2048, help='Number of points in the reconstructed point cloud [default: 2048]')
parser.add_argument("--attack_pc_idx", type=str, default='log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy', help="List of indices of point clouds for the attack")
parser.add_argument('--attack_folder', type=str, default='attack_res', help='Folder for loading attack data')
parser.add_argument('--defense_folder', type=str, default='defense_critical_res', help='Folder for loading defense data')
parser.add_argument("--output_folder_name", type=str, default='classifier_res', help="Output folder name")
parser.add_argument('--num_classes', type=int, default=13, help='Number of classes [default: 13]')
flags = parser.parse_args()

print('Run classifier flags:', flags)

assert flags.data_type in ['clean', 'adversarial', 'before_defense', 'after_defense'], 'wrong data_type: %s.' % flags.data_type

# define basic parameters
top_out_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))  # Use to save Neural-Net check-points etc.
data_path = osp.join(top_out_dir, flags.ae_folder, 'eval')
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

classifier_path = osp.join(top_out_dir, flags.classifier_folder)

if flags.data_type == 'clean':
    classifier_data_path = data_path
    output_path = data_path
elif flags.data_type == 'adversarial':
    classifier_data_path = osp.join(data_path, flags.attack_folder)
    output_path = create_dir(osp.join(classifier_data_path, flags.output_folder_name))
elif flags.data_type == 'before_defense':
    classifier_data_path = osp.join(data_path, flags.attack_folder)
    output_path = create_dir(osp.join(classifier_data_path, flags.defense_folder, flags.output_folder_name))
else:
    classifier_data_path = osp.join(data_path, flags.attack_folder, flags.defense_folder)
    output_path = create_dir(osp.join(classifier_data_path, flags.output_folder_name))

# load configuration
if flags.data_type == 'clean':
    conf = Conf.load(osp.join(top_out_dir, flags.ae_folder, 'configuration'))
elif flags.data_type == 'adversarial':
    conf = Conf.load(osp.join(classifier_data_path, 'attack_configuration'))
elif flags.data_type == 'before_defense':
    conf = Conf.load(osp.join(classifier_data_path, flags.defense_folder, 'defense_configuration'))
else:
    conf = Conf.load(osp.join(classifier_data_path, 'defense_configuration'))

batch_size = 2 if flags.data_type == 'clean' else 10

# update classifier configuration
conf.classifier_path = classifier_path
conf.classifier_restore_epoch = flags.classifier_restore_epoch
conf.classifier_data_path = classifier_data_path

conf.save(osp.join(output_path, 'classifier_configuration'))

# load data
point_clouds, pc_classes, slice_idx, reconstructions = \
    load_data(data_path, files, ['point_clouds_test_set', 'pc_classes', 'slice_idx_test_set', 'reconstructions_test_set'])

if flags.data_type != 'clean':
    nn_idx_dict = {'latent_nn': 'latent_nn_idx_test_set','chamfer_nn_complete': 'chamfer_nn_idx_complete_test_set'}
    nn_idx = load_data(data_path, files, [nn_idx_dict[conf.target_pc_idx_type]])

    correct_pred = None
    if conf.correct_pred_only:
        pc_labels, pc_pred_labels = load_data(data_path, files, ['pc_label_test_set', 'pc_pred_labels_test_set'])
        correct_pred = (pc_labels == pc_pred_labels)

    # load indices for attack
    attack_pc_idx = np.load(osp.join(top_out_dir, flags.attack_pc_idx))
    attack_pc_idx = attack_pc_idx[:, :conf.num_pc_for_attack]

# build classifier model and reload a saved model
reset_tf_graph()
classifier = PointNetClassifier(classifier_path, flags.classifier_restore_epoch, num_points=flags.num_points, batch_size=batch_size, num_classes=flags.num_classes)

if flags.data_type == 'clean':
    reconstructions_pred = classifier.classify(reconstructions)

    reconstructions_file_name = [f for f in files if 'reconstructions_test_set' in f][0]
    file_name_parts = reconstructions_file_name.split('_')
    reconstructions_pred_file_name = '_'.join(['pc_pred_labels'] + file_name_parts[-3:])
    reconstructions_pred_file_path = osp.join(output_path, reconstructions_pred_file_name)
    np.save(reconstructions_pred_file_path, reconstructions_pred)
    exit()

classes_for_attack = conf.class_names
classes_for_target = conf.class_names

for i in range(len(pc_classes)):
    pc_class_name = pc_classes[i]
    if pc_class_name not in classes_for_attack:
        continue

    save_dir = create_dir(osp.join(output_path, pc_class_name))

    print('Classify shape class %s (%d out of %d classes) ' % (pc_class_name, i + 1, len(pc_classes)))
    start_time = time.time()

    # prepare target point clouds
    source_recon_ref, target_recon_ref = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, reconstructions, slice_idx, attack_pc_idx, conf.num_pc_for_target, conf.target_pc_idx_type, nn_idx, correct_pred)

    # load data
    load_dir = osp.join(classifier_data_path, pc_class_name)
    if flags.data_type in ['adversarial', 'before_defense']:
        adversarial_pc_recon = np.load(osp.join(load_dir, 'adversarial_pc_recon.npy'))

        # take adversarial point clouds of selected dist weight per attack
        source_target_norm_min_idx = np.load(osp.join(load_dir, 'analysis_results', 'source_target_norm_min_idx.npy'))
        adversarial_pc_recon = get_quantity_at_index([adversarial_pc_recon], source_target_norm_min_idx)

        # add axis to keep the interface of dist_weight as the first dim
        pc_recon = np.expand_dims(adversarial_pc_recon, axis=0)
    elif flags.data_type == 'after_defense':
        defense_on_adv = osp.exists(osp.join(load_dir, 'defended_pc_recon.npy'))
        if defense_on_adv:
            pc_recon = np.load(osp.join(load_dir, 'defended_pc_recon.npy'))  # defense on adversarial input
        else:
            defended_pc_recon = np.load(osp.join(load_dir, 'defended_source_recon.npy'))  # defense on clean input

            # add axis to keep the interface of dist_weight as the first dim
            pc_recon = np.expand_dims(defended_pc_recon, axis=0)
    else:
        assert False, 'wrong data_type: %s' % flags.data_type

    num_dist_weight, num_pc, _, _ = pc_recon.shape
    pc_recon_pred = np.zeros([num_dist_weight, num_pc], dtype=np.int8)

    # reconstructed pc prediction label
    for j in range(num_dist_weight):
        pc_recon_pred[j] = classifier.classify(pc_recon[j])

    # save results
    if flags.data_type in ['adversarial', 'before_defense']:
        np.save(osp.join(save_dir, 'adversarial_pc_recon_pred'), pc_recon_pred)
    elif flags.data_type == 'after_defense':
        if defense_on_adv:
            np.save(osp.join(save_dir, 'defended_pc_recon_pred'), pc_recon_pred)
        else:
            pc_recon_pred = np.squeeze(pc_recon_pred, axis=0)
            np.save(osp.join(save_dir, 'defended_source_recon_pred'), pc_recon_pred)
    else:
        assert False, 'wrong data_type: %s' % flags.data_type

    duration = time.time() - start_time
    print("Duration (minutes): %.2f" % (duration / 60.0))
