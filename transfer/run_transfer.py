"""
Created on September 1st, 2020
@author: itailang
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
from src.pointnet_ae import PointNetAutoEncoder
from src.adversary_utils import load_data, prepare_data_for_attack, get_quantity_at_index
from src.in_out import create_dir
from src.general_utils import plot_3d_point_cloud
from src.tf_utils import reset_tf_graph
try:
    from transfer.atlasnet.atlasnet_ae import AtlasNetAutoEncoder
except:
    print('Could not import AtlasNetAutoEncoder')
try:
    from transfer.foldingnet.foldingnet_ae import FoldingNetAutoEncoder
except:
    print('Could not import FoldingNetAutoEncoder')

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--transfer_ae_folder', type=str, default='log/autoencoder_for_transfer', help='Folder for loading a trained autoencoder for attack transfer [default: log/autoencoder_for_transfer]')
parser.add_argument('--transfer_ae_restore_epoch', type=int, default=500, help='Restore epoch of a trained autoencoder for attack transfer [default: 500]')
parser.add_argument('--transfer_ae_type', type=str, default='PointNet', help='Type of autoencoder used for transfer.  [default: PointNet]')
parser.add_argument('--ae_folder', type=str, default='log/autoencoder_victim', help='Folder for loading a trained autoencoder model [default: log/autoencoder_victim]')
parser.add_argument("--attack_pc_idx", type=str, default='log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy', help="List of indices of point clouds for the attack")
parser.add_argument("--do_sanity_checks", type=int, default=0, help="1: Do sanity checks, 0: Do not do sanity checks [default: 0]")
parser.add_argument('--attack_folder', type=str, default='attack_res', help='Folder for loading attack data')
parser.add_argument("--output_folder_name", type=str, default='attack_res_transfer', help="Output folder name")
flags = parser.parse_args()

assert flags.transfer_ae_type in ['PointNet', 'AtlasNet', 'FoldingNet'], 'wrong ae_type: %s.' % flags.transfer_ae_type

print('Run transfer flags:', flags)

# define basic parameters
top_out_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))  # Use to save Neural-Net check-points etc.
data_path = osp.join(top_out_dir, flags.ae_folder, 'eval')
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

attack_path = osp.join(data_path, flags.attack_folder)
output_path = create_dir(osp.join(top_out_dir, flags.transfer_ae_folder, 'eval', flags.output_folder_name))

# load attack configuration
conf = Conf.load(osp.join(attack_path, 'attack_configuration'))

# update autoencoder configuration
conf.experiment_name = 'autoencoder'
conf.train_dir = output_path
conf.is_denoising = True  # Required for having a separate placeholder for ground truth point cloud (for computing AE loss)
conf.encoder_args['return_layer_before_symmetry'] = True
assert conf.encoder_args['b_norm_decay'] == 1., 'Reruired for avoiding the update of batch normalization moving_mean and moving_variance parameters'
assert conf.decoder_args['b_norm_decay'] == 1., 'Reruired for avoiding the update of batch normalization moving_mean and moving_variance parameters'
assert conf.decoder_args['b_norm_decay_finish'] == 1., 'Reruired for avoiding the update of batch normalization moving_mean and moving_variance parameters'

# update transfer configuration
conf.attack_path = attack_path
conf.transfer_ae_restore_epoch = flags.transfer_ae_restore_epoch

conf.save(osp.join(output_path, 'transfer_configuration'))

# load data
point_clouds, pc_classes, slice_idx, ae_loss, reconstructions = \
    load_data(data_path, files, ['point_clouds_test_set', 'pc_classes', 'slice_idx_test_set', 'ae_loss_test_set', 'reconstructions_test_set'])

assert np.all(ae_loss > 0), 'Note: not all autoencoder loss values are larger than 0 as they should!'

nn_idx_dict = {'latent_nn': 'latent_nn_idx_test_set', 'chamfer_nn_complete': 'chamfer_nn_idx_complete_test_set'}
nn_idx = load_data(data_path, files, [nn_idx_dict[conf.target_pc_idx_type]])

correct_pred = None
if conf.correct_pred_only:
    pc_labels, pc_pred_labels = load_data(data_path, files, ['pc_label_test_set', 'pc_pred_labels_test_set'])
    correct_pred = (pc_labels == pc_pred_labels)

# load indices for attack
attack_pc_idx = np.load(osp.join(top_out_dir, flags.attack_pc_idx))
attack_pc_idx = attack_pc_idx[:, :conf.num_pc_for_attack]

# Build AE Model
reset_tf_graph()
if flags.transfer_ae_type == 'PointNet':
    ae_instance = PointNetAutoEncoder
elif flags.transfer_ae_type == 'AtlasNet':
    ae_instance = AtlasNetAutoEncoder
else:
    ae_instance = FoldingNetAutoEncoder

ae = ae_instance(conf.experiment_name, conf)

# Reload a saved model
transfer_ae_dir = osp.join(top_out_dir, flags.transfer_ae_folder)
ae.restore_model(transfer_ae_dir, epoch=flags.transfer_ae_restore_epoch, verbose=True)

classes_for_attack = conf.class_names
classes_for_target = conf.class_names

# transfer attack to the AE model
for i in range(len(pc_classes)):
    pc_class_name = pc_classes[i]
    if pc_class_name not in classes_for_attack:
        continue

    save_dir = create_dir(osp.join(output_path, pc_class_name))

    print('transfer shape class %s (%d out of %d classes) ' % (pc_class_name, i + 1, len(pc_classes)))
    start_time = time.time()

    # prepare target point clouds
    source_pc, target_pc = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, point_clouds, slice_idx, attack_pc_idx, conf.num_pc_for_target, conf.target_pc_idx_type, nn_idx, correct_pred)
    _, target_ae_loss_ref = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, ae_loss, slice_idx, attack_pc_idx, conf.num_pc_for_target, conf.target_pc_idx_type, nn_idx, correct_pred)
    _, target_recon_ref = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, reconstructions, slice_idx, attack_pc_idx, conf.num_pc_for_target, conf.target_pc_idx_type, nn_idx, correct_pred)

    target_ae_loss_ref = target_ae_loss_ref.reshape(-1)

    # load data
    load_dir = osp.join(attack_path, pc_class_name)
    adversarial_pc_input = np.load(osp.join(load_dir, 'adversarial_pc_input.npy'))
    adversarial_pc_recon = np.load(osp.join(load_dir, 'adversarial_pc_recon.npy'))
    adversarial_metrics = np.load(osp.join(load_dir, 'adversarial_metrics.npy'))

    # take adversarial point clouds of selected dist weight per attack
    source_target_norm_min_idx = np.load(osp.join(load_dir, 'analysis_results', 'source_target_norm_min_idx.npy'))
    adversarial_pc_input, adversarial_pc_recon, adversarial_metrics = \
        get_quantity_at_index([adversarial_pc_input, adversarial_pc_recon, adversarial_metrics], source_target_norm_min_idx)

    # add axis to keep the interface of dist_weight as the first dim
    adversarial_pc_input, adversarial_pc_recon, adversarial_metrics = \
        [np.expand_dims(data, axis=0) for data in [adversarial_pc_input, adversarial_pc_recon, adversarial_metrics]]

    num_dist_weight, num_pc, _, _ = adversarial_pc_input.shape

    if flags.transfer_ae_type == 'PointNet':
        transferred_pc_recon = np.zeros_like(adversarial_pc_recon)
    elif flags.transfer_ae_type == 'AtlasNet':
        transferred_pc_recon = np.zeros([1, num_pc, 2500, 3], dtype=adversarial_pc_recon.dtype)
    else:
        transferred_pc_recon = np.zeros([1, num_pc, 2025, 3], dtype=adversarial_pc_recon.dtype)

    transferred_target_recon_error = np.zeros([num_dist_weight, num_pc], dtype=adversarial_metrics.dtype)
    transferred_target_nre = np.zeros([num_dist_weight, num_pc], dtype=adversarial_metrics.dtype)

    for j in range(num_dist_weight):
        pc_input = adversarial_pc_input[j]
        if flags.transfer_ae_type == 'PointNet':
            pc_recon = ae.get_reconstructions(pc_input)
        else:
            pc_recon = ae.get_reconstructions(pc_input, flags)
        transferred_pc_recon[j] = pc_recon

        if flags.transfer_ae_type == 'PointNet':
            # the computation of pc_recon is done inside the function, then the the loss is calculated between pc_recon and target_pc
            target_recon_error = ae.get_loss_per_pc(pc_input, target_pc).astype(adversarial_metrics.dtype)
        else:
            # the loss is calculated between pc_recon and target_pc
            target_recon_error = ae.get_loss_per_pc(pc_recon, target_pc).astype(adversarial_metrics.dtype)
        transferred_target_recon_error[j] = target_recon_error

        target_nre = np.divide(target_recon_error, target_ae_loss_ref)
        transferred_target_nre[j] = target_nre

    adversarial_target_recon_error = adversarial_metrics[:, :, 4]
    adversarial_target_nre = adversarial_metrics[:, :, 3]

    # sanity checks
    if flags.transfer_ae_folder == flags.ae_folder and flags.transfer_ae_restore_epoch == conf.ae_restore_epoch and flags.do_sanity_checks:
        assert flags.transfer_ae_type == 'PointNet', 'the sanity checks are for transfer_ae_type "PointNet" (got "%s")' % flags.transfer_ae_type
        target_recon = ae.get_reconstructions(target_pc)
        target_ae_loss = ae.get_loss_per_pc(target_pc)

        diff_target_recon_max = np.abs(target_recon - target_recon_ref).max()
        assert diff_target_recon_max < 1e-06, \
            'when transfer_ae_folder and ae_folder are the same, the ae target reconstructions should also be the same! (up to precision errors)'

        diff_target_ae_loss_max = np.abs(target_ae_loss - target_ae_loss_ref).max()
        assert diff_target_ae_loss_max < 1e-08, \
            'when transfer_ae_folder and ae_folder are the same, the ae target loss should also be the same! (up to precision errors)'

        diff_adversarial_recon_max = np.abs(transferred_pc_recon - adversarial_pc_recon).max()
        assert diff_adversarial_recon_max < 1e-06, \
            'when transfer_ae_folder and ae_folder are the same, the ae adversarial reconstructions should also be the same! (up to precision errors)'

        diff_target_recon_error_max = np.abs(transferred_target_recon_error - adversarial_target_recon_error).max()
        assert diff_target_recon_error_max < 1e-08, \
            'when transfer_ae_folder and ae_folder are the same, the ae target recon error should also be the same! (up to precision errors)'

        diff_target_nre_max = np.abs(transferred_target_nre - adversarial_target_nre).max()
        assert diff_target_nre_max < 1e-04, \
            'when transfer_ae_folder and ae_folder are the same, the ae target normalized recon error should also be the same! (up to precision errors)'

    transfer_metrics = np.concatenate([np.expand_dims(m, axis=-1) for m in
                                      [transferred_target_recon_error, transferred_target_nre,
                                       adversarial_target_recon_error, adversarial_target_nre]], axis=-1)

    show = False
    if show:
        j, k = 0, 0
        plot_3d_point_cloud(source_pc[k], title='source pc')
        plot_3d_point_cloud(target_pc[k], title='target pc')
        plot_3d_point_cloud(adversarial_pc_input[j, k], title='adversarial pc input')
        plot_3d_point_cloud(adversarial_pc_recon[j, k], title='adversarial pc recon')
        plot_3d_point_cloud(transferred_pc_recon[j, k], title='transferred pc recon')

    # save results
    if flags.transfer_ae_folder != flags.ae_folder:
        np.save(osp.join(save_dir, 'transferred_pc_recon'), transferred_pc_recon)
        np.save(osp.join(save_dir, 'transfer_metrics'), transfer_metrics)

    duration = time.time() - start_time
    print("Duration (minutes): %.2f" % (duration / 60.0))
