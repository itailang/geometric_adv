"""
Created on September 5th, 2020
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
from src.adversary_utils import load_data, prepare_data_for_attack, get_quantity_at_index, get_outlier_pc_inlier_pc
from src.in_out import create_dir
from src.general_utils import plot_3d_point_cloud
from src.tf_utils import reset_tf_graph

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ae_folder', type=str, default='log/autoencoder_victim', help='Folder for loading a trained autoencoder model [default: log/autoencoder_victim]')
parser.add_argument("--attack_pc_idx", type=str, default='log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy', help="List of indices of point clouds for the attack")
parser.add_argument('--attack_folder', type=str, default='attack_res', help='Folder for loading attack data')
parser.add_argument("--num_knn_for_defense", type=int, default=2, help='Number of nearest neighbors for computing defense statistics [default: 2]')
parser.add_argument("--knn_dist_thresh", type=float, default=0.04, help='Threshold on average knn distance to consider a point as an outlier [default: 0.04]')
parser.add_argument("--do_sanity_checks", type=int, default=0, help="1: Do sanity checks, 0: Do not do sanity checks [default: 0]")
parser.add_argument("--output_folder_name", type=str, default='defense_surface_res', help="Output folder name")
flags = parser.parse_args()

print('Run defense surface flags:', flags)

# define basic parameters
top_out_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))  # Use to save Neural-Net check-points etc.
data_path = osp.join(top_out_dir, flags.ae_folder, 'eval')
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

attack_dir = osp.join(top_out_dir, flags.ae_folder, 'eval', flags.attack_folder)
output_path = create_dir(osp.join(attack_dir, flags.output_folder_name))
output_path_orig = create_dir(osp.join(attack_dir, flags.output_folder_name + '_orig'))

# load data
point_clouds, latent_vectors, pc_classes, slice_idx, ae_loss, reconstructions = \
    load_data(data_path, files, ['point_clouds_test_set', 'latent_vectors_test_set',
                                 'pc_classes', 'slice_idx_test_set', 'ae_loss_test_set', 'reconstructions_test_set'])

assert np.all(ae_loss > 0), 'Note: not all autoencoder loss values are larger than 0 as they should!'
bottleneck_size = latent_vectors.shape[1]

# load attack configuration
conf = Conf.load(osp.join(attack_dir, 'attack_configuration'))

# update autoencoder configuration
conf.experiment_name = 'autoencoder'
conf.train_dir = output_path
conf.ae_dir = osp.join(top_out_dir, flags.ae_folder)
conf.is_denoising = True  # Required for having a separate placeholder for ground truth point cloud (for computing AE loss)
conf.encoder_args['return_layer_before_symmetry'] = True
assert conf.encoder_args['b_norm_decay'] == 1., 'Reruired for avoiding the update of batch normalization moving_mean and moving_variance parameters'
assert conf.decoder_args['b_norm_decay'] == 1., 'Reruired for avoiding the update of batch normalization moving_mean and moving_variance parameters'
assert conf.decoder_args['b_norm_decay_finish'] == 1., 'Reruired for avoiding the update of batch normalization moving_mean and moving_variance parameters'

# update defense configuration
conf.num_knn_for_defense = flags.num_knn_for_defense
conf.knn_dist_thresh = flags.knn_dist_thresh

conf.save(osp.join(output_path, 'defense_configuration'))

conf.train_dir = output_path_orig
conf.save(osp.join(output_path_orig, 'defense_configuration'))

# load indices for attack
attack_pc_idx = np.load(osp.join(top_out_dir, flags.attack_pc_idx))
attack_pc_idx = attack_pc_idx[:, :conf.num_pc_for_attack]

nn_idx_dict = {'latent_nn': 'latent_nn_idx_test_set', 'chamfer_nn_complete': 'chamfer_nn_idx_complete_test_set'}
nn_idx = load_data(data_path, files, [nn_idx_dict[conf.target_pc_idx_type]])

correct_pred = None
if conf.correct_pred_only:
    pc_labels, pc_pred_labels = load_data(data_path, files, ['pc_label_test_set', 'pc_pred_labels_test_set'])
    correct_pred = (pc_labels == pc_pred_labels)

# build AE model
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

# reload a saved model
ae.restore_model(conf.ae_dir, epoch=conf.ae_restore_epoch, verbose=True)

classes_for_attack = conf.class_names
classes_for_target = conf.class_names

# defend the AE model
for i in range(len(pc_classes)):
    pc_class_name = pc_classes[i]
    if pc_class_name not in classes_for_attack:
        continue

    save_dir = create_dir(osp.join(output_path, pc_class_name))
    save_dir_orig = create_dir(osp.join(output_path_orig, pc_class_name))

    print('defend shape class %s (%d out of %d classes) ' % (pc_class_name, i + 1, len(pc_classes)))
    start_time = time.time()

    # prepare data for defense
    source_pc, target_pc = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, point_clouds, slice_idx, attack_pc_idx, conf.num_pc_for_target, nn_idx, correct_pred)
    source_ae_loss_ref, target_ae_loss_ref = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, ae_loss, slice_idx, attack_pc_idx, conf.num_pc_for_target, nn_idx, correct_pred)
    source_recon_ref, _ = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, reconstructions, slice_idx, attack_pc_idx, conf.num_pc_for_target, nn_idx, correct_pred)

    source_ae_loss_ref = source_ae_loss_ref.reshape(-1)
    target_ae_loss_ref = target_ae_loss_ref.reshape(-1)

    # sanity checks
    if flags.do_sanity_checks:
        source_recon = ae.get_reconstructions(source_pc)
        source_ae_loss = ae.get_loss_per_pc(source_pc)

        diff_source_recon_max = np.abs(source_recon - source_recon_ref).max()
        assert diff_source_recon_max < 1e-06, \
            'The ae source reconstructions should also be the same! (up to precision errors)'

        diff_source_ae_loss_max = np.abs(source_ae_loss - source_ae_loss_ref).max()
        assert diff_source_ae_loss_max < 1e-08, \
            'the ae source loss should also be the same! (up to precision errors)'

    # load data
    load_dir = osp.join(attack_dir, pc_class_name)
    adversarial_pc_input = np.load(osp.join(load_dir, 'adversarial_pc_input.npy'))
    adversarial_pc_recon = np.load(osp.join(load_dir, 'adversarial_pc_recon.npy'))
    adversarial_metrics = np.load(osp.join(load_dir, 'adversarial_metrics.npy'))

    knn_dists_adversarial_pc_input = np.load(osp.join(save_dir, 'knn_dists_adversarial_pc_input.npy'))

    # take adversarial point clouds of selected dist weight per attack
    source_target_norm_min_idx = np.load(osp.join(load_dir, 'analysis_results', 'source_target_norm_min_idx.npy'))
    adversarial_pc_input, adversarial_pc_recon, adversarial_metrics = \
        get_quantity_at_index([adversarial_pc_input, adversarial_pc_recon, adversarial_metrics], source_target_norm_min_idx)

    # add axis to keep the interface of dist_weight as the first dim
    adversarial_pc_input, adversarial_pc_recon, adversarial_metrics = \
        [np.expand_dims(data, axis=0) for data in [adversarial_pc_input, adversarial_pc_recon, adversarial_metrics]]

    num_dist_weight, num_pc, num_points, _ = adversarial_pc_input.shape

    adversarial_outlier_points = np.zeros([num_dist_weight, num_pc, num_points, 3], dtype=adversarial_pc_input.dtype)
    adversarial_outlier_idx = np.zeros([num_dist_weight, num_pc, num_points], dtype=np.int16)
    adversarial_outlier_num = np.zeros([num_dist_weight, num_pc], dtype=np.int16)
    defended_pc_input = np.zeros_like(adversarial_pc_input)
    defended_pc_recon = np.zeros_like(adversarial_pc_input)
    defended_source_recon_error = np.zeros([num_dist_weight, num_pc], dtype=adversarial_metrics.dtype)
    defended_source_nre = np.zeros([num_dist_weight, num_pc], dtype=adversarial_metrics.dtype)
    adversarial_source_recon_error = np.zeros([num_dist_weight, num_pc], dtype=adversarial_metrics.dtype)
    adversarial_source_nre = np.zeros([num_dist_weight, num_pc], dtype=adversarial_metrics.dtype)

    for j in range(num_dist_weight):
        pc_input = adversarial_pc_input[j]
        pc_recon = adversarial_pc_recon[j]
        target_recon_error_ref = adversarial_metrics[j, :, 4]
        target_nre_ref = adversarial_metrics[j, :, 3]

        # sanity checks
        if flags.do_sanity_checks:
            adv_recon = ae.get_reconstructions(pc_input)
            target_recon_error = ae.get_loss_per_pc(pc_input, target_pc).astype(adversarial_metrics.dtype)
            target_nre = np.divide(target_recon_error, target_ae_loss_ref)

            diff_adv_recon_max = np.abs(pc_recon - adv_recon).max()
            assert diff_adv_recon_max < 1e-06, \
                'Reconstructions from the attack and reconstructions when running the adversarial point clouds through the AE should also be the same! (up to precision errors)'

            diff_target_recon_error_max = np.abs(target_recon_error - target_recon_error_ref).max()
            assert diff_target_recon_error_max < 1e-08, \
                'The target recon error from the AE and from the attack should be the same! (up to precision errors)'

            diff_target_nre_max = np.abs(target_nre - target_nre_ref).max()
            assert diff_target_nre_max < 1e-04, \
                'The target normalized recon error from the AE and from the attack should be the same! (up to precision errors)'

        knn_dists = knn_dists_adversarial_pc_input[j]
        knn_dists_mean = np.mean(knn_dists[:, :, :flags.num_knn_for_defense], axis=-1)

        outlier_pc, outlier_idx, outlier_num, pc_defended = \
            get_outlier_pc_inlier_pc(pc_input, knn_dists_mean, flags.knn_dist_thresh)

        adversarial_outlier_points[j], adversarial_outlier_idx[j], adversarial_outlier_num[j] = \
            outlier_pc, outlier_idx, outlier_num

        pc_defended_recon = ae.get_reconstructions(pc_defended)
        pc_defended_recon_error = ae.get_loss_per_pc(pc_defended, source_pc).astype(adversarial_metrics.dtype)
        pc_defended_nre = np.divide(pc_defended_recon_error, source_ae_loss_ref)

        # adv_source_recon appears above as: adv_recon = ae.get_reconstructions(pc_input). it is already saved from the attack as adversarial_pc_recon
        adv_source_recon_error = ae.get_loss_per_pc(pc_input, source_pc).astype(adversarial_metrics.dtype)
        adv_source_nre = np.divide(adv_source_recon_error, source_ae_loss_ref)

        defended_pc_input[j] = pc_defended
        defended_pc_recon[j] = pc_defended_recon
        defended_source_recon_error[j] = pc_defended_recon_error
        defended_source_nre[j] = pc_defended_nre

        adversarial_source_recon_error[j] = adv_source_recon_error
        adversarial_source_nre[j] = adv_source_nre

    defense_metrics = np.concatenate([np.expand_dims(m, axis=-1) for m in
                                      [defended_source_recon_error, defended_source_nre,
                                       adversarial_source_recon_error, adversarial_source_nre]], axis=-1)

    # data above max number of outliers can be discarded
    outlier_num_max = adversarial_outlier_num.max()
    adversarial_outlier_points = adversarial_outlier_points[:, :, :outlier_num_max, :]
    adversarial_outlier_idx = adversarial_outlier_idx[:, :, :outlier_num_max]

    show = False
    if show:
        j, k = 0, 0
        plot_3d_point_cloud(adversarial_pc_input[j, k])
        plot_3d_point_cloud(adversarial_pc_recon[j, k])
        plot_3d_point_cloud(defended_pc_input[j, k])
        plot_3d_point_cloud(defended_pc_recon[j, k])

    # save results
    np.save(osp.join(save_dir, 'adversarial_critical_points'), adversarial_outlier_points)
    np.save(osp.join(save_dir, 'adversarial_critical_idx'), adversarial_outlier_idx)
    np.save(osp.join(save_dir, 'adversarial_critical_num'), adversarial_outlier_num)
    np.save(osp.join(save_dir, 'defended_pc_input'), defended_pc_input)
    np.save(osp.join(save_dir, 'defended_pc_recon'), defended_pc_recon)
    np.save(osp.join(save_dir, 'defense_metrics'), defense_metrics)

    ##########################################################################
    # check the influence of the defense on the original source point clouds #
    ##########################################################################
    knn_dists_source_pc = np.load(osp.join(save_dir_orig, 'knn_dists_source_pc.npy'))
    s_knn_dists_mean = np.mean(knn_dists_source_pc[:, :, :flags.num_knn_for_defense], axis=-1)

    s_outlier_points, s_outlier_idx, s_outlier_num, s_pc_defended = \
        get_outlier_pc_inlier_pc(source_pc, s_knn_dists_mean, flags.knn_dist_thresh)

    s_pc_defended_recon = ae.get_reconstructions(s_pc_defended)
    s_pc_defended_recon_error = ae.get_loss_per_pc(s_pc_defended, source_pc).astype(adversarial_metrics.dtype)
    s_pc_defended_nre = np.divide(s_pc_defended_recon_error, source_ae_loss_ref)

    s_pc_orig_recon_error = source_ae_loss_ref
    s_pc_orig_nre = np.ones_like(source_ae_loss_ref)

    defense_s_metrics = np.concatenate([np.expand_dims(m, axis=-1) for m in
                                        [s_pc_defended_recon_error, s_pc_defended_nre,
                                         s_pc_orig_recon_error, s_pc_orig_nre]], axis=-1)

    # data above max number of outliers can be discarded
    s_outlier_num_max = s_outlier_num.max()
    s_outlier_points = s_outlier_points[:, :s_outlier_num_max, :]
    s_outlier_idx = s_outlier_idx[:, :s_outlier_num_max]

    # save results
    np.save(osp.join(save_dir_orig, 'original_source_critical_points'), s_outlier_points)
    np.save(osp.join(save_dir_orig, 'original_critical_idx'), s_outlier_idx)
    np.save(osp.join(save_dir_orig, 'original_critical_num'), s_outlier_num)
    np.save(osp.join(save_dir_orig, 'defended_source_input'), s_pc_defended)
    np.save(osp.join(save_dir_orig, 'defended_source_recon'), s_pc_defended_recon)
    np.save(osp.join(save_dir_orig, 'defense_source_metrics'), defense_s_metrics)

    duration = time.time() - start_time
    print("Duration (minutes): %.2f" % (duration / 60.0))
