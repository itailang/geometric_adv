"""
Created on September 1st, 2020
@author: itailang
"""

# import system modules
import os
import os.path as osp
import sys
import time
from shutil import copy2
import argparse
import numpy as np
import matplotlib.pylab as plt

# add paths
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from src.autoencoder import Configuration as Conf
from src.in_out import create_dir
from src.adversary_utils import load_data, get_idx_for_correct_pred, get_quantity_for_targeted_untargeted_attack, write_defense_statistics_to_file
from src.general_utils import plot_3d_point_cloud, plot_heatmap_graph

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ae_folder', type=str, default='log/autoencoder_victim', help='Folder for loading a trained autoencoder model [default: log/autoencoder_victim]')
parser.add_argument("--attack_pc_idx", type=str, default='log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy', help="List of indices of point clouds for the attack")
parser.add_argument('--attack_folder', type=str, default='attack_res', help='Folder for loading attack data')
parser.add_argument("--do_sanity_checks", type=int, default=0, help="1: Do sanity checks, 0: Do not do sanity checks [default: 0]")
parser.add_argument("--output_folder_name", type=str, default='defense_critical_res', help="Output folder name")
parser.add_argument("--use_adversarial_data", type=int, default=1, help="1: evaluate defense on adversarial data, 0: evaluate defense on original source data")
parser.add_argument("--use_params_for_stat_file_name", type=int, default=0, help="1: use defense parameters for stat file name, 0: use default stat file name [default: 0]")
parser.add_argument('--save_graphs', type=int, default=0, help='1: Save statistics graphs, 0: Do not save statistics graphs [default: 0]')
parser.add_argument('--save_pc_plots', type=int, default=0, help='1: Save point cloud plots, 0: Do not save point cloud plots [default: 0]')
flags = parser.parse_args()

print('Evaluate defense flags:', flags)

# define basic parameters
top_out_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))  # Use to save Neural-Net check-points etc.
data_path = osp.join(top_out_dir, flags.ae_folder, 'eval')
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

attack_path = create_dir(osp.join(data_path, flags.attack_folder))

if flags.use_adversarial_data:
    output_path = create_dir(osp.join(attack_path, flags.output_folder_name))
else:
    output_path = create_dir(osp.join(attack_path, flags.output_folder_name + '_orig'))

# load attack configuration
conf = Conf.load(osp.join(attack_path, 'attack_configuration'))

# load data
point_clouds, latent_vectors, reconstructions, pc_classes, slice_idx, ae_loss = \
    load_data(data_path, files, ['point_clouds_test_set', 'latent_vectors_test_set', 'reconstructions_test_set',
                                 'pc_classes', 'slice_idx_test_set', 'ae_loss_test_set'])

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

classes_for_attack = conf.class_names
classes_for_target = conf.class_names

# load configuration
def_conf = Conf.load(osp.join(output_path, 'defense_configuration'))

# log files
over_classes_dir = create_dir(osp.join(output_path, 'over_classes'))
if flags.use_params_for_stat_file_name and def_conf.exists_and_is_not_none('num_knn_for_defense') and def_conf.exists_and_is_not_none('knn_dist_thresh'):
    ftar_name = 'targeted_attacks_k_%d_th_%.2f.txt' % (def_conf.num_knn_for_defense, def_conf.knn_dist_thresh)
    funtar_name = 'untargeted_attacks_k_%d_th_%.2f.txt' % (def_conf.num_knn_for_defense, def_conf.knn_dist_thresh)
else:
    ftar_name = 'targeted_attacks.txt'
    funtar_name = 'untargeted_attacks.txt'

ftar = open(osp.join(over_classes_dir, ftar_name), 'w', 1)
funtar = open(osp.join(over_classes_dir, funtar_name), 'w', 1)

def_source_chamfer_at_norm_min_best_target_class_list = []
def_source_nre_at_norm_min_best_target_class_list = []
adv_source_chamfer_at_norm_min_best_target_class_list = []
adv_source_nre_at_norm_min_best_target_class_list = []

def_source_chamfer_at_norm_min_per_target_class_list = []
def_source_nre_at_norm_min_per_target_class_list = []
adv_source_chamfer_at_norm_min_per_target_class_list = []
adv_source_nre_at_norm_min_per_target_class_list = []

best_attacks_path_list = []

source_target_norm_min_mat_list = []
rows_label_list = []

# evaluate the attack
for i in range(len(pc_classes)):
    pc_class_name = pc_classes[i]
    if pc_class_name not in classes_for_attack:
        continue

    print('evaluate shape class %s (%d out of %d classes) ' % (pc_class_name, i+1, len(pc_classes)))
    start_time = time.time()

    # load attack data
    load_dir_attack = osp.join(attack_path, pc_class_name)

    adversarial_pc_input = np.load(osp.join(load_dir_attack, 'adversarial_pc_input.npy'))
    adversarial_pc_recon = np.load(osp.join(load_dir_attack, 'adversarial_pc_recon.npy'))
    dist_weight_list = np.load(osp.join(load_dir_attack, 'dist_weight.npy'))
    source_target_norm_min_idx = np.load(osp.join(load_dir_attack, 'analysis_results', 'source_target_norm_min_idx.npy'))
    source_target_norm_min_per_target_class_idx = np.load(osp.join(load_dir_attack, 'analysis_results', 'source_target_norm_min_per_target_class_idx.npy'))
    source_target_norm_min_target_all_idx = np.load(osp.join(load_dir_attack, 'analysis_results', 'source_target_norm_min_target_all_idx.npy'))

    num_dist_weight = len(dist_weight_list)
    num_points = adversarial_pc_input.shape[2]

    # load defense data
    load_dir_defense = osp.join(output_path, pc_class_name)

    if flags.use_adversarial_data:
        adversarial_critical_points = np.load(osp.join(load_dir_defense, 'adversarial_critical_points.npy'))
        adversarial_critical_idx = np.load(osp.join(load_dir_defense, 'adversarial_critical_idx.npy'))
        adversarial_critical_num = np.load(osp.join(load_dir_defense, 'adversarial_critical_num.npy'))
        defended_pc_input = np.load(osp.join(load_dir_defense, 'defended_pc_input.npy'))
        defended_pc_recon = np.load(osp.join(load_dir_defense, 'defended_pc_recon.npy'))
        # defense metrics: defended_source_recon_error, defended_source_nre, adversarial_source_recon_error, adversarial_source_nre
        defense_metrics = np.load(osp.join(load_dir_defense, 'defense_metrics.npy'))

        adversarial_critical_points = np.vstack([adversarial_critical_points] * round((num_dist_weight / len(adversarial_critical_points))))
        adversarial_critical_idx = np.vstack([adversarial_critical_idx] * round((num_dist_weight / len(adversarial_critical_idx))))
        adversarial_critical_num = np.vstack([adversarial_critical_num] * round((num_dist_weight / len(adversarial_critical_num))))
        defended_pc_input = np.vstack([defended_pc_input] * round((num_dist_weight / len(defended_pc_input))))
        defended_pc_recon = np.vstack([defended_pc_recon] * round((num_dist_weight / len(defended_pc_recon))))
        defense_metrics = np.vstack([defense_metrics] * round((num_dist_weight / len(defense_metrics))))
    else:
        original_source_critical_points = np.load(osp.join(load_dir_defense, 'original_source_critical_points.npy'))
        original_critical_idx = np.load(osp.join(load_dir_defense, 'original_critical_idx.npy'))
        original_critical_num = np.load(osp.join(load_dir_defense, 'original_critical_num.npy'))
        defended_source_input = np.load(osp.join(load_dir_defense, 'defended_source_input.npy'))
        defended_source_recon = np.load(osp.join(load_dir_defense, 'defended_source_recon.npy'))
        # defense source metrics: defended_orig_source_recon_error, defended_orig_source_nre, orig_source_recon_error, orig_source_nre (which is ones)
        defense_source_metrics = np.load(osp.join(load_dir_defense, 'defense_source_metrics.npy'))

        adversarial_critical_points = np.vstack([np.expand_dims(original_source_critical_points, axis=0)] * num_dist_weight)
        adversarial_critical_idx = np.vstack([np.expand_dims(original_critical_idx, axis=0)] * num_dist_weight)
        adversarial_critical_num = np.vstack([np.expand_dims(original_critical_num, axis=0)] * num_dist_weight)
        defended_pc_input = np.vstack([np.expand_dims(defended_source_input, axis=0)] * num_dist_weight)
        defended_pc_recon = np.vstack([np.expand_dims(defended_source_recon, axis=0)] * num_dist_weight)
        defense_metrics = np.vstack([np.expand_dims(defense_source_metrics, axis=0)] * num_dist_weight)

    point_clouds_class = point_clouds[slice_idx[i]:slice_idx[i+1]]
    reconstructions_class = reconstructions[slice_idx[i]:slice_idx[i+1]]
    nn_idx_class = nn_idx[slice_idx[i]:slice_idx[i + 1], :]

    attack_idx_class = attack_pc_idx[i]
    point_clouds_for_attack = point_clouds_class[attack_idx_class]
    reconstructions_for_attack = reconstructions_class[attack_idx_class]
    nn_idx_for_attack = nn_idx_class[attack_idx_class]

    num_instance_for_attack = conf.num_pc_for_attack
    num_attacks = adversarial_pc_input.shape[1]
    num_attack_per_instance = num_attacks/num_instance_for_attack

    # prepare indices of source and target shapes
    idx_range = np.arange(num_instance_for_attack)
    source_idx_reshape = np.tile(np.expand_dims(idx_range, axis=1), [1, int(num_attack_per_instance)])

    target_class_idx = [np.where(pc_classes == name)[0][0] for name in classes_for_target if name != pc_class_name]
    target_class_name = np.array([name for name in classes_for_target if name != pc_class_name])
    num_target_classes = len(target_class_name)
    target_idx_reshape = np.tile(np.tile(idx_range, num_target_classes), [num_instance_for_attack, 1])

    # split defense metrics
    defended_source_recon_error, defended_source_nre, adversarial_source_recon_error, adversarial_source_nre = \
        np.split(defense_metrics, defense_metrics.shape[-1], axis=-1)

    defended_source_recon_error = np.squeeze(defended_source_recon_error, axis=2)
    defended_source_nre = np.squeeze(defended_source_nre, axis=2)
    adversarial_source_recon_error = np.squeeze(adversarial_source_recon_error, axis=2)
    adversarial_source_nre = np.squeeze(adversarial_source_nre, axis=2)

    # quantities for defense for targeted and untargeted attack (defended source chamfer, defended source nre, adversarial source chamfer, adversarial source nre)
    def_source_chamfer_at_norm_min_reshape, def_source_chamfer_at_norm_min_per_target_class, def_s_chamfer_best = \
        get_quantity_for_targeted_untargeted_attack(
            defended_source_recon_error, source_target_norm_min_idx, source_target_norm_min_per_target_class_idx, source_target_norm_min_target_all_idx)

    def_source_nre_at_norm_min_reshape, def_source_nre_at_norm_min_per_target_class, def_s_nre_best = \
        get_quantity_for_targeted_untargeted_attack(
            defended_source_nre, source_target_norm_min_idx, source_target_norm_min_per_target_class_idx, source_target_norm_min_target_all_idx)

    adv_source_chamfer_at_norm_min_reshape, adv_source_chamfer_at_norm_min_per_target_class, adv_s_chamfer_best = \
        get_quantity_for_targeted_untargeted_attack(
            adversarial_source_recon_error, source_target_norm_min_idx, source_target_norm_min_per_target_class_idx, source_target_norm_min_target_all_idx)

    adv_source_nre_at_norm_min_reshape, adv_source_nre_at_norm_min_per_target_class, adv_s_nre_best = \
        get_quantity_for_targeted_untargeted_attack(
            adversarial_source_nre, source_target_norm_min_idx, source_target_norm_min_per_target_class_idx, source_target_norm_min_target_all_idx)

    def_source_chamfer_at_norm_min_per_target_class_list.append(def_source_chamfer_at_norm_min_per_target_class)
    def_source_nre_at_norm_min_per_target_class_list.append(def_source_nre_at_norm_min_per_target_class)
    adv_source_chamfer_at_norm_min_per_target_class_list.append(adv_source_chamfer_at_norm_min_per_target_class)
    adv_source_nre_at_norm_min_per_target_class_list.append(adv_source_nre_at_norm_min_per_target_class)

    def_source_chamfer_at_norm_min_best_target_class_list.append(def_s_chamfer_best)
    def_source_nre_at_norm_min_best_target_class_list.append(def_s_nre_best)
    adv_source_chamfer_at_norm_min_best_target_class_list.append(adv_s_chamfer_best)
    adv_source_nre_at_norm_min_best_target_class_list.append(adv_s_nre_best)

    # matrices for heatmap plots
    def_source_chamfer_at_norm_min_mat = np.insert(def_source_chamfer_at_norm_min_reshape, i * conf.num_pc_for_target, np.zeros([conf.num_pc_for_target, num_instance_for_attack]), axis=1)
    def_source_nre_at_norm_min_mat = np.insert(def_source_nre_at_norm_min_reshape, i * conf.num_pc_for_target, np.ones([conf.num_pc_for_target, num_instance_for_attack]), axis=1)
    adv_source_chamfer_at_norm_min_mat = np.insert(adv_source_chamfer_at_norm_min_reshape, i * conf.num_pc_for_target, np.zeros([conf.num_pc_for_target, num_instance_for_attack]), axis=1)
    adv_source_nre_at_norm_min_mat = np.insert(adv_source_nre_at_norm_min_reshape, i * conf.num_pc_for_target, np.ones([conf.num_pc_for_target, num_instance_for_attack]), axis=1)

    def_source_chamfer_targeted_mat = np.insert(def_source_chamfer_at_norm_min_per_target_class, i * 1, np.zeros([1, num_instance_for_attack]), axis=1)
    def_source_nre_targeted_mat = np.insert(def_source_nre_at_norm_min_per_target_class, i * 1, np.zeros([1, num_instance_for_attack]), axis=1)
    adv_source_chamfer_targeted_mat = np.insert(adv_source_chamfer_at_norm_min_per_target_class, i * 1, np.zeros([1, num_instance_for_attack]), axis=1)
    adv_source_nre_targeted_mat = np.insert(adv_source_nre_at_norm_min_per_target_class, i * 1, np.zeros([1, num_instance_for_attack]), axis=1)

    ##################
    # heatmap graphs #
    ##################
    if flags.save_graphs:
        save_dir_graphs = create_dir(osp.join(load_dir_defense, 'analysis_results', 'stats'))

        columns_name_insert = np.insert(target_class_name, i, pc_class_name)
        columns_name = np.hstack([[columns_name_insert[j]] * num_instance_for_attack for j in range(num_target_classes + 1)])
        columns_idx = np.tile(idx_range, 1 + num_target_classes)
        columns_label = ['%s_%d' % (s, d) for s, d in zip(columns_name, columns_idx)]

        rows_label = ['%s_%d' % (s, d) for s, d in zip([pc_class_name] * num_instance_for_attack, idx_range)]
        rows_label_list.append(rows_label)

        # quantities for targeted attack
        plot_heatmap_graph(def_source_chamfer_targeted_mat, rows_label, columns_name_insert, pc_class_name, 'Target Class', 'Source Index',
                           '.5f', osp.join(save_dir_graphs, 'targeted_def_source_re.png'), (len(columns_name_insert), len(rows_label)))
        plot_heatmap_graph(def_source_nre_targeted_mat, rows_label, columns_name_insert, pc_class_name, 'Target Class', 'Source Index',
                           '.2f', osp.join(save_dir_graphs, 'targeted_def_source_nre.png'), (len(columns_name_insert), len(rows_label)))
        plot_heatmap_graph(adv_source_chamfer_targeted_mat, rows_label, columns_name_insert, pc_class_name, 'Target Class', 'Source Index',
                           '.5f', osp.join(save_dir_graphs, 'targeted_adv_source_re.png'), (len(columns_name_insert), len(rows_label)))
        plot_heatmap_graph(adv_source_nre_targeted_mat, rows_label, columns_name_insert, pc_class_name, 'Target Class', 'Source Index',
                           '.2f', osp.join(save_dir_graphs, 'targeted_adv_source_nre.png'), (len(columns_name_insert), len(rows_label)))

    # targeted attacks
    ftar.write('Shape class: %s\n' % pc_class_name)
    ftar.write('--------------------------------------\n')
    for j in range(num_instance_for_attack):
        for k in range(num_target_classes):
            t_c_idx = target_class_idx[k]
            t_c_name = target_class_name[k]
            best_t_idx = source_target_norm_min_per_target_class_idx[j, k]

            def_s_chamfer, def_s_nre, adv_s_chamfer, adv_s_nre = def_source_chamfer_at_norm_min_per_target_class[j, k], \
                                                                 def_source_nre_at_norm_min_per_target_class[j, k], \
                                                                 adv_source_chamfer_at_norm_min_per_target_class[j, k], \
                                                                 adv_source_nre_at_norm_min_per_target_class[j, k]

            attack_name = 'def_%s_%d_target_%s_%d' % (pc_class_name, j, t_c_name, best_t_idx)
            spaces = ''.join([' '] * (40 - len(attack_name)))
            funtar.write('%s%stra T-RE: %.5f   tra T-NRE: %.2f   adv T-RE: %.5f   adv T-NRE: %.2f\n' %
                         (attack_name, spaces, def_s_chamfer, def_s_nre, adv_s_chamfer, adv_s_nre))

            ############
            # pc plots #
            ############
            if flags.save_pc_plots:
                pc_t_class = point_clouds[slice_idx[t_c_idx]:slice_idx[t_c_idx + 1]]
                recon_t_class = reconstructions[slice_idx[t_c_idx]:slice_idx[t_c_idx + 1]]
                nn_t_class = nn_idx_for_attack[:, slice_idx[t_c_idx]:slice_idx[t_c_idx + 1]].copy()

                if correct_pred is not None:
                    nn_t_class = get_idx_for_correct_pred(nn_t_class, correct_pred, slice_idx, t_c_idx)

                attack_idx_t_class = nn_t_class[j, :conf.num_pc_for_target]

                pc_t_class_for_attack = pc_t_class[attack_idx_t_class]
                recon_t_class_for_attack = recon_t_class[attack_idx_t_class]

                source_pc = point_clouds_for_attack[j]
                target_pc = pc_t_class_for_attack[best_t_idx]

                source_pc_recon = reconstructions_for_attack[j]
                target_pc_recon = recon_t_class_for_attack[best_t_idx]

                best_t_idx_flatten = j * num_attack_per_instance + k * conf.num_pc_for_target + best_t_idx
                best_dist_weight_idx = source_target_norm_min_idx[int(best_t_idx_flatten)]
                adv_pc_input = adversarial_pc_input[best_dist_weight_idx, int(best_t_idx_flatten)]
                adv_pc_recon = adversarial_pc_recon[best_dist_weight_idx, int(best_t_idx_flatten)]

                if not flags.use_adversarial_data:
                    adv_pc_input = source_pc
                    adv_pc_recon = source_pc_recon

                adv_critical_idx = adversarial_critical_idx[best_dist_weight_idx, int(best_t_idx_flatten)]
                adv_critical_num = adversarial_critical_num[best_dist_weight_idx, int(best_t_idx_flatten)]
                adv_critical_points = adversarial_critical_points[best_dist_weight_idx, int(best_t_idx_flatten)]

                # sanity check
                if flags.do_sanity_checks:
                    assert np.array_equal(adv_pc_input[adv_critical_idx[:adv_critical_num]], adv_critical_points[:adv_critical_num]), \
                        'pc at critical idx should be equal to critical points!'

                adv_pc_input_point_color = np.array(['b'] * num_points)
                adv_pc_input_point_color[adv_critical_idx[:adv_critical_num]] = 'r'

                def_pc_input = defended_pc_input[best_dist_weight_idx, int(best_t_idx_flatten), :-adv_critical_num]
                def_pc_recon = defended_pc_recon[best_dist_weight_idx, int(best_t_idx_flatten)]

                azim = -40
                elev = 20
                show = False

                save_dir_pc_plots = create_dir(osp.join(load_dir_defense, 'analysis_results', 'pc_plots'))

                save_path = osp.join(save_dir_pc_plots, 'def_%s_%d_target_%s_%d_inputs.png' % (pc_class_name, j, t_c_name, best_t_idx))
                fig = plt.figure(figsize=(15, 5))
                ax = fig.add_subplot(131, projection='3d')
                plot_3d_point_cloud(source_pc, azim=azim, elev=elev, show=show, axis=ax)
                ax = fig.add_subplot(132, projection='3d')
                plot_3d_point_cloud(adv_pc_input, azim=azim, elev=elev, show=show, axis=ax, c=adv_pc_input_point_color)
                ax = fig.add_subplot(133, projection='3d')
                plot_3d_point_cloud(def_pc_input, azim=azim, elev=elev, show=show, axis=ax)
                plt.savefig(save_path)
                plt.close()

                save_path = osp.join(save_dir_pc_plots, 'def_%s_%d_target_%s_%d_recons.png' % (pc_class_name, j, t_c_name, best_t_idx))
                fig = plt.figure(figsize=(15, 5))
                ax = fig.add_subplot(131, projection='3d')
                plot_3d_point_cloud(source_pc_recon, azim=azim, elev=elev, show=show, axis=ax)
                ax = fig.add_subplot(132, projection='3d')
                plot_3d_point_cloud(adv_pc_recon, azim=azim, elev=elev, show=show, axis=ax)
                ax = fig.add_subplot(133, projection='3d')
                plot_3d_point_cloud(def_pc_recon, azim=azim, elev=elev, show=show, axis=ax)
                plt.savefig(save_path)
                plt.close()

    ftar.write('\n')

    # untargeted attacks
    funtar.write('Shape class: %s\n' % pc_class_name)
    funtar.write('--------------------------------------\n')
    for j in range(num_instance_for_attack):
        c_idx = source_target_norm_min_target_all_idx[j]
        best_t_idx = source_target_norm_min_per_target_class_idx[j, c_idx]
        t_c_name = target_class_name[c_idx]

        def_s_chamfer, def_s_nre, adv_s_chamfer, adv_s_nre = def_s_chamfer_best[j], def_s_nre_best[j], adv_s_chamfer_best[j], adv_s_nre_best[j]

        attack_name = 'def_%s_%d_target_%s_%d' % (pc_class_name, j, t_c_name, best_t_idx)
        spaces = ''.join([' '] * (40 - len(attack_name)))
        funtar.write('%s%sdef S-RE: %.5f   def S-NRE: %.2f   adv S-RE: %.5f   adv S-NRE: %.2f\n' %
                     (attack_name, spaces, def_s_chamfer, def_s_nre, adv_s_chamfer, adv_s_nre))

        if flags.save_pc_plots:
            inputs_path = osp.join(save_dir_pc_plots, 'def_%s_%d_target_%s_%d_inputs.png' % (pc_class_name, j, t_c_name, best_t_idx))
            recons_path = osp.join(save_dir_pc_plots, 'def_%s_%d_target_%s_%d_recons.png' % (pc_class_name, j, t_c_name, best_t_idx))
            best_attacks_path_list.append(inputs_path)
            best_attacks_path_list.append(recons_path)

    funtar.write('\n')

    duration = time.time() - start_time
    print("Duration (minutes): %.2f" % (duration / 60.0))

ftar.close()
funtar.close()

# copy untargeted attacks to over_classes directory
if flags.save_pc_plots:
    dest_dir = create_dir(osp.join(over_classes_dir, 'untargeted_attacks'))
    for f in best_attacks_path_list:
        if osp.exists(f):
            copy2(f, dest_dir)

# write statistics to log file
if flags.use_params_for_stat_file_name and def_conf.exists_and_is_not_none('num_knn_for_defense') and def_conf.exists_and_is_not_none('knn_dist_thresh'):
    fout_name = 'eval_stats_k_%d_th_%.2f.txt' % (def_conf.num_knn_for_defense, def_conf.knn_dist_thresh)
else:
    fout_name = 'eval_stats.txt'

fout = open(osp.join(over_classes_dir, fout_name), 'w', 1)

fout.write('Statistics for targeted attack\n')
fout.write('--------------------------------------\n')
write_defense_statistics_to_file(fout, classes_for_attack,
                                 def_source_chamfer_at_norm_min_per_target_class_list, def_source_nre_at_norm_min_per_target_class_list,
                                 adv_source_chamfer_at_norm_min_per_target_class_list, adv_source_nre_at_norm_min_per_target_class_list)

fout.write('\n')
fout.write('Statistics for untargeted attack\n')
fout.write('--------------------------------------\n')
write_defense_statistics_to_file(fout, classes_for_attack,
                                 def_source_chamfer_at_norm_min_best_target_class_list, def_source_nre_at_norm_min_best_target_class_list,
                                 adv_source_chamfer_at_norm_min_best_target_class_list, adv_source_nre_at_norm_min_best_target_class_list)

fout.close()
