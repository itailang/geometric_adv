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
from src.in_out import create_dir
from src.adversary_utils import load_data, get_quantity_for_targeted_untargeted_attack, write_classification_statistics_to_file, prepare_data_for_attack
from src.general_utils import plot_heatmap_graph, plot_3d_point_cloud

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--classifier_folder', type=str, default='log/pointnet', help='Folder of the classifier to be used [default: log/pointnet]')
parser.add_argument('--data_type', type=str, default='adversarial', help='Data type to be classified [default: adversarial]')
parser.add_argument('--ae_folder', type=str, default='log/autoencoder_victim', help='Folder for loading a trained autoencoder model [default: log/autoencoder_victim]')
parser.add_argument("--attack_pc_idx", type=str, default='log/autoencoder_victim/eval/sel_idx_rand_100_test_set_13l.npy', help="List of indices of point clouds for the attack")
parser.add_argument('--attack_folder', type=str, default='attack_res', help='Folder for loading attack data')
parser.add_argument('--defense_folder', type=str, default='defense_critical_res', help='Folder for loading defense data')
parser.add_argument("--output_folder_name", type=str, default='classifier_res', help="Output folder name")
parser.add_argument('--save_graphs', type=int, default=0, help='1: Save statistics graphs, 0: Do not save statistics graphs [default: 0]')
flags = parser.parse_args()

print('Evaluate classifier flags:', flags)

assert flags.data_type in ['target', 'adversarial', 'source', 'before_defense', 'after_defense'], 'wrong data_type: %s.' % flags.data_type

# define basic parameters
top_out_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))  # Use to save Neural-Net check-points etc.
data_path = osp.join(top_out_dir, flags.ae_folder, 'eval')
files = [f for f in os.listdir(data_path) if osp.isfile(osp.join(data_path, f))]

attack_path = create_dir(osp.join(data_path, flags.attack_folder))

if flags.data_type == 'target':
    output_path = create_dir(osp.join(attack_path, flags.output_folder_name + '_orig'))
elif flags.data_type == 'adversarial':
    output_path = create_dir(osp.join(attack_path, flags.output_folder_name))
elif flags.data_type == 'source':
    output_path = create_dir(osp.join(attack_path, flags.defense_folder, flags.output_folder_name + '_orig'))
elif flags.data_type == 'before_defense':
    adversarial_data_path = create_dir(osp.join(attack_path, flags.output_folder_name))
    output_path = create_dir(osp.join(attack_path, flags.defense_folder, flags.output_folder_name))
elif flags.data_type == 'after_defense':
    output_path = create_dir(osp.join(attack_path, flags.defense_folder, flags.output_folder_name))

# load attack configuration
conf = Conf.load(osp.join(attack_path, 'attack_configuration'))

# load data
point_clouds, latent_vectors, reconstructions, pc_classes, slice_idx, pc_labels, pc_pred_labels = \
    load_data(data_path, files, ['point_clouds_test_set', 'latent_vectors_test_set', 'reconstructions_test_set',
                                 'pc_classes', 'slice_idx_test_set', 'pc_label_test_set', 'pc_pred_labels_test_set'])

nn_idx_dict = {'latent_nn': 'latent_nn_idx_test_set', 'chamfer_nn_complete': 'chamfer_nn_idx_complete_test_set'}
nn_idx = load_data(data_path, files, [nn_idx_dict[conf.target_pc_idx_type]])

correct_pred = None
if conf.correct_pred_only:
    correct_pred = (pc_labels == pc_pred_labels)

# load indices for attack
attack_pc_idx = np.load(osp.join(top_out_dir, flags.attack_pc_idx))
attack_pc_idx = attack_pc_idx[:, :conf.num_pc_for_attack]

classes_for_attack = conf.class_names
classes_for_target = conf.class_names

# log files
over_classes_dir = create_dir(osp.join(output_path, 'over_classes'))
if flags.data_type == 'before_defense':
    ftar_name = 'targeted_attacks_before_defense.txt'
    funtar_name = 'untargeted_attacks_before_defense.txt'
elif flags.data_type == 'after_defense':
    ftar_name = 'targeted_attacks_after_defense.txt'
    funtar_name = 'untargeted_attacks_after_defense.txt'
else:
    ftar_name = 'targeted_attacks.txt'
    funtar_name = 'untargeted_attacks.txt'

ftar = open(osp.join(over_classes_dir, ftar_name), 'w', 1)
funtar = open(osp.join(over_classes_dir, funtar_name), 'w', 1)

recon_cls_at_norm_min_best_target_class_list = []

recon_cls_at_norm_min_per_target_class_list = []

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

    # load classification data and pc labels
    source_recon_ref, target_recon_ref = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, reconstructions, slice_idx, attack_pc_idx, conf.num_pc_for_target, conf.target_pc_idx_type, nn_idx, correct_pred)
    source_pc_labels, target_pc_labels = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, pc_labels, slice_idx, attack_pc_idx, conf.num_pc_for_target, conf.target_pc_idx_type, nn_idx, correct_pred)
    source_pc_pred_labels, target_pc_pred_labels = prepare_data_for_attack(pc_classes, [pc_class_name], classes_for_target, pc_pred_labels, slice_idx, attack_pc_idx, conf.num_pc_for_target, conf.target_pc_idx_type, nn_idx, correct_pred)

    source_pc_labels = source_pc_labels.reshape(-1)
    target_pc_labels = target_pc_labels.reshape(-1)
    source_pc_pred_labels = source_pc_pred_labels.reshape(-1)
    target_pc_pred_labels = target_pc_pred_labels.reshape(-1)

    load_dir_classifier = osp.join(output_path, pc_class_name)

    if flags.data_type == 'target':
        target_recon_ref_cls_correct = np.equal(target_pc_pred_labels, target_pc_labels)
        pc_recon_cls_correct = np.vstack([np.expand_dims(target_recon_ref_cls_correct, axis=0)] * num_dist_weight)
    elif flags.data_type == 'adversarial':
        adversarial_pc_recon_pred = np.load(osp.join(load_dir_classifier, 'adversarial_pc_recon_pred.npy'))
        pc_label = np.vstack([target_pc_labels] * len(adversarial_pc_recon_pred))
        pc_recon_cls_correct = np.equal(adversarial_pc_recon_pred, pc_label)
        pc_recon_cls_correct = np.vstack([pc_recon_cls_correct] * round((num_dist_weight / len(pc_recon_cls_correct))))
    elif flags.data_type == 'source':
        source_recon_ref_cls_correct = np.equal(source_pc_pred_labels, source_pc_labels)
        pc_recon_cls_correct = np.vstack([np.expand_dims(source_recon_ref_cls_correct, axis=0)] * num_dist_weight)
    elif flags.data_type == 'before_defense':
        adversarial_pc_recon_pred = np.load(osp.join(adversarial_data_path, pc_class_name, 'adversarial_pc_recon_pred.npy'))
        pc_label = np.vstack([source_pc_labels] * len(adversarial_pc_recon_pred))
        pc_recon_cls_correct = np.equal(adversarial_pc_recon_pred, pc_label)
        pc_recon_cls_correct = np.vstack([pc_recon_cls_correct] * round((num_dist_weight / len(pc_recon_cls_correct))))
    elif flags.data_type == 'after_defense':
        defense_on_adv = osp.exists(osp.join(load_dir_classifier, 'defended_pc_recon_pred.npy'))
        if defense_on_adv:
            defended_pc_recon_pred = np.load(osp.join(load_dir_classifier, 'defended_pc_recon_pred.npy'))  # defense on adversarial input
        else:
            defended_pc_recon_pred = np.load(osp.join(load_dir_classifier, 'defended_source_recon_pred.npy'))  # defense on clean input
            defended_pc_recon_pred = np.expand_dims(defended_pc_recon_pred, axis=0)
        pc_label = np.vstack([source_pc_labels] * len(defended_pc_recon_pred))
        defended_pc_recon_cls_correct = np.equal(defended_pc_recon_pred, pc_label)
        defended_pc_recon_cls_correct = np.vstack([defended_pc_recon_cls_correct] * round((num_dist_weight / len(defended_pc_recon_cls_correct))))
        pc_recon_cls_correct = defended_pc_recon_cls_correct

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

    # quantity for classification for targeted and untargeted attack (classification correctness)
    recon_cls_at_norm_min_reshape, recon_cls_at_norm_min_per_target_class, r_cls_best = \
        get_quantity_for_targeted_untargeted_attack(
            pc_recon_cls_correct, source_target_norm_min_idx, source_target_norm_min_per_target_class_idx, source_target_norm_min_target_all_idx)

    recon_cls_at_norm_min_per_target_class_list.append(recon_cls_at_norm_min_per_target_class)

    recon_cls_at_norm_min_best_target_class_list.append(r_cls_best)

    # matrices for heatmap plots
    recon_cls_at_norm_min_mat = np.insert(recon_cls_at_norm_min_reshape, i * conf.num_pc_for_target, np.ones([conf.num_pc_for_target, num_instance_for_attack]), axis=1)

    adv_recon_cls_targeted_mat = np.insert(recon_cls_at_norm_min_per_target_class, i * 1, np.ones([1, num_instance_for_attack]), axis=1)

    ##################
    # heatmap graphs #
    ##################
    if flags.save_graphs:
        save_dir_graphs = create_dir(osp.join(load_dir_classifier, 'analysis_results', 'stats'))

        columns_name_insert = np.insert(target_class_name, i, pc_class_name)
        columns_name = np.hstack([[columns_name_insert[j]] * num_instance_for_attack for j in range(num_target_classes + 1)])
        columns_idx = np.tile(idx_range, 1 + num_target_classes)
        columns_label = ['%s_%d' % (s, d) for s, d in zip(columns_name, columns_idx)]

        rows_label = ['%s_%d' % (s, d) for s, d in zip([pc_class_name] * num_instance_for_attack, idx_range)]
        rows_label_list.append(rows_label)

        # quantities for targeted attack
        if flags.data_type == 'before_defense':
            plot_heatmap_graph(adv_recon_cls_targeted_mat, rows_label, columns_name_insert, pc_class_name, 'Target Class', 'Source Index',
                               '.2f', osp.join(save_dir_graphs, 'targeted_recon_cls_before_defense.png'), (len(columns_name_insert), len(rows_label)))
        elif flags.data_type == 'after_defense':
            plot_heatmap_graph(adv_recon_cls_targeted_mat, rows_label, columns_name_insert, pc_class_name, 'Target Class', 'Source Index',
                               '.2f', osp.join(save_dir_graphs, 'targeted_recon_cls_after_defense.png'), (len(columns_name_insert), len(rows_label)))
        else:
            plot_heatmap_graph(adv_recon_cls_targeted_mat, rows_label, columns_name_insert, pc_class_name, 'Target Class', 'Source Index',
                               '.2f', osp.join(save_dir_graphs, 'targeted_recon_cls.png'), (len(columns_name_insert), len(rows_label)))

    # targeted attacks
    ftar.write('Shape class: %s\n' % pc_class_name)
    ftar.write('--------------------------------------\n')
    for j in range(num_instance_for_attack):
        for k in range(num_target_classes):
            t_c_idx = target_class_idx[k]
            t_c_name = target_class_name[k]
            best_t_idx = source_target_norm_min_per_target_class_idx[j, k]

            r_cls = recon_cls_at_norm_min_per_target_class[j, k]

            attack_name = 'cls_%s_%d_target_%s_%d' % (pc_class_name, j, t_c_name, best_t_idx)
            spaces = ''.join([' '] * (40 - len(attack_name)))
            ftar.write('%s%saccuracy: %.4f\n' % (attack_name, spaces, r_cls))

    ftar.write('\n')

    # untargeted attacks
    funtar.write('Shape class: %s\n' % pc_class_name)
    funtar.write('--------------------------------------\n')
    for j in range(num_instance_for_attack):
        c_idx = source_target_norm_min_target_all_idx[j]
        best_t_idx = source_target_norm_min_per_target_class_idx[j, c_idx]
        t_c_name = target_class_name[c_idx]

        r_cls = r_cls_best[j]

        attack_name = 'cls_%s_%d_target_%s_%d' % (pc_class_name, j, t_c_name, best_t_idx)
        spaces = ''.join([' '] * (40 - len(attack_name)))
        funtar.write('%s%saccuracy: %.4f\n' % (attack_name, spaces, r_cls))

    funtar.write('\n')

    duration = time.time() - start_time
    print("Duration (minutes): %.2f" % (duration / 60.0))

ftar.close()
funtar.close()

# write statistics to log file
if flags.data_type == 'before_defense':
    fout_name = 'eval_stats_before_defense.txt'
elif flags.data_type == 'after_defense':
    fout_name = 'eval_stats_after_defense.txt'
else:
    fout_name = 'eval_stats.txt'

fout = open(osp.join(over_classes_dir, fout_name), 'w', 1)

fout.write('Statistics for targeted attack\n')
fout.write('--------------------------------------\n')
write_classification_statistics_to_file(fout, classes_for_attack, recon_cls_at_norm_min_per_target_class_list, flags.data_type)

fout.write('\n')
fout.write('Statistics for untargeted attack\n')
fout.write('--------------------------------------\n')
write_classification_statistics_to_file(fout, classes_for_attack, recon_cls_at_norm_min_best_target_class_list, flags.data_type)

fout.close()
