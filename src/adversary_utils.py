"""
Created on July 8th, 2020
@author: itailang
"""

import os.path as osp
import sys
import numpy as np

from src.general_utils import plot_3d_point_cloud


def load_data(data_path, file_list, base_name_list):
    data_list = [None] * len(base_name_list)

    for i, base_name in enumerate(base_name_list):
        file_name = [f for f in file_list if base_name in f][0]
        data_list[i] = np.load(osp.join(data_path, file_name))

    if len(data_list) == 1:
        data_list = data_list[0]

    return data_list


def prepare_data_for_attack(pc_classes, source_classes_for_attack, target_classes_for_attack, classes_data, slice_idx, attack_pc_idx, num_pc_for_target, target_pc_idx_type, nn_idx_mat, correct_pred):
    num_classes = len(pc_classes)

    source_data_list = []
    target_data_list = []

    for i in range(num_classes):
        source_class_name = pc_classes[i]
        if source_class_name not in source_classes_for_attack:
            continue

        source_attack_idx = attack_pc_idx[i]
        num_source_pc_for_attack = len(source_attack_idx)

        source_class_data = classes_data[slice_idx[i]:slice_idx[i + 1]]
        source_class_data_for_attack = source_class_data[source_attack_idx]

        num_attack_per_pc = 0
        target_data_for_attack_list = []

        for j in range(num_classes):
            target_class_name = pc_classes[j]
            if target_class_name not in target_classes_for_attack or target_class_name == source_class_name:
                continue

            nn_idx_s_class_t_class = nn_idx_mat[slice_idx[i]:slice_idx[i + 1], slice_idx[j]:slice_idx[j + 1]]
            nn_idx_s_for_attack_t_class = nn_idx_s_class_t_class[source_attack_idx].copy()

            if correct_pred is not None:
                nn_idx_s_for_attack_t_class = get_idx_for_correct_pred(nn_idx_s_for_attack_t_class, correct_pred, slice_idx, j)

            num_attack_per_pc += num_pc_for_target

            target_class_data = classes_data[slice_idx[j]:slice_idx[j + 1]]
            target_class_data_for_attack_list = []
            for s in range(num_source_pc_for_attack):
                target_attack_idx = nn_idx_s_for_attack_t_class[s, :num_pc_for_target]
                target_class_data_for_attack_curr = target_class_data[target_attack_idx]
                target_class_data_for_attack_list.append(np.expand_dims(target_class_data_for_attack_curr, axis=0))

            target_data_for_attack = np.vstack(target_class_data_for_attack_list)
            target_data_for_attack_list.append(target_data_for_attack)

        target_data_for_attack_concat = np.concatenate(target_data_for_attack_list, axis=1)
        old_shape = target_data_for_attack_concat.shape
        new_shape = [old_shape[0]*old_shape[1]] + [old_shape[n] for n in range(2, len(old_shape))]
        target_data_curr = np.reshape(target_data_for_attack_concat, new_shape)

        target_data_list.append(target_data_curr)

        source_data_curr = np.vstack([[source_class_data_for_attack[n]] * num_attack_per_pc for n in range(num_source_pc_for_attack)])
        source_data_list.append(source_data_curr)

        #plot_3d_point_cloud(source_data_curr[0])
        #plot_3d_point_cloud(target_data_curr[0])

    source_data = np.vstack(source_data_list)
    target_data = np.vstack(target_data_list)

    return source_data, target_data


def get_idx_for_correct_pred(nn_idx_s_for_attack_t_class, correct_pred, slice_idx, t_class_index):
    correct_pred_t_class = correct_pred[slice_idx[t_class_index]:slice_idx[t_class_index + 1]]
    correct_pred_t_idx = np.where(correct_pred_t_class)[0]

    for l in range(len(nn_idx_s_for_attack_t_class)):
        nn_idx_s = nn_idx_s_for_attack_t_class[l]
        nn_idx_s_correct_pred_t = np.array([idx for idx in nn_idx_s if idx in correct_pred_t_idx], dtype=nn_idx_s.dtype)
        nn_idx_s_for_attack_t_class[l, :len(nn_idx_s_correct_pred_t)] = nn_idx_s_correct_pred_t
        nn_idx_s_for_attack_t_class[l, len(nn_idx_s_correct_pred_t):] = nn_idx_s_correct_pred_t[0]

    return nn_idx_s_for_attack_t_class


def get_quantity_at_index(quantity_list, index):
    quantity_at_index_list = [np.zeros(quantity_list[i].shape[1:], dtype=quantity_list[i].dtype) for i in range(len(quantity_list))]

    for i, quantity in enumerate(quantity_list):
        quantity_at_index = quantity_at_index_list[i]
        for j in range(len(index)):
            quantity_at_index[j] = quantity[index[j], j]

    if len(quantity_at_index_list) == 1:
        quantity_at_index_list = quantity_at_index_list[0]

    return quantity_at_index_list


def get_quantity_at_index_per_target_class(quantity, per_target_class_idx):
    num_instance_for_attack, num_target_classes = per_target_class_idx.shape
    quantity_at_index_per_target_class = np.zeros([num_instance_for_attack, num_target_classes], dtype=quantity.dtype)
    num_pc_for_target = quantity.shape[1] / per_target_class_idx.shape[1]
    for k in range(num_target_classes):
        quantity_for_target_class = quantity[:, k * int(num_pc_for_target):(k + 1) * int(num_pc_for_target)]
        quantity_at_index_per_target_class[:, k] = get_quantity_at_index([quantity_for_target_class.T], per_target_class_idx[:, k])

    return quantity_at_index_per_target_class


def get_quantity_for_targeted_untargeted_attack(quantity, dist_weight_idx, targeted_idx, untargeted_idx):
    num_attacks = quantity.shape[1]
    num_instance_for_attack, num_target_classes = targeted_idx.shape
    num_attack_per_instance = num_attacks / num_instance_for_attack
    num_pc_for_target = num_attack_per_instance / num_target_classes

    # quantity for distance weight
    quantity_dist_weight = get_quantity_at_index([quantity], dist_weight_idx)
    quantity_dist_weight_reshape = quantity_dist_weight.reshape([num_instance_for_attack, int(num_attack_per_instance)])

    # quantity for targeted attack
    quantity_targeted = get_quantity_at_index_per_target_class(quantity_dist_weight_reshape, targeted_idx)

    # quantity for untargeted attack
    quantity_untargeted = np.zeros(num_instance_for_attack, dtype=quantity_targeted.dtype)
    for j in range(num_instance_for_attack):
        c_idx = untargeted_idx[j]
        best_t_idx = targeted_idx[j, c_idx]
        quantity_untargeted[j] = quantity_dist_weight_reshape[j, c_idx * int(num_pc_for_target) + best_t_idx]

    return quantity_dist_weight_reshape, quantity_targeted, quantity_untargeted


def get_outlier_pc_inlier_pc(point_clouds, knn_dists, knn_dist_thresh):
    num_pc, num_points, _ = point_clouds.shape

    outlier_pc = np.zeros_like(point_clouds)
    outlier_idx = np.zeros([num_pc, num_points], dtype=np.int16)
    outlier_num = np.zeros(num_pc, dtype=np.int16)
    inlier_pc = np.zeros_like(point_clouds)
    for l in range(num_pc):
        knn_dists_pc = knn_dists[l]

        outlier_idx_pc = np.where(knn_dists_pc > knn_dist_thresh)[0]
        outlier_num_pc = len(outlier_idx_pc)
        outlier_points_pc = point_clouds[l, outlier_idx_pc, :]

        outlier_idx[l, :outlier_num_pc] = outlier_idx_pc
        outlier_num[l] = outlier_num_pc

        outlier_pc[l, :outlier_num_pc] = outlier_points_pc
        if 0 < outlier_num_pc < num_points:
            outlier_pc[l, outlier_num_pc:] = outlier_points_pc[-1]  # duplication of last point does not change the latent vector due to global pooling

        inlier_idx_pc = np.where(knn_dists_pc <= knn_dist_thresh)[0]
        inlier_num_pc = len(inlier_idx_pc)
        inlier_points_pc = point_clouds[l, inlier_idx_pc, :]

        inlier_pc[l, :inlier_num_pc, :] = inlier_points_pc
        if 0 < inlier_num_pc < num_points:
            inlier_pc[l, inlier_num_pc:, :] = inlier_points_pc[-1]  # duplication of last point does not change the latent vector due to global pooling

    return outlier_pc, outlier_idx, outlier_num, inlier_pc


def write_attack_statistics_to_file(fout, classes_for_attack, source_target_norm_min_list,
                                    num_outlier_at_norm_min_list, source_chamfer_at_norm_min_list,
                                    target_chamfer_at_norm_min_list, target_nre_at_norm_min_list):
    fout.write('Shape\t\tAttack\t\tAdv\t\tAdv\t\tAdv\t\tAdv\n')
    fout.write('Class\t\tScore\t\t#OS\t\tS-CD\t\tT-RE\t\tT-NRE\n')
    fout.write('\n')

    for c, pc_class_name in enumerate(classes_for_attack):
        source_target_norm_min_mean = source_target_norm_min_list[c].mean()
        num_outlier_at_norm_min_mean = num_outlier_at_norm_min_list[c].mean()
        source_chamfer_at_norm_min_mean = source_chamfer_at_norm_min_list[c].mean()
        target_chamfer_at_norm_min_mean = target_chamfer_at_norm_min_list[c].mean()
        target_nre_at_norm_min_mean = target_nre_at_norm_min_list[c].mean()

        spaces = ''.join([' '] * (16 - len(pc_class_name)))
        fout.write('%s%s%.5f\t\t%03d\t\t%.5f\t\t%.5f\t\t%.2f\n' %
                   (pc_class_name, spaces, source_target_norm_min_mean,
                    num_outlier_at_norm_min_mean, source_chamfer_at_norm_min_mean,
                    target_chamfer_at_norm_min_mean, target_nre_at_norm_min_mean))

    # over classes statistics
    source_target_norm_min_classes = np.vstack(source_target_norm_min_list)
    source_target_norm_min_classes_mean = source_target_norm_min_classes.mean()
    num_outlier_at_norm_min_classes = np.vstack(num_outlier_at_norm_min_list)
    num_outlier_at_norm_min_classes_mean = num_outlier_at_norm_min_classes.mean()
    source_chamfer_at_norm_min_classes = np.vstack(source_chamfer_at_norm_min_list)
    source_chamfer_at_norm_min_classes_mean = source_chamfer_at_norm_min_classes.mean()
    target_chamfer_at_norm_min_classes = np.vstack(target_chamfer_at_norm_min_list)
    target_chamfer_at_norm_min_classes_mean = target_chamfer_at_norm_min_classes.mean()
    target_nre_at_norm_min_classes = np.vstack(target_nre_at_norm_min_list)
    target_nre_at_norm_min_classes_mean = target_nre_at_norm_min_classes.mean()

    fout.write('\n')
    pc_class_name = 'over classes'
    spaces = ''.join([' '] * (16 - len(pc_class_name)))
    fout.write('%s%s%.5f\t\t%03d\t\t%.5f\t\t%.5f\t\t%.2f\n' %
               (pc_class_name, spaces, source_target_norm_min_classes_mean,
                num_outlier_at_norm_min_classes_mean, source_chamfer_at_norm_min_classes_mean,
                target_chamfer_at_norm_min_classes_mean, target_nre_at_norm_min_classes_mean))


def write_defense_statistics_to_file(fout, classes_for_attack,
                                     def_source_chamfer_at_norm_min_list, def_source_nre_at_norm_min_list,
                                     adv_source_chamfer_at_norm_min_list, adv_source_nre_at_norm_min_list):
    fout.write('Shape\t\tDef\t\tDef\t\tAdv\t\tAdv\n')
    fout.write('Class\t\tS-RE\t\tS-NRE\t\tS-RE\t\tS-NRE\n')
    fout.write('\n')

    for c, pc_class_name in enumerate(classes_for_attack):
        def_source_chamfer_at_norm_min_mean = def_source_chamfer_at_norm_min_list[c].mean()
        def_source_nre_at_norm_min_mean = def_source_nre_at_norm_min_list[c].mean()
        adv_source_chamfer_at_norm_min_mean = adv_source_chamfer_at_norm_min_list[c].mean()
        adv_source_nre_at_norm_min_mean = adv_source_nre_at_norm_min_list[c].mean()

        spaces = ''.join([' '] * (16 - len(pc_class_name)))
        fout.write('%s%s%.5f\t\t%.2f\t\t%.5f\t\t%.2f\n' %
                   (pc_class_name, spaces,
                    def_source_chamfer_at_norm_min_mean, def_source_nre_at_norm_min_mean,
                    adv_source_chamfer_at_norm_min_mean, adv_source_nre_at_norm_min_mean))

    # over classes statistics
    def_source_chamfer_at_norm_min_classes = np.vstack(def_source_chamfer_at_norm_min_list)
    def_source_chamfer_at_norm_min_classes_mean = def_source_chamfer_at_norm_min_classes .mean()
    def_source_nre_at_norm_min_classes = np.vstack(def_source_nre_at_norm_min_list)
    def_source_nre_at_norm_min_classes_mean = def_source_nre_at_norm_min_classes.mean()
    avd_source_chamfer_at_norm_min_classes = np.vstack(adv_source_chamfer_at_norm_min_list)
    avd_source_chamfer_at_norm_min_classes_mean = avd_source_chamfer_at_norm_min_classes.mean()
    adv_source_nre_at_norm_min_classes = np.vstack(adv_source_nre_at_norm_min_list)
    adv_source_nre_at_norm_min_classes_mean = adv_source_nre_at_norm_min_classes.mean()

    fout.write('\n')
    pc_class_name = 'over classes'
    spaces = ''.join([' '] * (16 - len(pc_class_name)))
    fout.write('%s%s%.5f\t\t%.2f\t\t%.5f\t\t%.2f\n' %
               (pc_class_name, spaces,
                def_source_chamfer_at_norm_min_classes_mean, def_source_nre_at_norm_min_classes_mean,
                avd_source_chamfer_at_norm_min_classes_mean, adv_source_nre_at_norm_min_classes_mean))


def write_transfer_statistics_to_file(fout, classes_for_attack,
                                      tra_target_chamfer_at_norm_min_list, tra_target_nre_at_norm_min_list,
                                      adv_target_chamfer_at_norm_min_list, adv_target_nre_at_norm_min_list):
    fout.write('Shape\t\tTra\t\tTra\t\tAdv\t\tAdv\n')
    fout.write('Class\t\tT-RE\t\tT-NRE\t\tT-RE\t\tT-NRE\n')
    fout.write('\n')

    for c, pc_class_name in enumerate(classes_for_attack):
        tra_target_chamfer_at_norm_min_mean = tra_target_chamfer_at_norm_min_list[c].mean()
        tra_target_nre_at_norm_min_mean = tra_target_nre_at_norm_min_list[c].mean()
        adv_target_chamfer_at_norm_min_mean = adv_target_chamfer_at_norm_min_list[c].mean()
        adv_target_nre_at_norm_min_mean = adv_target_nre_at_norm_min_list[c].mean()

        spaces = ''.join([' '] * (16 - len(pc_class_name)))
        fout.write('%s%s%.5f\t\t%.2f\t\t%.5f\t\t%.2f\n' %
                   (pc_class_name, spaces,
                    tra_target_chamfer_at_norm_min_mean, tra_target_nre_at_norm_min_mean,
                    adv_target_chamfer_at_norm_min_mean, adv_target_nre_at_norm_min_mean))

    # over classes statistics
    tra_target_chamfer_at_norm_min_classes = np.vstack(tra_target_chamfer_at_norm_min_list)
    tra_target_chamfer_at_norm_min_classes_mean = tra_target_chamfer_at_norm_min_classes .mean()
    tra_target_nre_at_norm_min_classes = np.vstack(tra_target_nre_at_norm_min_list)
    tra_target_nre_at_norm_min_classes_mean = tra_target_nre_at_norm_min_classes.mean()
    avd_target_chamfer_at_norm_min_classes = np.vstack(adv_target_chamfer_at_norm_min_list)
    avd_target_chamfer_at_norm_min_classes_mean = avd_target_chamfer_at_norm_min_classes.mean()
    adv_target_nre_at_norm_min_classes = np.vstack(adv_target_nre_at_norm_min_list)
    adv_target_nre_at_norm_min_classes_mean = adv_target_nre_at_norm_min_classes.mean()

    fout.write('\n')
    pc_class_name = 'over classes'
    spaces = ''.join([' '] * (16 - len(pc_class_name)))
    fout.write('%s%s%.5f\t\t%.2f\t\t%.5f\t\t%.2f\n' %
               (pc_class_name, spaces,
                tra_target_chamfer_at_norm_min_classes_mean, tra_target_nre_at_norm_min_classes_mean,
                avd_target_chamfer_at_norm_min_classes_mean, adv_target_nre_at_norm_min_classes_mean))


def write_classification_statistics_to_file(fout, classes_for_attack, recon_cls_at_norm_min_list, data_type):
    if data_type == 'target':
        fout.write('Shape\t\tOrig target recon\n')
        fout.write('Shape\t\tTarget accuracy\n')
    elif data_type == 'adversarial':
        fout.write('Shape\t\tAdv recon\n')
        fout.write('Shape\t\tTarget accuracy\n')
    elif data_type == 'source':
        fout.write('Shape\t\tOrig source recon\n')
        fout.write('Shape\t\tSource accuracy\n')
    elif data_type == 'before_defense':
        fout.write('Shape\t\tAdv recon\n')
        fout.write('Shape\t\tSource accuracy\n')
    elif data_type == 'after_defense':
        fout.write('Shape\t\tDef recon\n')
        fout.write('Shape\t\tSource accuracy\n')
    fout.write('\n')

    for c, pc_class_name in enumerate(classes_for_attack):
        cls_adv_pc_recon_at_norm_min_mean = recon_cls_at_norm_min_list[c].mean()

        spaces = ''.join([' '] * (16 - len(pc_class_name)))
        fout.write('%s%s%.4f\n' % (pc_class_name, spaces, cls_adv_pc_recon_at_norm_min_mean))

    # over classes statistics
    cls_adv_pc_recon_at_norm_min_classes = np.vstack(recon_cls_at_norm_min_list)
    cls_adv_pc_recon_at_norm_min_mean = cls_adv_pc_recon_at_norm_min_classes.mean()

    fout.write('\n')
    pc_class_name = 'over classes'
    spaces = ''.join([' '] * (16 - len(pc_class_name)))
    fout.write('%s%s%.4f\n' % (pc_class_name, spaces, cls_adv_pc_recon_at_norm_min_mean))
