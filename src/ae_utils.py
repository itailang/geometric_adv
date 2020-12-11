"""
Created on September 5th, 2020
@author: itailang
"""

import os.path as osp
import numpy as np

from src.general_utils import get_complementary_points, plot_3d_point_cloud


def get_critical_points(point_clouds, pre_symmetry_data, data_path, suff_list, save_data=True):
    num_pc, _, bottleneck_size = pre_symmetry_data.shape
    critical_points = np.zeros([num_pc, bottleneck_size, 3], dtype=point_clouds.dtype)
    idx_critical = np.zeros([num_pc, bottleneck_size], dtype=np.int16)
    num_critical = np.zeros(num_pc, dtype=np.int16)
    for i in range(num_pc):
        pre_symmetry_pc = pre_symmetry_data[i]
        max_val = np.max(pre_symmetry_pc, axis=0)
        max_idx = np.argmax(pre_symmetry_pc, axis=0)
        max_idx_non_zero = max_idx[max_val > 0.0]  # remove entries for which the entire column of pre_symmetry_pc is 0
        idx_critical_pc, counts = np.unique(max_idx_non_zero, return_counts=True)
        num_critical_pc = idx_critical_pc.shape[0]
        num_critical[i] = num_critical_pc

        idx_sort = np.argsort(counts)[::-1]  # most critical points first
        idx_critical_pc_sorted = idx_critical_pc[idx_sort]
        critical_points_pc = point_clouds[i][idx_critical_pc_sorted]
        critical_points[i, :num_critical_pc, :] = critical_points_pc  # fill the critical points in a zeros matrix (in order to use a numpy array)
        idx_critical[i, :num_critical_pc] = idx_critical_pc_sorted  # fill the critical index in a zeros matrix (in order to use a numpy array)

    if save_data:
        # save critical points
        critical_points_file_name = '_'.join(['critical_points'] + suff_list)
        critical_points_file_path = osp.join(data_path, critical_points_file_name)
        np.save(critical_points_file_path, critical_points)

        # save critical points index
        critical_idx_file_name = '_'.join(['critical_idx'] + suff_list)
        critical_idx_file_path = osp.join(data_path, critical_idx_file_name)
        np.save(critical_idx_file_path, idx_critical)

        # save number of critical points (to know which entries to use from the critical_points matrix)
        critical_num_file_name = '_'.join(['critical_num'] + suff_list)
        critical_num_file_path = osp.join(data_path, critical_num_file_name)
        np.save(critical_num_file_path, num_critical)

    return critical_points, idx_critical, num_critical


def get_critical_pc_non_critical_pc(point_clouds, pre_symmetry_data):
    critical_points, critical_idx, critical_num = get_critical_points(point_clouds, pre_symmetry_data, None, None, save_data=False)

    num_pc = len(point_clouds)
    critical_pc = np.zeros_like(point_clouds)
    non_critical_pc = np.zeros_like(point_clouds)
    for k in range(num_pc):
        # sanity check
        assert np.any(critical_idx[k, critical_num[k]:]) == False, \
            'critical_idx form critical_num to the end of the row should all be zeros'

        critical_idx_pc = critical_idx[k, :critical_num[k]]

        # critical points
        critical_points_pc = point_clouds[k, critical_idx_pc, :]
        critical_pc[k, :critical_num[k], :] = critical_points_pc
        critical_pc[k, critical_num[k]:, :] = critical_points_pc[-1]  # duplication of last point does not change the latent vector due to global pooling

        # sanity check
        assert np.array_equal(critical_points_pc, critical_points[k, :critical_num[k]]), \
            'Input point cloud at critical points index should be equal to the critical points!'

        # non critical points
        non_critical_points_pc, non_critical_idx_pc = get_complementary_points(point_clouds[k], critical_idx_pc)
        non_critical_num = len(non_critical_points_pc)

        non_critical_pc[k, :non_critical_num] = non_critical_points_pc
        non_critical_pc[k, non_critical_num:] = non_critical_points_pc[-1]  # duplication of last point does not change the latent vector due to global pooling

    return critical_points, critical_idx, critical_num, critical_pc, non_critical_pc
