import numpy as np 
import tensorflow as tf


def scale_object(data, scale):
    center = (np.max(data, axis=0) + np.min(data, axis=0)) / 2
    data_centered = data - np.expand_dims(center, axis=0)
    norm = np.linalg.norm(data_centered, axis=1)
    radius = np.max(norm)
    data_normed = (data / radius) * scale
    return data_normed


def samp_object(obj, num_point):
    obj_copy = obj.copy()
    if obj_copy.shape[0] > num_point:
        np.random.shuffle(obj_copy)
        samp = obj_copy[:num_point]
    return samp


def sort_axes(point_clouds, neg_rot=True):
    """
    Sort axes of points clouds, such that the long, medium and short axes are x, y, z, respectively.
    If neg_rot is True, the rotation is by a negative angle, otherwise a positive angle.
    """
    axis_idx = int(neg_rot)

    axes_sort_idx, axes_len = get_sort_axes_idx(point_clouds)

    point_clouds_axes_sorted = np.zeros_like(point_clouds)
    num_pc = len(point_clouds)
    for i in range(num_pc):
        point_clouds_axes_sorted[i] = point_clouds[i, :, axes_sort_idx[i]].T

        if axes_len[i, 0] < axes_len[i, 1]:
            # x axis was swapped with y axis. mirror current x/y axis to get a proper +90/-90 degrees rotation around the z axis
            point_clouds_axes_sorted[i, :, axis_idx] = -point_clouds_axes_sorted[i, :, axis_idx]

    # sanity check
    _, axes_len_sorted = get_sort_axes_idx(point_clouds_axes_sorted)
    assert np.all(axes_len_sorted[:, 0] >= axes_len_sorted[:, 1]), 'Wrong axes sorting. The x axis length should be >= than the y axis length'

    return point_clouds_axes_sorted


def get_sort_axes_idx(point_clouds):
    """
    Get indices for sorting xy axes of points clouds, such that long and short axes are along x and y axes, respectively (the z axis is not changed).
    """
    assert len(point_clouds.shape) == 3, 'point_clouds should have 3 dimensions, got %d' % len(point_clouds.shape)
    max_val = point_clouds.max(axis=1)
    min_val = point_clouds.min(axis=1)
    axes_len = max_val - min_val

    axes_len_for_sort = axes_len
    axes_len_for_sort[:, 2] = 0.

    axes_sort_idx = np.argsort(axes_len_for_sort, axis=1)[:, ::-1]  # larger axis first
    assert np.all(axes_sort_idx[:, 2] == 2), 'Sorting only xy axes, the z axis should remain the same!'

    return axes_sort_idx, axes_len


def euler2mat_np(point_cloud, rotation, z_only=True):
    assert rotation.shape == (3,), 'The rotation should be a vector of size 3'

    x, y, z = rotation

    cosz = np.cos(z)
    sinz = np.sin(z)

    Mz = np.array(
        [[cosz, -sinz, 0],
         [sinz,  cosz, 0],
         [0,  0, 1]])

    if z_only:
        rotate_mat = Mz
    else:
        cosy = tf.cos(y)
        siny = tf.sin(y)
        My = np.array(
            [[cosy,  0, siny],
             [0,  1,  0],
             [-siny, 0, cosy]])

        cosx = np.cos(x)
        sinx = np.sin(x)
        Mx = np.array(
            [[1,  0,  0],
             [0, cosx, -sinx],
             [0, sinx,  cosx]])

        rotate_mat = tf.matmul(Mx, tf.matmul(My, Mz))

    rotate_mat = rotate_mat.astype(np.float32)
    rotate_mat[np.abs(rotate_mat) < 1e-10] = 0.

    point_cloud_rot = np.dot(point_cloud, rotate_mat)
    return point_cloud_rot


def euler2mat_tf(point_cloud, rotations, z_only=False):
    batch_size = rotations.get_shape()[0].value
    assert rotations.get_shape()[1].value == 3
    rotated_list = []
    one = tf.constant([1.])
    zero = tf.constant([0.])
    #print(zero.get_shape())

    for i in range(batch_size):
        x = rotations[i, 0]
        y = rotations[i, 1]
        z = rotations[i, 2]

        cosz = tf.cos([z])
        sinz = tf.sin([z])
        #print(cosz.get_shape())

        Mz = tf.stack(
            [[cosz, -sinz, zero],
             [sinz,  cosz, zero],
             [zero,  zero, one]])
        Mz = tf.squeeze(Mz)

        if z_only:
            rotate_mat = Mz
        else:
            cosy = tf.cos([y])
            siny = tf.sin([y])
            My = tf.stack(
                [[cosy,  zero, siny],
                 [zero,  one,  zero],
                 [-siny, zero, cosy]])
            My = tf.squeeze(My)

            cosx = tf.cos([x])
            sinx = tf.sin([x])
            Mx = tf.stack(
                [[one,  zero,  zero],
                 [zero, cosx, -sinx],
                 [zero, sinx,  cosx]])
            Mx = tf.squeeze(Mx)

            rotate_mat = tf.matmul(Mx, tf.matmul(My, Mz))

        rotated_list.append(tf.matmul(point_cloud[i], rotate_mat))

    return tf.stack(rotated_list)


if __name__=='__main__':
    import sys
    import os.path as osp
    import matplotlib.pylab as plt

    # add paths
    parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from src.in_out import snc_category_to_synth_id, load_and_split_all_point_clouds_under_folder
    from src.general_utils import plot_3d_point_cloud

    project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    top_in_dir = osp.join(project_dir, 'data', 'shape_net_core_uniform_samples_2048')  # Top-dir of where point-clouds are stored.

    class_name = 'chair'
    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir, syn_id)
    _, _, pc_data_test = load_and_split_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

    ################
    # euler2mat_np #
    ################
    pc = pc_data_test.point_clouds[0:4]
    rot = np.array([0, 0, -0.5 * np.pi])
    pc_rot = euler2mat_np(pc, rot)

    plot_3d_point_cloud(pc[0], title='input 0')
    plot_3d_point_cloud(pc_rot[0], title='input 0 rot 90 deg z')
    plot_3d_point_cloud(pc[1], title='input 1')
    plot_3d_point_cloud(pc_rot[1], title='input 1 rot 90 deg z')
    plot_3d_point_cloud(pc[2], title='input 2')
    plot_3d_point_cloud(pc_rot[2], title='input 2 rot 90 deg z')
    plot_3d_point_cloud(pc[3], title='input 3')
    plot_3d_point_cloud(pc_rot[3], title='input 3 rot 90 deg z')

    ################
    # euler2mat_tf #
    ################
    pc = np.tile(pc_data_test.point_clouds[0:1], [4, 1, 1])
    rot = np.array([[-0.5 * np.pi, 0, 0],  # 90 deg x
                    [0, -0.5 * np.pi, 0],  # 90 deg y
                    [0, 0, -0.5 * np.pi],  # 90 deg z
                    [-0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]   # 90 deg each axis
                    ], dtype=np.float32)

    pc_pl = tf.placeholder(tf.float32, [4, 2048, 3])
    rot_pl = tf.placeholder(tf.float32, [4, 3])
    pc_rot_tf = euler2mat_tf(pc_pl, rot_pl)

    with tf.Session('') as sess:
        pc_rot = pc_rot_tf.eval(feed_dict={pc_pl: pc, rot_pl: rot})

    plot_3d_point_cloud(pc[0], title='input')
    plot_3d_point_cloud(pc_rot[0], title='input rot 90 deg x')
    plot_3d_point_cloud(pc_rot[1], title='input rot 90 deg y')
    plot_3d_point_cloud(pc_rot[2], title='input rot 90 deg z')
    plot_3d_point_cloud(pc_rot[3], title='input rot 90 deg x y z')

    plt.show()
