"""
Created on November 26, 2017
@author: optas
Edited by itailang
"""

import os.path as osp
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
import pandas as pd


def rand_rotation_matrix(deflection=1.0, z_only=True, seed=None):
    '''Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, completely random
    rotation. Small deflection => small perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    '''
    if seed is not None:
        np.random.seed(seed)

    randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi    # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi     # For direction of pole deflection.
    z = z * 2.0 * deflection    # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0),
                  (-st, ct, 0),
                  (0, 0, 1)))

    if not z_only:
        r = np.sqrt(z)
        V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.
        M = (np.outer(V, V) - np.eye(3)).dot(R)
    else:
        M = R

    return M


def get_complementary_points(pcloud, idx):
    dim_num = len(pcloud.shape)
    n = pcloud.shape[dim_num - 2]
    k = idx.shape[dim_num - 2]

    if dim_num == 2:
        comp_idx = get_complementary_idx(idx, n)
        comp_points = pcloud[comp_idx, :]
    else:
        n_example = pcloud.shape[0]
        comp_points = np.zeros([n_example, n-k, pcloud.shape[2]])
        comp_idx = np.zeros([n_example, n-k], dtype=int)

        for i in range(n_example):
            comp_idx[i, :] = get_complementary_idx(idx[i, :], n)
            comp_points[i, :, :] = pcloud[i, comp_idx[i, :], :]

    return comp_points, comp_idx


def get_complementary_idx(idx, n):
    range_n = np.arange(n, dtype=int)
    comp_indicator = np.full(n, True)

    comp_indicator[idx] = False
    comp_idx = range_n[comp_indicator]

    return comp_idx


def get_dist_mat(data):
    assert len(data.shape) == 2, 'The data is assumed to have 2 dimensions, got a shape of %s' % str(data.shape)
    num_examples = len(data)
    tile_source = [num_examples, 1, 1]
    tile_target = [1, num_examples, 1]
    source_data = np.tile(np.expand_dims(data, axis=0), tile_source)
    target_data = np.tile(np.expand_dims(data, axis=1), tile_target)

    # source on the rows (axis 0), target in the columns (axis 1)
    dist_mat = np.linalg.norm(source_data - target_data, axis=-1)
    assert np.array_equal(dist_mat, dist_mat.T), 'The distance matrix should be a symmetric matrix!'

    return dist_mat


def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]

        
def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def apply_augmentations(batch, conf):
    if conf.gauss_augment is not None or conf.z_rotate:
        batch_copy = batch.copy()
    else:
        batch_copy = batch

    if conf.gauss_augment is not None:
        mu = conf.gauss_augment['mu']
        sigma = conf.gauss_augment['sigma']
        batch_copy += np.random.normal(mu, sigma, batch_copy.shape)

    if conf.z_rotate:
        r_rotation = rand_rotation_matrix()
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
        batch_copy = batch_copy.dot(r_rotation)

    return batch_copy


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def plot_3d_point_cloud(pc, show=True, show_axis=True, in_u_sphere=True, marker='.', c='b', s=8, alpha=.8, figsize=(5, 5), elev=10, azim=240, miv=None, mav=None, squeeze=0.7, axis=None, title=None, *args, **kwargs):
    x, y, z = (pc[:, 0], pc[:, 1], pc[:, 2])

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, c=c, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
        miv = -0.5
        mav = 0.5
    else:
        if miv is None:
            miv = squeeze * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 'squeeze' to squeeze free-space.
        if mav is None:
            mav = squeeze * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig, miv, mav


def plot_heatmap_graph(heatmap_vals, rows_label, columns_label, pc_class_name, xlabel, ylabel, fmt, save_path, figsize=(5, 5), font_size=16):
    plt.figure(figsize=figsize)
    df_cm = pd.DataFrame(heatmap_vals, rows_label, columns_label)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, fmt=fmt, annot_kws={"size": 10})
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.title('Shape Class $\\bf{%s}$' % pc_class_name, fontsize=font_size)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
