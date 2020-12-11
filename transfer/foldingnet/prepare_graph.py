"""
Created on September 11th, 2020
@author: urikotlicki
"""

# Based on code taken from: https://github.com/YanWei123/Pytorch-implementation-of-FoldingNet-encoder-and-decoder-with-graph-pooling-covariance-add-quanti

import numpy as np
import scipy.sparse
from sklearn.neighbors import KDTree
import multiprocessing as multiproc
from functools import partial
import torch


class GraphOptions():
    def __init__(self):
        self.mode = "M"  # "mode used to compute graphs: M, P"
        self.metric = 'euclidean'  # "metric for distance calculation (manhattan/euclidean)"
        self.knn = 16
        self.num_points = 2048


def edges2A(edges, n_nodes, mode='P', sparse_mat_type=scipy.sparse.csr_matrix):
    '''
    note: assume no (i,i)-like edge
    edges: <2xE>
    '''
    edges = np.array(edges).astype(int)

    data_D = np.zeros(n_nodes, dtype=np.float32)
    for d in range(n_nodes):
        data_D[ d ] = len(np.where(edges[0] == d)[0])   # compute the number of node which pick node_i as their neighbor

    if mode.upper() == 'M':  # 'M' means max pooling, which use the same graph matrix as the adjacency matrix
        data = np.ones(edges[0].shape[0], dtype=np.int32)
    elif mode.upper() == 'P':
        data = 1. / data_D[ edges[0] ]
    else:
        raise NotImplementedError("edges2A with unknown mode=" + mode)

    return sparse_mat_type((data, edges), shape=(n_nodes, n_nodes))


def knn_search(data, knn=16, metric="euclidean", symmetric=True):
    """
    Args:
      data: Nx3
      knn: default=16
    """
    assert(knn>0)
    n_data_i = data.shape[0]
    kdt = KDTree(data, leaf_size=30, metric=metric)

    nbs = kdt.query(data, k=knn+1, return_distance=True)    # nbs[0]:NN distance,N*17. nbs[1]:NN index,N*17
    cov = np.zeros((n_data_i,9), dtype=np.float32)
    adjdict = dict()
    # wadj = np.zeros((n_data_i, n_data_i), dtype=np.float32)
    for i in range(n_data_i):
        # nbsd = nbs[0][i]
        nbsi = nbs[1][i]    #index i, N*17 YW comment
        cov[i] = np.cov(data[nbsi[1:]].T).reshape(-1) #compute local covariance matrix
        for j in range(knn):
            if symmetric:
                adjdict[(i, nbsi[j+1])] = 1
                adjdict[(nbsi[j+1], i)] = 1
                # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
                # wadj[nbsi[j + 1], i] = 1.0 / nbsd[j + 1]
            else:
                adjdict[(i, nbsi[j+1])] = 1
                # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
    edges = np.array(list(adjdict.keys()), dtype=int).T
    return edges, nbs[0], cov #, wadj


def build_graph_core(ith_datai, args):
    try:
        #ith, xyi = ith_datai #xyi: 2048x3
        xyi = ith_datai  # xyi: 2048x3
        n_data_i = xyi.shape[0]
        edges, nbsd, cov = knn_search(xyi, knn=args.knn, metric=args.metric)
        ith_graph = edges2A(edges, n_data_i, args.mode, sparse_mat_type=scipy.sparse.csr_matrix)
        nbsd=np.asarray(nbsd)[:, 1:]
        nbsd=np.reshape(nbsd, -1)

        #if ith % 500 == 0:
            #logger.info('{} processed: {}'.format(args.flag, ith))

        #return ith, ith_graph, nbsd, cov
        return ith_graph, nbsd, cov
    except KeyboardInterrupt:
        exit(-1)


def build_graph(points, args=None):      # points: batch, num of points, 3
    if args is None:
        args = GraphOptions()  # Use default args

    batch_graph = []
    Cov = torch.zeros(points.shape[0], args.num_points, 9)

    pool = multiproc.Pool(2)
    pool_func = partial(build_graph_core, args=args)
    rets = pool.map(pool_func, points)
    pool.close()
    count = 0
    for ret in rets:
        ith_graph, _, cov = ret
        batch_graph.append(ith_graph)
        Cov[count, :, :] = torch.from_numpy(cov)
        count = count+1
    del rets

    return batch_graph, Cov
