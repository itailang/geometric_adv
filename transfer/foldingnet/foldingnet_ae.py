"""
Created on September 12th, 2020
@author: urikotlicki
"""

# Part of the code is based on: https://github.com/YanWei123/Pytorch-implementation-of-FoldingNet-encoder-and-decoder-with-graph-pooling-covariance-add-quanti

# import system modules
import os.path as osp
import sys
import numpy as np
import torch
from torch.autograd import Variable

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from transfer.foldingnet.foldingnet import FoldingNet_graph, ChamferLoss
from transfer.foldingnet.prepare_graph import build_graph


class FoldingNetAutoEncoder:
    '''
    FoldingNet autoencoder for point-clouds
    '''

    def __init__(self, *_):

        self.chamferloss = ChamferLoss()
        self.foldingnet = FoldingNet_graph()

    def restore_model(self, transfer_ae_dir, epoch, verbose=None):
        checkpoint = torch.load(transfer_ae_dir + '/checkpoint_%s.pth' % epoch)
        self.foldingnet.load_state_dict(checkpoint['model'])

    def get_reconstructions(self, pc_input, flags=None):
        self.foldingnet = self.foldingnet.cuda()
        self.foldingnet = self.foldingnet.eval()

        pc_recon = np.zeros([pc_input.shape[0], 2025, 3], dtype=pc_input.dtype)

        # In order to prevent out of memory error, work in batches of 8 point clouds
        batch_size = 4
        dataset = torch.utils.data.TensorDataset(torch.tensor(pc_input))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)

        for i, data in enumerate(dataloader, 0):
            print('Batch %d out of %d' % (i+1, np.ceil(pc_input.shape[0]/batch_size)))
            pc_input_batch = data[0]
            # build graph
            batch_graph, Cov = build_graph(pc_input_batch)

            Cov = Cov.transpose(2, 1)
            Cov = Cov.cuda()
            pc_input_batch = Variable(pc_input_batch)
            pc_input_batch = pc_input_batch.transpose(2, 1)
            pc_input_batch = pc_input_batch.cuda()

            pc_recon_batch, mid_pc_batch, _ = self.foldingnet(pc_input_batch, Cov, batch_graph)
            pc_recon_batch = pc_recon_batch.transpose(2, 1)
            pc_recon[i*batch_size:(i+1)*batch_size, :, :] = pc_recon_batch.cpu().detach().numpy()

        return pc_recon

    def get_loss_per_pc(self, pc_input, target_pc):
        assert len(pc_input.shape) == 3, 'The pc_input should have 3 dimensions'
        assert len(target_pc.shape) == 3, 'The target_pc should have 3 dimensions'
        assert pc_input.shape[0] == target_pc.shape[0], 'Number of point clouds must match'

        n_examples = pc_input.shape[0]
        loss = torch.zeros(n_examples)
        loss = loss.cuda()
        self.chamferloss = self.chamferloss.cuda()

        for i in range(0, n_examples, 1):
            pc_input_ch = torch.tensor(pc_input[i:i+1])
            pc_input_ch= pc_input_ch.cuda()
            target_pc_ch = torch.tensor(target_pc[i:i+1])
            target_pc_ch = target_pc_ch.cuda()

            loss[i] = self.chamferloss(pc_input_ch, target_pc_ch)

        loss = loss.cpu().detach().numpy()
        return loss
