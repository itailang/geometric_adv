"""
Created on October 12th, 2020
@author: urikotlicki
"""

# Part of the code is based on: https://github.com/ThibaultGROUEIX/AtlasNet

# import system modules
import os.path as osp
import sys
import time
import numpy as np
import torch

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
import transfer.atlasnet.auxiliary.argument_parser as argument_parser
import transfer.atlasnet.auxiliary.my_utils as my_utils
import transfer.atlasnet.auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from src.general_utils import plot_3d_point_cloud


class AtlasNetAutoEncoder:
    '''
    AtlasNet autoencoder for point-clouds
    '''

    def __init__(self, *_):
        self.chamferloss = dist_chamfer_3D.chamfer_3DDist()
    #     self.foldingnet = FoldingNet_graph()

    def restore_model(self, transfer_ae_dir, epoch, verbose=None):
        """This function is degenerated. Loading the model
        occurs under get_reconstructions-->trainer.build_network()"""

    def get_reconstructions(self, pc_input, flags):
        opt = argument_parser.parser_transfer(flags)
        opt.mode = 'test'
        opt.custom_data = True
        pc_recon = np.zeros([pc_input.shape[0], 2500, 3], dtype=pc_input.dtype)
        torch.cuda.set_device(opt.multi_gpu[0])
        my_utils.plant_seeds(random_seed=opt.random_seed)

        import transfer.atlasnet.training.trainer as trainer

        trainer = trainer.Trainer(opt)
        trainer.build_dataset(test_pc=pc_input, shuffle_test=False)
        trainer.build_network()
        trainer.build_optimizer()
        trainer.build_losses()
        trainer.start_train_time = time.time()

        with torch.no_grad():
            pc_recon = trainer.test_epoch(pc_recon=pc_recon)

        show = False
        if show:
            plot_3d_point_cloud(pc_input[0])
            plot_3d_point_cloud(pc_recon[0])

        return pc_recon

    def get_loss_per_pc(self, pc_recon, target_pc):
        assert len(pc_recon.shape) == 3, 'The pc_input should have 3 dimensions'
        assert len(target_pc.shape) == 3, 'The target_pc should have 3 dimensions'
        assert pc_recon.shape[0] == target_pc.shape[0], 'Number of point clouds must match'

        n_examples = pc_recon.shape[0]
        loss = torch.zeros(n_examples)
        loss = loss.cuda()

        for i in range(0, n_examples, 1):
            pc_recon_ch = torch.tensor(pc_recon[i:i+1])
            pc_recon_ch = pc_recon_ch.cuda()
            target_pc_ch = torch.tensor(target_pc[i:i+1])
            target_pc_ch = target_pc_ch.cuda()

            dist1, dist2, idx1, idx2 = self.chamferloss(pc_recon_ch, target_pc_ch)  # mean over points

            loss[i] = torch.mean(dist1) + torch.mean(dist2)  # mean over points

        loss = loss.cpu().detach().numpy()

        return loss
