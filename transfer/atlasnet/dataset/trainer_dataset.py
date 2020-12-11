"""
Author: Thibault Groueix 01.11.2019
Modified on October 12th, 2020
@modifier: urikotlicki
"""

# import system modules
import os.path as osp
import sys
from easydict import EasyDict
import torch

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
import transfer.atlasnet.dataset.dataset_shapenet as dataset_shapenet
import transfer.atlasnet.dataset.augmenter as augmenter


class TrainerDataset(object):
    def __init__(self):
        super(TrainerDataset, self).__init__()

    def build_dataset(self, test_pc=None, shuffle_test=True):
        """
        Create dataset
        Author: Thibault Groueix 01.11.2019
        """

        self.datasets = EasyDict()
        # Create Datasets
        if self.opt.mode == 'train':
            self.datasets.dataset_train = dataset_shapenet.ShapeNet(self.opt, mode='train')
            self.datasets.dataset_test = dataset_shapenet.ShapeNet(self.opt, mode='eval')

            if not self.opt.demo:
                # Create dataloaders
                self.datasets.dataloader_train = torch.utils.data.DataLoader(self.datasets.dataset_train,
                                                                             batch_size=self.opt.batch_size,
                                                                             shuffle=True,
                                                                             num_workers=int(self.opt.workers))
                self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                            batch_size=self.opt.batch_size_test,
                                                                            shuffle=True, num_workers=int(self.opt.workers))
                axis = []
                if self.opt.data_augmentation_axis_rotation:
                    axis = [1]

                flips = []
                if self.opt.data_augmentation_random_flips:
                    flips = [0, 2]

                # Create Data Augmentation
                self.datasets.data_augmenter = augmenter.Augmenter(translation=self.opt.random_translation,
                                                                   rotation_axis=axis,
                                                                   anisotropic_scaling=self.opt.anisotropic_scaling,
                                                                   rotation_3D=self.opt.random_rotation,
                                                                   flips=flips)

                self.datasets.len_dataset = len(self.datasets.dataset_train)
                self.datasets.len_dataset_test = len(self.datasets.dataset_test)
        elif self.opt.mode == 'test':
            if test_pc is None:
                self.datasets.dataset_test = dataset_shapenet.ShapeNet(self.opt, mode='test')

                self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                            batch_size=self.opt.batch_size_test,
                                                                            shuffle=shuffle_test,
                                                                            num_workers=int(self.opt.workers))

                self.datasets.len_dataset_test = len(self.datasets.dataset_test)
            else:
                self.datasets.dataset_test = dataset_shapenet.ShapeNet(self.opt, mode='test', test_pc=test_pc)

                self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                            batch_size=self.opt.batch_size_test,
                                                                            shuffle=shuffle_test,
                                                                            num_workers=int(self.opt.workers))
                self.datasets.len_dataset_test = len(self.datasets.dataset_test.data_points)
