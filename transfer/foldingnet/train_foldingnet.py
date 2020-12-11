"""
Created on September 29th, 2020
@author: urikotlicki
"""

# Based on code taken from: https://github.com/YanWei123/Pytorch-implementation-of-FoldingNet-encoder-and-decoder-with-graph-pooling-covariance-add-quanti

# import system modules
from __future__ import print_function
import argparse
import os
import os.path as osp
import sys
import random
import numpy as np
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from transfer.foldingnet.foldingnet import FoldingNet_graph
from transfer.foldingnet.foldingnet import ChamferLoss
from transfer.foldingnet.prepare_graph import build_graph
from src.general_utils import plot_3d_point_cloud


parser = argparse.ArgumentParser()
parser.add_argument('--training_set', type=str, default='log/autoencoder_victim/eval_train/point_clouds_train_set_13l.npy',
                    help='Path to training set examples [default: log/autoencoder_victim/eval_train/point_clouds_train_set_13l.npy]')
parser.add_argument('--validation_set', type=str, default='log/autoencoder_victim/eval_val/point_clouds_val_set_13l.npy',
                    help='Path to validation set examples [default: log/autoencoder_victim/eval_val/point_clouds_val_set_13l.npy]')
parser.add_argument('--batchSize', type=int, default=8, help='Input batch size [default: 8]')
parser.add_argument('--num_points', type=int, default=2048, help='Input batch size [default: 2048]')
parser.add_argument('--workers', type=int,  default=4, help='Number of data loading workers [default: 4]')
parser.add_argument('--nepoch', type=int, default=25, help='Number of epochs to train for [default: 25]')
parser.add_argument('--outf', type=str, default='log/foldingnet',  help='Output folder [default: log/foldingnet]')
parser.add_argument('--checkpoint_num', type=int, default=0,  help='Checkpoint number to be loaded [default: 0]')
parser.add_argument('-md', '--mode', type=str, default="M", help="Mode used to compute graphs: M, P")
parser.add_argument('-m', '--metric', type=str, default='euclidean', help="Metric for distance calculation (manhattan/euclidean)")

opt = parser.parse_args()
opt.knn = 16
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

point_clouds = torch.tensor(np.load(osp.join(parent_dir, opt.training_set)))

train_dataset = torch.utils.data.TensorDataset(point_clouds)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                               shuffle=True, num_workers=int(opt.workers))

point_clouds_val = torch.tensor(np.load(parent_dir + '/' + opt.validation_set))
val_dataset = torch.utils.data.TensorDataset(point_clouds_val)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

print('Training set: %d examples\nValidation set: %d examples' % (len(train_dataset), len(val_dataset)))

try:
    os.makedirs(osp.join(parent_dir, opt.outf))
except OSError:
    pass

foldingnet = FoldingNet_graph()
optimizer = optim.Adam(foldingnet.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-6)
foldingnet.cuda()

if opt.checkpoint_num > 0:
    checkpoint = torch.load(osp.join(parent_dir, opt.outf, 'checkpoint_%d.pth' % opt.checkpoint_num))
    start_epoch = checkpoint['epoch'] + 1
    foldingnet.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Checkpoint successfully loaded')
else:
    start_epoch = 0
    print('Start training from epoch 0')

num_batch = len(train_dataset)/opt.batchSize

chamferloss = ChamferLoss()
chamferloss.cuda()

start_time = time.time()
time_p, loss_p, loss_m = [], [], []

for epoch in range(start_epoch, opt.nepoch):
    sum_loss = 0
    sum_step = 0
    sum_mid_loss = 0
    for i, data in enumerate(train_dataloader, 0):
        points = data[0]
        batch_graph, Cov = build_graph(points, opt)

        Cov = Cov.transpose(2, 1)
        Cov = Cov.cuda()
        points = points.transpose(2, 1)
        points = points.cuda()

        optimizer.zero_grad()
        foldingnet = foldingnet.train()
        recon_pc, mid_pc, _ = foldingnet(points, Cov, batch_graph)

        loss = chamferloss(points.transpose(2, 1), recon_pc.transpose(2, 1))
        loss.backward()
        optimizer.step()

        mid_loss = chamferloss(points.transpose(2, 1), mid_pc.transpose(2, 1))

        # store loss and step
        sum_loss += loss.item()*points.size(0)
        sum_mid_loss += mid_loss.item()*points.size(0)
        sum_step += points.size(0)
        
        print('[%d: %d/%d] train loss: %f middle loss: %f' % (epoch, i, num_batch, loss.item(), mid_loss.item()))

        if i % 100 == 0:
            j, data = next(enumerate(val_dataloader, 0))
            # points, target = data
            points = data[0]
            # build graph
            batch_graph, Cov = build_graph(points, opt)
            Cov = Cov.transpose(2, 1)
            Cov = Cov.cuda()

            # points, target = Variable(points), Variable(target[:,0])
            points = Variable(points)
            points = points.transpose(2, 1)
            # points, target = points.cuda(), target.cuda()
            points = points.cuda()
            foldingnet = foldingnet.eval()
            recon_pc, mid_pc, _ = foldingnet(points, Cov, batch_graph)
            loss = chamferloss(points.transpose(2, 1), recon_pc.transpose(2,1))

            mid_loss = chamferloss(points.transpose(2, 1), mid_pc.transpose(2,1))

            # prepare show result
            points_show = points.cpu().detach().numpy()
            recon_show = recon_pc.cpu().detach().numpy()

            fig_orig = plt.figure()
            a1 = fig_orig.add_subplot(111, projection='3d')
            a1.scatter(points_show[0, 0, :], points_show[0, 1, :], points_show[0, 2, :])
            plt.savefig(osp.join(parent_dir, opt.outf, 'orig_pc_example.png'))

            fig_recon = plt.figure()
            a2 = fig_recon.add_subplot(111,projection='3d')
            a2.scatter(recon_show[0, 0, :], recon_show[0, 1, :], recon_show[0, 2, :])
            plt.savefig(osp.join(parent_dir, opt.outf, 'recon_pc_example.png'))

            # plot results
            time_p.append(time.time()-start_time)
            loss_p.append(sum_loss/sum_step)
            loss_m.append(sum_mid_loss/sum_step)

            print('[%d: %d/%d] %s val loss: %f middle val loss: %f' %
                  (epoch, i, num_batch, blue('validation'), loss.item(), mid_loss.item()))
            sum_step = 0
            sum_loss = 0
            sum_mid_loss = 0

    checkpoint = {
        'epoch': epoch,
        'model': foldingnet.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, osp.join(parent_dir, opt.outf, 'checkpoint_%d.pth' % epoch))
