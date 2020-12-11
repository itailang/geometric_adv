"""
Created on September 29th, 2020
@author: urikotlicki
"""

# Based on code taken from: https://github.com/YanWei123/Pytorch-implementation-of-FoldingNet-encoder-and-decoder-with-graph-pooling-covariance-add-quanti

# import system modules
import argparse
import os.path as osp
import sys
import random
import numpy as np
import time
import torch.utils.data
from torch.autograd import Variable

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from transfer.foldingnet.foldingnet import FoldingNet_graph
from transfer.foldingnet.foldingnet import ChamferLoss
from transfer.foldingnet.prepare_graph import build_graph

parser = argparse.ArgumentParser()
parser.add_argument('--test_set', type=str, default='log/autoencoder_victim/eval/point_clouds_test_set_13l.npy',
                    help='Path to test set examples [default: log/autoencoder_victim/eval/point_clouds_test_set_13l.npy]')
parser.add_argument('--batchSize', type=int, default=16, help='Input batch size [default: 16]')
parser.add_argument('--num_points', type=int, default=2048, help='Input batch size [default: 2048]')
parser.add_argument('--workers', type=int,  default=4, help='Number of data loading workers [default: 4]')
parser.add_argument('--outf', type=str, default='log/foldingnet',  help='Output folder [default: log/foldingnet]')
parser.add_argument('--checkpoint_num', type=int, default=24,  help='Checkpoint number to be loaded [default: 24]')
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

point_clouds = torch.tensor(np.load(osp.join(parent_dir, opt.test_set)))

test_dataset = torch.utils.data.TensorDataset(point_clouds)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))

print('Test set: %d examples' % (len(test_dataset)))

foldingnet = FoldingNet_graph()
foldingnet.cuda()

checkpoint = torch.load(osp.join(parent_dir, opt.outf, 'checkpoint_%d.pth' % opt.checkpoint_num))
foldingnet.load_state_dict(checkpoint['model'])
print('Checkpoint successfully loaded')

num_batch = len(test_dataset)/opt.batchSize

chamferloss = ChamferLoss()
chamferloss.cuda()

foldingnet = foldingnet.eval()
sum_loss = 0
sum_mid_loss = 0
sum_step = 0

for j, data in enumerate(test_dataloader, 0):
    start_time = time.time()
    points = data[0]

    # build graph
    batch_graph, Cov = build_graph(points, opt)
    Cov = Cov.transpose(2, 1)
    Cov = Cov.cuda()

    points = Variable(points)
    points = points.transpose(2, 1)
    points = points.cuda()

    recon_pc, mid_pc, _ = foldingnet(points, Cov, batch_graph)
    loss = chamferloss(points.transpose(2, 1), recon_pc.transpose(2,1))

    mid_loss = chamferloss(points.transpose(2, 1), mid_pc.transpose(2,1))
    # store loss and step
    sum_loss += loss.item() * points.size(0)
    sum_mid_loss += mid_loss.item() * points.size(0)
    sum_step += points.size(0)
    print('Batch %d/%d\t Duration (minutes): %.3f ' % (j, num_batch, (time.time()-start_time)/60.0))

print('%s test loss: %f middle test loss: %f' % (blue('Testing'), sum_loss/sum_step, sum_mid_loss/sum_step))
