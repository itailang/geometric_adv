"""
Author: Thibault Groueix 01.11.2019
Modified on October 11th, 2020
@modifier: urikotlicki
"""

# Based on code taken from: https://github.com/ThibaultGROUEIX/AtlasNet

import argparse
import os
from os.path import exists, join
import os.path as osp
import sys
import datetime
import json
from termcolor import colored
from easydict import EasyDict

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
import transfer.atlasnet.auxiliary.my_utils as my_utils


def parser():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help="Select train or test mode")
    parser.add_argument('--custom_data', action="store_true", help="Manually enter training + eval data")
    parser.add_argument('--train_pc_path', type=str, default='log/autoencoder_victim/eval_train/point_clouds_train_set_13l.npy', help="Path to training data")
    parser.add_argument('--eval_pc_path', type=str, default='log/autoencoder_victim/eval_val/point_clouds_val_set_13l.npy', help="Path to evaluation data")
    parser.add_argument('--test_pc_path', type=str, default='log/autoencoder_victim/eval/point_clouds_test_set_13l.npy', help="Path to test data")

    parser.add_argument("--no_learning", action="store_true", help="Learning mode (batchnorms...)")
    parser.add_argument("--train_only_encoder", action="store_true", help="only train the encoder")
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument("--random_seed", action="store_true", help="Fix random seed or not")
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=120, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=140, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=145, help='learning rate decay 2')
    parser.add_argument("--run_single_eval", action="store_true", help="evaluate a trained network")
    parser.add_argument("--demo", action="store_true", help="run demo autoencoder or single-view")

    # Data
    parser.add_argument('--normalization', type=str, default="UnitBall",
                        choices=['UnitBall', 'BoundingBox', 'Identity'])
    parser.add_argument("--shapenet13", action="store_true", help="Load 13 usual shapenet categories")
    parser.add_argument("--SVR", action="store_true", help="Single_view Reconstruction")
    parser.add_argument("--sample", action="store_false", help="Sample the input pointclouds")
    parser.add_argument('--class_choice', nargs='+', default=["airplane"], type=str)
    parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object during training, and generated by atlasnet')
    parser.add_argument('--number_points_eval', type=int, default=2500,
                        help='Number of points generated by atlasnet (rounded to the nearest squared number) ')
    parser.add_argument("--random_rotation", action="store_true", help="apply data augmentation : random rotation")
    parser.add_argument("--data_augmentation_axis_rotation", action="store_true",
                        help="apply data augmentation : axial rotation ")
    parser.add_argument("--data_augmentation_random_flips", action="store_true",
                        help="apply data augmentation : random flips")
    parser.add_argument("--random_translation", action="store_true",
                        help="apply data augmentation :  random translation ")
    parser.add_argument("--anisotropic_scaling", action="store_true",
                        help="apply data augmentation : anisotropic scaling")

    # Save dirs and reload
    parser.add_argument('--id', type=str, default="0", help='training name')
    parser.add_argument('--env', type=str, default="Atlasnet", help='visdom environment')
    parser.add_argument('--visdom_port', type=int, default=8890, help="visdom port")
    parser.add_argument('--http_port', type=int, default=8891, help="http port")
    parser.add_argument('--dir_name', type=str, default="", help='name of the log folder.')
    parser.add_argument('--demo_input_path', type=str, default="./doc/pictures/plane_input_demo.png", help='dirname')
    parser.add_argument('--reload_decoder_path', type=str, default="", help='dirname')
    parser.add_argument('--reload_model_path', type=str, default='', help='optional reload model path')

    # Network
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--loop_per_epoch', type=int, default=1, help='number of data loop per epoch')
    parser.add_argument('--nb_primitives', type=int, default=1, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SPHERE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')
    parser.add_argument('--multi_gpu', nargs='+', type=int, default=[0], help='Use multiple gpus')
    parser.add_argument("--remove_all_batchNorms", action="store_true", help="Replace all batchnorms by identity")
    parser.add_argument('--bottleneck_size', type=int, default=1024, help='dim_out_patch')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')

    # Loss
    parser.add_argument("--no_metro", action="store_true", help="Compute metro distance")

    opt = parser.parse_args()

    opt.date = str(datetime.datetime.now())
    now = datetime.datetime.now()
    opt = EasyDict(opt.__dict__)

    if opt.dir_name == "":
        # Create default dirname
        opt.dir_name = join(parent_dir, 'log', opt.id + now.isoformat())
    else:
        opt.dir_name = join(parent_dir, opt.dir_name)

    opt.train_pc_path = osp.join(parent_dir, opt.train_pc_path)
    opt.eval_pc_path = osp.join(parent_dir, opt.eval_pc_path)
    opt.test_pc_path = osp.join(parent_dir, opt.test_pc_path)

    # If running a demo, check if input is an image or a pointcloud
    if opt.demo:
        ext = opt.demo_input_path.split('.')[-1]
        if ext == "ply" or ext == "npy" or ext == "obj":
            opt.SVR = False
        elif ext == "png":
            opt.SVR = True

    if opt.demo or opt.run_single_eval:
        if not exists("./training/trained_models/atlasnet_singleview_25_squares/network.pth"):
            print("Dowload Trained Models.")
            os.system("chmod +x training/download_trained_models.sh")
            os.system("./training/download_trained_models.sh")

        if opt.reload_model_path == "" and opt.SVR:
            opt.dir_name = "./training/trained_models/atlasnet_singleview_1_sphere"
        elif opt.reload_model_path == "" and not opt.SVR:
            opt.dir_name = "./training/trained_models/atlasnet_autoencoder_1_sphere"


    if exists(join(opt.dir_name, "options.json")):
        # Reload parameters from options.txt if it exists
        with open(join(opt.dir_name, "options.json"), 'r') as f:
            my_opt_dict = json.load(f)
        my_opt_dict.pop("run_single_eval")
        my_opt_dict.pop("no_metro")
        my_opt_dict.pop("train_only_encoder")
        my_opt_dict.pop("no_learning")
        my_opt_dict.pop("demo")
        my_opt_dict.pop("demo_input_path")
        my_opt_dict.pop("dir_name")
        my_opt_dict.pop("mode")
        my_opt_dict.pop("custom_data")
        my_opt_dict.pop("train_pc_path")
        my_opt_dict.pop("eval_pc_path")
        my_opt_dict.pop("test_pc_path")
        for key in my_opt_dict.keys():
            opt[key] = my_opt_dict[key]
        if not opt.demo:
            print("Modifying input arguments to match network in dirname")
            my_utils.cyan_print("PARAMETER: ")
            for a in my_opt_dict:
                print(
                    "         "
                    + colored(a, "yellow")
                    + " : "
                    + colored(str(my_opt_dict[a]), "cyan")
                )

    # Hard code dimension of the template.
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    opt.dim_template = dim_template_dict[opt.template_type]

    # Visdom env
    opt.env = opt.env + opt.dir_name.split('/')[-1]

    return opt


def parser_transfer(flags):
    opt = {
        'mode': 'test',
        'custom_data': True, # Used for transfer
        'train_pc_path': '',
        'eval_pc_path': '',
        'test_pc_path': '',
        'no_learning': True,
        'train_only_encoder': False,
        'batch_size': 32,
        'batch_size_test': 32, # Used for transfer
        'workers': 0,  # Used for transfer
        'nepoch': 150,
        'start_epoch': 0,
        'random_seed': False,
        'lrate': 0.001,
        'lr_decay_1': 120,
        'lr_decay_2': 140,
        'lr_decay_3': 145,
        'run_single_eval': False,
        'demo': False,
        'normalization': "UnitBall",
        'shapenet13': False,
        'SVR': False,
        'sample': True,
        'class_choice': "airplane",
        'number_points': 2500,
        'number_points_eval': 2500,  # Used for transfer
        'random_rotation': False,
        'data_augmentation_axis_rotation': False,
        'data_augmentation_random_flips': False,
        'random_translation': False,
        'anisotropic_scaling': False,
        'id': "0",
        'env': "Atlasnet",
        'visdom_port': 8890,
        'http_port': 8891,
        'dir_name': flags.transfer_ae_folder,
        'demo_input_path': "./doc/pictures/plane_input_demo.png",
        'reload_decoder_path': "",
        'reload_model_path': '',
        'num_layers': 2,
        'hidden_neurons': 512,
        'loop_per_epoch': 1,
        'nb_primitives': 1,
        'template_type': "SPHERE",
        'multi_gpu': [0],
        'remove_all_batchNorms': False,
        'bottleneck_size': 1024,
        'activation': 'relu',
        'no_metro': False,
        'date': str(datetime.datetime.now())
    }

    now = datetime.datetime.now()
    opt = EasyDict(opt)

    if opt.dir_name == "":
        # Create default dirname
        opt.dir_name = join(parent_dir, 'log', opt.id + now.isoformat())
    else:
        opt.dir_name = join(parent_dir, opt.dir_name)

    opt.train_pc_path = osp.join(parent_dir, opt.train_pc_path)
    opt.eval_pc_path = osp.join(parent_dir, opt.eval_pc_path)
    opt.test_pc_path = osp.join(parent_dir, opt.test_pc_path)

    # If running a demo, check if input is an image or a pointcloud
    if opt.demo:
        ext = opt.demo_input_path.split('.')[-1]
        if ext == "ply" or ext == "npy" or ext == "obj":
            opt.SVR = False
        elif ext == "png":
            opt.SVR = True

    if opt.demo or opt.run_single_eval:
        if not exists("./training/trained_models/atlasnet_singleview_25_squares/network.pth"):
            print("Dowload Trained Models.")
            os.system("chmod +x training/download_trained_models.sh")
            os.system("./training/download_trained_models.sh")

        if opt.reload_model_path == "" and opt.SVR:
            opt.dir_name = "./training/trained_models/atlasnet_singleview_1_sphere"
        elif opt.reload_model_path == "" and not opt.SVR:
            opt.dir_name = "./training/trained_models/atlasnet_autoencoder_1_sphere"


    if exists(join(opt.dir_name, "options.json")):
        # Reload parameters from options.txt if it exists
        with open(join(opt.dir_name, "options.json"), 'r') as f:
            my_opt_dict = json.load(f)
        my_opt_dict.pop("run_single_eval")
        my_opt_dict.pop("no_metro")
        my_opt_dict.pop("train_only_encoder")
        my_opt_dict.pop("no_learning")
        my_opt_dict.pop("demo")
        my_opt_dict.pop("demo_input_path")
        my_opt_dict.pop("dir_name")
        my_opt_dict.pop("mode")
        my_opt_dict.pop("custom_data")
        my_opt_dict.pop("train_pc_path")
        my_opt_dict.pop("eval_pc_path")
        my_opt_dict.pop("test_pc_path")
        for key in my_opt_dict.keys():
            opt[key] = my_opt_dict[key]
        if not opt.demo:
            print("Modifying input arguments to match network in dirname")
            my_utils.cyan_print("PARAMETER: ")
            for a in my_opt_dict:
                print(
                    "         "
                    + colored(a, "yellow")
                    + " : "
                    + colored(str(my_opt_dict[a]), "cyan")
                )

    # Hard code dimension of the template.
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    opt.dim_template = dim_template_dict[opt.template_type]

    # Visdom env
    opt.env = opt.env + opt.dir_name.split('/')[-1]

    return opt
