# import system modules
import os
from os.path import join, dirname, exists
import os.path as osp
import sys
import numpy as np
from PIL import Image
import pickle
from easydict import EasyDict
import json
from termcolor import colored
from copy import deepcopy
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
import transfer.atlasnet.auxiliary.my_utils as my_utils
import transfer.atlasnet.dataset.pointcloud_processor as pointcloud_processor


class ShapeNet(data.Dataset):
    """
    Shapenet Dataloader
    Uses Shapenet V1
    Make sure to respect shapenet Licence.
    Author: Thibault Groueix 01.11.2019
    """

    def __init__(self, opt, mode='train', test_pc=None):
        self.opt = opt
        self.num_sample = opt.number_points if mode == 'train' else 2500
        self.mode = mode
        self.init_normalization()
        self.init_singleview()

        if not opt.demo and not opt.custom_data:
            my_utils.red_print('Create Shapenet Dataset...')
            # Define core path array
            self.datapath = []
            self.category_datapath = {}


            # Load classes
            self.pointcloud_path = join(dirname(__file__), 'data/ShapeNetV1PointCloud')
            self.image_path = join(dirname(__file__), 'data/ShapeNetV1Renderings')

            # Load taxonomy file
            self.taxonomy_path = join(dirname(__file__), 'data/taxonomy.json')
            if not exists(self.taxonomy_path):
                os.system("chmod +x dataset/download_shapenet_pointclouds.sh")
                os.system("./dataset/download_shapenet_pointclouds.sh")

            self.classes = [x for x in next(os.walk(self.pointcloud_path))[1]]
            with open(self.taxonomy_path, 'r') as f:
                self.taxonomy = json.load(f)

            self.id2names = {}
            self.names2id = {}
            for dict_class in self.taxonomy:
                if dict_class['synsetId'] in self.classes:
                    name = dict_class['name'].split(sep=',')[0]
                    self.id2names[dict_class['synsetId']] = name
                    self.names2id[name] = dict_class['synsetId']

            # Select classes
            if opt.shapenet13:
                opt.class_choice = ["airplane", "bench", "cabinet", "car", "chair", "display", "lamp", "loudspeaker",
                                    "rifle", "sofa", "table", "telephone", "vessel"]

            if len(opt.class_choice) > 0:
                new_classes = []
                for category in opt.class_choice:
                    new_classes.append(self.names2id[category])
                self.classes = new_classes

            # Create Cache path
            self.path_dataset = join(dirname(__file__), 'data', 'cache')
            if not exists(self.path_dataset):
                os.mkdir(self.path_dataset)
            self.path_dataset = join(self.path_dataset,
                                     self.opt.normalization + str(train) + "_".join(self.opt.class_choice))

            if not exists(self.image_path):
                os.system("chmod +x dataset/download_shapenet_renderings.sh")
                os.system("./dataset/download_shapenet_renderings.sh")

            self.num_image_per_object = 24
            self.idx_image_val = 0

            # Compile list of pointcloud path by selected category
            for category in self.classes:
                dir_pointcloud = join(self.pointcloud_path, category)
                dir_image = join(self.image_path, category)
                list_pointcloud = sorted(os.listdir(dir_pointcloud))
                if self.mode == 'train':
                    list_pointcloud = list_pointcloud[:int(len(list_pointcloud) * 0.8)]
                elif self.mode == 'eval':
                    list_pointcloud = list_pointcloud[int(len(list_pointcloud) * 0.8):]
                print(
                    '    category '
                    + colored(category, "yellow")
                    + "  "
                    + colored(self.id2names[category], "cyan")
                    + ' Number Files :'
                    + colored(str(len(list_pointcloud)), "yellow")
                )

                if len(list_pointcloud) != 0:
                    self.category_datapath[category] = []
                    for pointcloud in list_pointcloud:
                        pointcloud_path = join(dir_pointcloud, pointcloud)
                        image_path = join(dir_image, pointcloud.split(".")[0], "rendering")
                        if not self.opt.SVR or exists(image_path):
                            self.category_datapath[category].append((pointcloud_path, image_path, pointcloud, category))
                        else:
                            my_utils.red_print(f"Rendering not found : {image_path}")

            # Add all retained path to a global vector
            for item in self.classes:
                for pointcloud in self.category_datapath[item]:
                    self.datapath.append(pointcloud)

            # Preprocess and cache files
            self.preprocess()

        elif not opt.demo and opt.custom_data:
            if self.mode == 'train':
                self.data_points = np.load(opt.train_pc_path)
            elif self.mode == 'eval':
                self.data_points = np.load(opt.eval_pc_path)
            elif self.mode == 'test':
                if test_pc is None:
                    self.data_points = np.load(opt.test_pc_path)
                else:
                    self.data_points = test_pc

            self.data_points = torch.from_numpy(self.data_points).float()

    def preprocess(self):
        if exists(self.path_dataset + "info.pkl"):
            # Reload dataset
            my_utils.red_print(f"Reload dataset : {self.path_dataset}")
            with open(self.path_dataset + "info.pkl", "rb") as fp:
                self.data_metadata = pickle.load(fp)

            self.data_points = torch.load(self.path_dataset + "points.pth")
        else:
            # Preprocess dataset and put in cache for future fast reload
            my_utils.red_print("preprocess dataset...")
            self.datas = [self._getitem(i) for i in range(self.__len__())]

            # Concatenate all proccessed files
            self.data_points = [a[0] for a in self.datas]
            self.data_points = torch.cat(self.data_points, 0)

            self.data_metadata = [{'pointcloud_path': a[1], 'image_path': a[2], 'name': a[3], 'category': a[4]} for a in
                                  self.datas]

            # Save in cache
            with open(self.path_dataset + "info.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.data_metadata, fp)
            torch.save(self.data_points, self.path_dataset + "points.pth")

        my_utils.red_print("Dataset Size: " + str(len(self.data_metadata)))

    def init_normalization(self):
        if not self.opt.demo:
            my_utils.red_print("Dataset normalization : " + self.opt.normalization)

        if self.opt.normalization == "UnitBall":
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.opt.normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional

    def init_singleview(self):
        ## Define Image Transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor(),
        ])

        # RandomResizedCrop or RandomCrop
        self.dataAugmentation = transforms.Compose([
            transforms.RandomCrop(127),
            transforms.RandomHorizontalFlip(),
        ])

        self.validating = transforms.Compose([
            transforms.CenterCrop(127),
        ])

    def _getitem(self, index):
        pointcloud_path, image_path, pointcloud, category = self.datapath[index]
        points = np.load(pointcloud_path)
        points = torch.from_numpy(points).float()
        points[:, :3] = self.normalization_function(points[:, :3])
        return points.unsqueeze(0), pointcloud_path, image_path, pointcloud, category

    def __getitem__(self, index):
        if not self.opt.custom_data:
            return_dict = deepcopy(self.data_metadata[index])
            # Point processing
            points = self.data_points[index]
            points = points.clone()
            if self.opt.sample:
                choice = np.random.choice(points.size(0), self.num_sample, replace=True)
                points = points[choice, :]
            return_dict['points'] = points[:, :3].contiguous()

            # Image processing
            if self.opt.SVR:
                if self.mode == 'train':
                    N = np.random.randint(1, self.num_image_per_object)
                    im = Image.open(join(return_dict['image_path'], ShapeNet.int2str(N) + ".png"))
                    im = self.dataAugmentation(im)  # random crop
                elif self.mode == 'eval':
                    im = Image.open(join(return_dict['image_path'], ShapeNet.int2str(self.idx_image_val) + ".png"))
                    im = self.validating(im)  # center crop
                im = self.transforms(im)  # scale
                im = im[:3, :, :]
                return_dict['image'] = im
            return return_dict
        else:
            return_dict = {'points': self.data_points[index]}
            return return_dict

    def __len__(self):
        if self.opt.custom_data:
            return len(self.data_points)
        else:
            return len(self.datapath)

    @staticmethod
    def int2str(N):
        if N < 10:
            return "0" + str(N)
        else:
            return str(N)

    def load(self, path):
        ext = path.split('.')[-1]
        if ext == "npy" or ext == "ply" or ext == "obj":
            return self.load_point_input(path)
        else:
            return self.load_image(path)

    def load_point_input(self, path):
        ext = path.split('.')[-1]
        if ext == "npy":
            points = np.load(path)
        elif ext == "ply" or ext == "obj":
            import pymesh
            points = pymesh.load_mesh(path).vertices
        else:
            print("invalid file extension")

        points = torch.from_numpy(points).float()
        operation = pointcloud_processor.Normalization(points, keep_track=True)
        if self.opt.normalization == "UnitBall":
            operation.normalize_unitL2ball()
        elif self.opt.normalization == "BoundingBox":
            operation.normalize_bounding_box()
        else:
            pass
        return_dict = {
            'points': points,
            'operation': operation,
            'path': path,
        }
        return return_dict

    def load_image(self, path):
        im = Image.open(path)
        im = self.validating(im)
        im = self.transforms(im)
        im = im[:3, :, :]
        return_dict = {
            'image': im.unsqueeze_(0),
            'operation': None,
            'path': path,
        }
        return return_dict


if __name__ == '__main__':
    print('Testing Shapenet dataset')
    opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": True, "sample": True, "npoints": 2500,
           "shapenet13": True}
    d = ShapeNet(EasyDict(opt), mode=False, keep_track=True)
    print(d[1])
    a = len(d)
