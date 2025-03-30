# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import pandas as pd
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import Subset
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
import h5py

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
     
class DepthDataset(Dataset):
    def __init__(self, root_dir, input_size=224, train=True):
        self.root_dir = root_dir
        self.if_train = train
        self.input_size = input_size

        self.train_file = os.path.join(root_dir, "depth_train.h5")
        self.test_file = os.path.join(root_dir, "depth_test.h5")

        with h5py.File(self.train_file, "r") as f:
            self.train_len = len(f["image"])
        with h5py.File(self.test_file, "r") as f:
            self.test_len = len(f["image"])

    def __len__(self):
        return self.train_len if self.if_train else self.test_len

    def __getitem__(self, idx):
        file_path = self.train_file if self.if_train else self.test_file
        with h5py.File(file_path, "r") as f:
            image = f["image"][idx]    # (H, W, 3), dtype=uint8
            depth = f["depth"][idx]    # (H, W, 1) or (H, W)

        # --- Convert image: (H, W, C) uint8 → float32 Tensor in (C, H, W) ---
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.  # (3, H, W)

        # --- Convert depth: (H, W, 1) or (H, W) → (1, H, W) ---
        depth = np.squeeze(depth)  # remove singleton dim
        depth = torch.from_numpy(depth).float().unsqueeze(0)       # (1, H, W)

        # --- Resize both to (C, 224, 224) ---
        image = F.interpolate(image.unsqueeze(0), size=(self.input_size, self.input_size), mode='bilinear', align_corners=False).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=(self.input_size, self.input_size), mode='bilinear', align_corners=False).squeeze(0)

        return image, depth

def build_depth_dataset(is_train, args):
    dataset = DepthDataset(
        root_dir=args.depth_data_path,
        train=is_train
    )
    return dataset