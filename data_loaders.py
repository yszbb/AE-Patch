import json
import os
import random
import warnings
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, ImageFolder

from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST

warnings.filterwarnings('ignore')


# code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
# Improved Regularization of Convolutional Neural Networks with Cutout.
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def build_dvscifar(root, frames_number=30):
    def t(data):
        aug = transforms.Compose([transforms.Resize(size=(48, 48)), transforms.RandomHorizontalFlip()])
        data = torch.from_numpy(data)
        data = aug(data).float()

        return data

    def tt(data):
        aug = transforms.Resize(size=(48, 48))
        data = torch.from_numpy(data)
        data = aug(data).float()

        return data

    data1 = CIFAR10DVS(root=root, data_type='frame', frames_number=frames_number, split_by='number', transform=t)

    # 将只有训练数据集的 CIFAR10DVS 进行划分
    train_dataset, _ = torch.utils.data.random_split(data1, [9000, 1000], generator=torch.Generator().manual_seed(42))

    data2 = CIFAR10DVS(root=root, data_type='frame', frames_number=frames_number, split_by='number', transform=tt)
    _, val_dataset = torch.utils.data.random_split(data2, [9000, 1000], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset, None


def build_dvscifar_patch(root, frames_number=30):
    def t(data):
        aug = transforms.Compose([transforms.Resize(size=(64, 64)), transforms.RandomHorizontalFlip()])
        data = torch.from_numpy(data)
        data = aug(data).float()

        return data

    def tt(data):
        aug = transforms.Resize(size=(64, 64))
        data = torch.from_numpy(data)
        data = aug(data).float()

        return data

    data1 = CIFAR10DVS(root=root, data_type='frame', frames_number=frames_number, split_by='number', transform=t)

    # 将只有训练数据集的 CIFAR10DVS 进行划分
    train_dataset, _ = torch.utils.data.random_split(data1, [1200, 8800], generator=torch.Generator().manual_seed(42))

    data2 = CIFAR10DVS(root=root, data_type='frame', frames_number=frames_number, split_by='number', transform=tt)
    _, val_dataset = torch.utils.data.random_split(data2, [9700, 300], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset, None


def build_dvsgesture(root, frames_number=30):
    def t(data):
        aug = transforms.Compose([transforms.Resize(size=(64, 64)), transforms.RandomHorizontalFlip()])
        data = torch.from_numpy(data)
        data = aug(data).float()

        return data

    def tt(data):
        aug = transforms.Resize(size=(64, 64))
        data = torch.from_numpy(data)
        data = aug(data).float()

        return data

    train_dataset = DVS128Gesture(root=root, train=True, data_type='frame', frames_number=frames_number,
                                  split_by='number', transform=t)

    val_dataset = DVS128Gesture(root=root, train=False, data_type='frame', frames_number=frames_number,
                                split_by='number', transform=tt)

    return train_dataset, val_dataset, None


def build_nmnist(root, frame_number):
    def t(data):
        aug = transforms.RandomHorizontalFlip()
        data = torch.from_numpy(data)
        data = aug(data).float()

        return data

    def tt(data):
        data = torch.from_numpy(data).float()
        return data

    train_dataset = NMNIST(root=root, train=True, data_type='frame', frames_number=frame_number, split_by='number',
                           transform=t)

    val_dataset = NMNIST(root=root, train=False, data_type='frame', frames_number=frame_number, split_by='number',
                         transform=tt)

    return train_dataset, val_dataset, None
