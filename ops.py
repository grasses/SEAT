#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/18, homeway'


import trans
import torchvision
import torchvision.transforms as transforms
from models.lenet import *
from models.vgg import *
ROOT = osp.abspath(osp.dirname(__file__))


def load_model(arch):
    if arch.lower() == "lenet":
        return lenet(pretrained=True)
    elif "vgg" in arch.lower():
        return eval(f"{arch}(pretrained=True)")
    else:
        raise NotImplementedError(f"-> arch:{arch} not implemented!!")


def load_data(task, batch_size=128, query_size=1000):
    if task.lower() == "cifar10":
        return load_cifar10(batch_size=batch_size, query_size=query_size)
    elif task.lower() == "mnist":
        return load_mnist(batch_size=batch_size, query_size=query_size)
    else:
        raise NotImplementedError(f"-> dataset:{task} not implemented!!")


def load_cifar10(root=osp.join(ROOT, "datasets/data"), batch_size=128, query_size=1000):
    mean = (0.43768206, 0.44376972, 0.47280434)
    std = (0.19803014, 0.20101564, 0.19703615)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=query_size, shuffle=False, num_workers=2)
    bounds = trans.get_bounds(mean=mean, std=std)
    return train_loader, test_loader, bounds


def load_mnist(root=osp.join(ROOT, "datasets/data"), batch_size=128, query_size=1000):
    mean = (0.1307)
    std = (0.3081)
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
         ]
    )
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=query_size, shuffle=False, num_workers=2)
    bounds = trans.get_bounds(mean=mean, std=std)
    return train_loader, test_loader, bounds
