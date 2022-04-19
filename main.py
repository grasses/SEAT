#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/01/25, homeway'


"""
This is the implementation of paper: "SEAT: Similarity Encoder by Adversarial Training for Detecting Model Extraction Attack Queries".
We use the pretrain model downloaded from: https://github.com/huyvnphan/PyTorch_CIFAR10
"""
import numpy as np
import os.path as osp
import os, shutil, torch, zipfile, gdown
import torchvision
import torchvision.transforms as transforms
import argparse
from models.vgg import vgg16_bn
from tqdm import tqdm
from seat import SEAT
ROOT = osp.abspath(osp.dirname(osp.dirname(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--data_root", default=osp.join(ROOT, "datasets/data"))
parser.add_argument("--cpkt_root", default=osp.join(ROOT, "models/ckpt"))
args = parser.parse_args()
args.device = torch.device(f"cuda:{args.device}")


def get_bounds(mean, std):
    bounds = [-1, 1]
    if type(mean) == type(()):
        c = len(mean)
        _min = (np.zeros([c]) - np.array(mean)) / np.array([std])
        _max = (np.ones([c]) - np.array(mean)) / np.array([std])
        bounds = [np.min(_min).item(), np.max(_max).item()]
    elif type(mean) == float:
        bounds = [(0.0 - mean) / std, (1.0 - mean) / std]
    return bounds

def load_data(root="./datasets/data", batch_size=128):
    mean = (0.43768206, 0.44376972, 0.47280434)
    std = (0.19803014, 0.20101564, 0.19703615)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
             (0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)
         )]
    )
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    bounds = get_bounds(mean=mean, std=std)
    return train_loader, test_loader, bounds


def load_pretrained_encoder(arch="vgg16_bn"):
    local_root = osp.join(ROOT, "models")
    target_file = osp.join(local_root, f"ckpt/{arch}.pt")
    local_file = osp.join(local_root, "state_dicts.zip")
    if not osp.exists(local_file):
        print("-> Pretrained model not found!! download now...")
        remote_url = "https://drive.google.com/u/0/uc?id=17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq&export=download"
        gdown.download(remote_url, local_file, quiet=False)
    if not osp.exists(target_file):
        print("-> Unzip state_dicts.zip file...")
        with zipfile.ZipFile(local_file, "r") as zip_ref:
            zip_ref.extractall(local_root)
            for file in os.listdir(osp.join(local_root, "state_dicts")):
                source = osp.join(local_root, "state_dicts", file)
                destination = osp.join(local_root, "ckpt", file)
                shutil.move(source, destination)
    return torch.load(target_file, map_location="cpu")


def fine_tuning_encoder(seat, train_loader, epochs=50):
    '''
    for step, (x, y) in tqdm(enumerate(train_loader), desc="fine-tuning now..."):
        loss = seat.fine_tuning(x, y)
        print(f"-> step:{step} loss:{loss.item()}")
    '''
    '''
    model = seat.encoder.to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    '''
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            loss1, loss2 = seat.fine_tuning(x, y)
            if step % 10 == 0:
                print(f"-> epoch:{epoch} step:{step} loss_positive:{loss1} loss_negative:{loss2}")
        # TODO: save model

def evaluate_SEAT(seat, test_loader):
    pass



def main():
    # for the basic operations
    for path in [args.data_root, args.cpkt_root]:
        try:
            os.makedirs(path)
        except Exception as e:
            pass

    """step1: load dataset"""
    train_loader, test_loader, bounds = load_data()

    """step2: load pretrained encoder"""
    load_pretrained_encoder(arch="vgg16_bn")
    encoder = vgg16_bn(pretrained=True)
    encoder.to(args.device)

    """step3: fine-tuning similarity encoder with contrastive loss"""
    seat = SEAT(encoder, bounds=bounds)
    fine_tuning_encoder(seat=seat, train_loader=train_loader)

    """step4: evaluate similarity encoder"""
    evaluate_SEAT(seat=seat, test_loader=test_loader)


if __name__ == "__main__":
    main()












