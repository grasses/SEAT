#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/18, homeway'


"""
This is the implementation of paper: "SEAT: Similarity Encoder by Adversarial Training for Detecting Model Extraction Attack Queries".
We use the pretrain model downloaded from: https://github.com/huyvnphan/PyTorch_CIFAR10
"""

import random
import numpy as np
import os.path as osp
import os, copy, shutil, zipfile, gdown
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import trans
from models.vgg import vgg16_bn
from tqdm import tqdm
from seat import SEAT
ROOT = osp.abspath(osp.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--seed", type=int, default=999999)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--arch", type=str, default="vgg16_bn")
parser.add_argument("--data_root", default=osp.join(ROOT, "datasets/data"))
parser.add_argument("--cpkt_root", default=osp.join(ROOT, "models/ckpt"))
args = parser.parse_args()
args.device = torch.device(f"cuda:{args.device}")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_bounds(mean, std):
    '''
    get bound of dataset
    :param mean: list, float
    :param std: list, float
    :return: list, [lower_bound, upper_bound]
    '''
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


def load_pretrained_encoder(arch):
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


def fine_tuning_encoder(seat, train_loader, epochs=50, arch="vgg16_bn"):
    """
    fine-tuning encoder using last layer of CNN feature maps as latent space of encoder
    :param seat: SEAT object
    :param train_loader:
    :param epochs:
    :return: None
    """
    path = osp.join(ROOT, f"models/ckpt/enc_{arch}_{epochs}.pt")
    if osp.exists(path):
        print(f"-> load pretrained encoder from: {path}\n")
        weights = torch.load(path, map_location=args.device)
        seat.encoder.load_state_dict(weights)
        return

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(seat.optimizer, T_max=epochs)
    for epoch in range(epochs):
        path = osp.join(ROOT, f"models/ckpt/enc_{arch}_{epoch + 1}.pt")
        if osp.exists(path):
            weights = torch.load(path, map_location=args.device)
            seat.encoder.load_state_dict(weights)
            scheduler.step()
            continue
        pbar = tqdm(enumerate(train_loader))
        size = len(train_loader)
        sum_loss1 = 0.0
        sum_loss2 = 0.0
        for step, (x, y) in pbar:
            loss1, loss2 = seat.fine_tuning(x, y)
            pbar.set_description(
                "-> Epoch{:d}: [{:d}/{:d}] loss_pos:{:.8f}+loss_neg:{:.8f}={:.8f}\t".format(
                    epoch, step, size,
                    loss1, loss2,
                    loss1 + loss2)
            )
            sum_loss1 += loss1
            sum_loss2 += loss2
            pbar.update(1)
        scheduler.step()
        torch.save(copy.deepcopy(seat.encoder).state_dict(), path)
        print(f"-> Epoch:{epoch} l1:{1.0*sum_loss1/size} l2:{1.0*sum_loss2/size}  save model to: {path}\n")
    return seat


def evaluate_SEAT(seat, test_loader):
    size = len(test_loader)
    FP, TP, FN, TN = 0, 0, 0, 0

    print("-> detect benign query")
    pbar = tqdm(enumerate(test_loader))
    for step, (x, y) in pbar:
        query = x.to(args.device)
        alarm, pred, dist = seat.detect(query=query)
        FN += np.sum(pred)
        TP += (len(query) - np.sum(pred))
        pbar.set_description(
            f"-> [{step}/{size}] adv_query_count:{seat.count} total_query:{len(seat.history_feats)}")

    seat.reset()
    adv = trans.Adv(model=copy.deepcopy(seat.encoder), bounds=seat.bounds)
    print("\n-> detect malicious query")
    pbar = tqdm(enumerate(test_loader))
    for step, (x, y) in pbar:
        x = x.to(args.device)
        y = torch.randint(0, 10, list(y.shape)).to(args.device)
        query = adv.pgd(x, y, eps=40./255., alpha=40./255., steps=30, random_start=True)
        alarm, pred, dist = seat.detect(query=query)
        TN += np.sum(pred)
        FP += (len(query) - np.sum(pred))
        pbar.set_description(
            f"-> [{step}/{size}] adv_query_count:{seat.count} total_query:{len(seat.history_feats)}")

    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    ACC = 100.0 * (TP + TN) / (FP + TP + FN + TN)
    TPR = 1.0 * TP / (TP + FN)
    FPR = 1.0 * FP / (TN + FP)
    print(f"-> ACC:{ACC} TPR:{TPR} FPR:{FPR} recall:{recall} precision:{precision}")


def main():
    # for the basic operations
    for path in [args.data_root, args.cpkt_root]:
        try:
            os.makedirs(path)
        except Exception as e:
            pass

    print("""\n-> step1: load dataset""")
    train_loader, test_loader, bounds = load_data(batch_size=args.batch_size)

    print("""\n-> step2: load pretrained encoder""")
    load_pretrained_encoder(arch=args.arch)
    encoder = vgg16_bn(pretrained=True)
    encoder.to(args.device)

    print("""\n-> step3: fine-tuning similarity encoder with contrastive loss""")
    seat = SEAT(encoder, bounds=bounds)
    fine_tuning_encoder(seat=seat, train_loader=train_loader, epochs=50, arch=args.arch)

    print("""\n-> step4: evaluate similarity encoder""")
    evaluate_SEAT(seat=seat, test_loader=test_loader)


if __name__ == "__main__":
    main()












