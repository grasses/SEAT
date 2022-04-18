#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/18, homeway'

import trans
import torch
from torch import nn
import copy
from torch.nn import functional as F


class SEAT:
    def __init__(self, encoder, m=0.2, bounds=[-1, 1]):
        self.encoder = encoder
        self.device = next(encoder.parameters()).device
        self.bounds = bounds
        self.z = torch.zeros(1, device=self.device)
        self.m2 = torch.mul(m, m).to(self.device)

        self.history = []
        self.adv = trans.Adv(model=self.encoder, bounds=self.bounds)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.01)

    def create_pairs(self, x0, y0, trans_pos="pgd", trans_neg="rotation"):
        """
        Statement from paper:
        𝑥_pos is a positive sample which 𝑓 should consider closed to 𝑥0
        𝑥_neg is a negative sample, a different natural image from 𝑥0 that 𝑓 should consider far away from 𝑥0
        𝑥_neg is generated from random transformations (e.g. rotation, scaling, cropping, etc.)
        """
        # generate positive sample
        if trans_pos == "pgd":
            transform = self.adv.pgd
        elif trans_pos == "cw":
            transform = self.adv.cw
        else:
            raise NotImplementedError(f"-> Error! transform method:{trans_pos} not implemented!")
        x_pos = transform(copy.deepcopy(x0), y0)

        # generate negative sample
        if trans_neg == "rotation":
            transform = trans.Rotation()
        elif trans_neg == "scaling":
            transform = trans.RandomResizedCropLayer()
        elif trans_neg == "flip":
            transform = trans.HorizontalFlipRandomCrop(max_range=min(self.bounds[0], self.bounds[1]))
        else:
            raise NotImplementedError(f"-> Error! transform method:{trans_neg} not implemented!")
        x_neg = transform(copy.deepcopy(x0))

        x0 = x0.to(self.device)
        x_pos = x_pos.to(self.device)
        x_neg = x_neg.to(self.device)
        return x0, x_pos, x_neg

    def fine_tuning(self, x, y):
        self.encoder.train()
        self.optimizer.zero_grad()
        x, x_pos, x_neg = self.create_pairs(x, y)

        self.encoder.train()
        feat = self.encoder.feats_forward(x)
        feat_pos = self.encoder.feats_forward(x_pos)
        feat_neg = self.encoder.feats_forward(x_neg)

        loss = self.criterion(feat, feat_pos) + torch.max(self.z, self.m2 - self.criterion(feat, feat_neg))
        loss.backward()
        self.optimizer.step()
        return loss


    def detect(self, query):
        pass