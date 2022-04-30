#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/18, homeway'

import random
import numpy as np
import trans
import torch, copy
from torch.nn import functional as F


class SEAT:
    def __init__(self, encoder, m=3.1622776601683795, delta=0.001, N=100, bounds=[-1, 1], device=None):
        self.encoder = encoder
        self.device = device
        if device is None:
            self.device = next(encoder.parameters()).device
        self.zero = torch.zeros(1, device=self.device)
        self.mm = torch.tensor(m * m).to(self.device)

        self.N = N
        self.delta = delta
        self.bounds = bounds
        self.reset()

    def reset(self):
        self.count = 0
        self.hist_feats = []
        self.adv = trans.Adv(model=copy.deepcopy(self.encoder), bounds=self.bounds)
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.SGD(self.encoder.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    def create_pairs(self, x0, y0, trans_pos="pgd", trans_neg="rotation"):
        """
        Statement from paper:
        𝑥_pos is a positive sample which 𝑓 should consider closed to 𝑥0
        𝑥_neg is a negative sample, a different natural image from 𝑥0 that 𝑓 should consider far away from 𝑥0
        𝑥_neg is generated from random transformations (e.g. rotation, scaling, cropping, etc.)
        """
        # generate positive sample
        if trans_pos == "pgd":
            x_pos = self.adv.pgd(copy.deepcopy(x0), y0, eps=8./255, steps=40)
        elif trans_pos == "cw":
            x_pos = self.adv.cw(copy.deepcopy(x0), y0)
        elif trans_pos == "fgsm":
            x_pos = self.adv.fgsm(copy.deepcopy(x0), y0)
        elif trans_pos == "bim":
            x_pos = self.adv.bim(copy.deepcopy(x0), y0)
        else:
            raise NotImplementedError(f"-> Error! transform method:{trans_pos} not implemented!")

        # generate negative sample
        if trans_neg == "rotation":
            transform = trans.Rotation()
        elif trans_neg == "scaling":
            transform = trans.RandomResizedCropLayer()
        elif trans_neg == "flip":
            transform = trans.HorizontalFlipRandomCrop(max_range=min(-self.bounds[0], self.bounds[1]))
        else:
            raise NotImplementedError(f"-> Error! transform method:{trans_neg} not implemented!")
        x_neg = transform(copy.deepcopy(x0))

        # return x0, x_pos, x_neg
        x0 = x0.to(self.device)
        x_pos = x_pos.to(self.device)
        x_neg = x_neg.to(self.device)
        return x0, x_pos, x_neg

    def fine_tuning(self, x, y):
        trans_neg = ["rotation", "scaling", "flip"][random.randint(0, 2)]
        self.encoder.train()
        self.optimizer.zero_grad()
        x0, x_pos, x_neg = self.create_pairs(x, y, trans_neg=trans_neg)
        feat = self.encoder.feats_forward(x0)
        feat_pos = self.encoder.feats_forward(x_pos)
        feat_neg = self.encoder.feats_forward(x_neg)
        loss_pos = self.criterion(feat, feat_pos)
        loss_neg = torch.max(self.zero, self.mm - self.criterion(feat, feat_neg))[0]
        loss = loss_pos + loss_neg
        loss.backward()
        self.optimizer.step()
        return loss_pos.item(), loss_neg.item()

    def query(self, x):
        self.reset()
        self.encoder.eval()
        x = x.to(self.device)
        feats = self.encoder.feats_forward(x).detach().cpu()

        alarm = False
        dist = np.zeros(len(x))
        pred = np.zeros(len(x), dtype=np.int32)
        for idx in range(len(x)):
            for hist_feats in self.hist_feats:
                dist[idx] = float(F.mse_loss(feats[idx], hist_feats))
                if dist[idx] < self.delta:
                    pred[idx] = 1
                    self.count += 1
                    # send extraction alarm
                    if self.count > self.N:
                        alarm = True
                    break
            self.hist_feats.append(feats[idx])
        return alarm, pred, dist












