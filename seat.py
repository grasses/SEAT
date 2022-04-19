#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/18, homeway'

import trans
import torch, copy
from torch.nn import functional as F


class SEAT:
    def __init__(self, encoder, m=3.1622776601683795, delta=0.1, N=200, bounds=[-1, 1]):
        self.encoder = encoder
        self.device = next(encoder.parameters()).device
        self.zero = torch.zeros(1, device=self.device)
        self.mm = torch.tensor(m * m).to(self.device)

        self.N = N
        self.delta = delta
        self.count = 0
        self.history_feats = []
        self.bounds = bounds
        self.adv = trans.Adv(model=copy.deepcopy(self.encoder), bounds=self.bounds)
        self.criterion = torch.nn.MSELoss(reduce="sum")
        self.optimizer = torch.optim.SGD(self.encoder.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

    def reset(self):
        self.count = 0
        self.history_feats = []

    def create_pairs(self, x0, y0, trans_pos="pgd", trans_neg="scaling"):
        """
        Statement from paper:
        𝑥_pos is a positive sample which 𝑓 should consider closed to 𝑥0
        𝑥_neg is a negative sample, a different natural image from 𝑥0 that 𝑓 should consider far away from 𝑥0
        𝑥_neg is generated from random transformations (e.g. rotation, scaling, cropping, etc.)
        """
        # generate positive sample
        if trans_pos == "pgd":
            x_pos = self.adv.pgd(copy.deepcopy(x0), y0)
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
        feat = self.encoder.feats_forward(x)
        feat_pos = self.encoder.feats_forward(x_pos)
        feat_neg = self.encoder.feats_forward(x_neg)
        loss1 = self.criterion(feat, feat_pos)
        loss2 = torch.max(self.zero, self.mm - self.criterion(feat, feat_neg))[0]
        loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()
        return loss1.item(), loss2.item()

    def detect(self, query):
        self.encoder.eval()
        query = query.to(self.device)
        feats = self.encoder.feats_forward(query).cpu()

        adv_dist = []
        for idx in range(len(query)):
            for hist_feats in self.history_feats:
                dist = F.mse_loss(feats[idx], hist_feats)
                if dist < self.delta:
                    self.count += 1
                    adv_dist.append(dist.item())
                    break

                if self.count > self.N:
                    pass
                    #print(f"-> Find adversary, malicious query count:{self.count}!!!")
            self.history_feats.append(feats[idx])
        return adv_dist












