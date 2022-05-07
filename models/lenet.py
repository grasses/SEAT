from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import os.path as osp
ROOT = osp.abspath(osp.dirname(osp.dirname(__file__)))


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.feats_forward(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x

    def feats_forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        return torch.flatten(x, 1)


def lenet(pretrained=False, num_classes=10):
    model = LeNet(num_classes=num_classes)
    if pretrained:
        state_dict = load_pretrained_lenet()
        model.load_state_dict(state_dict)
    return model


def load_pretrained_lenet():
    target_file = osp.join(ROOT, f"models/ckpt/MNIST_LeNet.pt")
    if not osp.exists(target_file):
        raise FileNotFoundError(f"-> pretrained file:{target_file} not found!")
    return torch.load(target_file, map_location="cpu")