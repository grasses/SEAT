from torch.nn import Module
from torch import nn
import torch
import os.path as osp
ROOT = osp.abspath(osp.dirname(osp.dirname(__file__)))

class LeNet5(Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)
        self.relu5 = nn.ReLU()

    def feats_forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        return y

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


def lenet5(pretrained=False):
    model = LeNet5(num_classes=10)
    if pretrained:
        state_dict = load_pretrained_lenet5()
        model.load_state_dict(state_dict)
    return model


def load_pretrained_lenet5():
    target_file = osp.join(ROOT, f"models/ckpt/MNIST_LeNet5.pt")
    if not osp.exists(target_file):
        raise FileNotFoundError(f"-> pretrained file:{target_file} not found!")
    return torch.load(target_file, map_location="cpu")
