import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import PatchGuard.nets.bagnet as bagnet
import PatchGuard.nets.resnet
from PatchGuard.utils.defense_utils import *

import os
import joblib
import argparse
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
from math import ceil
import PIL

class PatchGuard(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = bagnet.bagnet17(pretrained=True, clip_range=None, aggregation='none').cuda()
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 10)
        self.net = torch.nn.DataParallel(self.net)
        checkpoint = torch.load('PatchGuard/checkpoints/bagnet17_192_cifar.pth')
        self.net.load_state_dict(checkpoint['net'])
        self.preprocessing = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        rf_size = 17
        rf_stride = 8
        patch_size = 30
        self.window_size = ceil((patch_size + rf_size - 1) / rf_stride)

    def forward(self, *x):
        if type(x) == tuple:
            assert len(x) == 1
            x = x[0]
        z = (x + 0.5) * 255
        z = F.interpolate(z, 192, mode='bicubic') / 255
        z = self.net(self.preprocessing(z)).detach().cpu().numpy()
        preds = []
        for i in range(z.shape[0]):
            local_feature = z[i]
            pred = masking_defense(local_feature, thres=0.0, window_shape=[self.window_size, self.window_size])
            preds.append(pred)
        return torch.tensor(preds).cuda()

    def predict(self, x):
        return self.forward(x)


class PatchGuardNoDefense(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = bagnet.bagnet17(pretrained=True, clip_range=None, aggregation='mean').cuda()
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 10)
        self.net = torch.nn.DataParallel(self.net)
        checkpoint = torch.load('PatchGuard/checkpoints/bagnet17_192_cifar.pth')
        self.net.load_state_dict(checkpoint['net'])
        self.preprocessing = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        rf_size = 17
        rf_stride = 8
        patch_size = 30
        self.window_size = ceil((patch_size + rf_size - 1) / rf_stride)

    def forward(self, *x):
        if type(x) == tuple:
            assert len(x) == 1
            x = x[0]
        z = (x + 0.5) * 255
        z = F.interpolate(z, 192, mode='bicubic') / 255
        z = self.net(self.preprocessing(z)).detach().cpu().numpy()
        preds = []
        for i in range(z.shape[0]):
            local_feature = z[i]
            pred = np.argmax(
                local_feature)
            preds.append(pred)
        return torch.tensor(preds).cuda()

    def predict(self, x):
        return self.forward(x)
