"""VCOPN"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class Wrapper_Model(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len, dims=128, with_contrast=True):
        """
        Args:
            feature_size (int): 512
        """
        super(Wrapper_Model, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.class_num = tuple_len
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc7 = nn.Linear(self.feature_size*2, 512)
        self.fc8 = nn.Linear(512, self.class_num)

        # self.contrast_dropout = nn.Dropout(p=0.5)
        # self.contrast_relu = nn.ReLU(inplace=True)
        self.with_contrast = with_contrast
        if with_contrast:
            self.contrast_fc1 = nn.Linear(self.feature_size*2, dims)
        # self.contrast_fc2 = nn.Linear(512, dims)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(2):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))
        pf = torch.cat(f,dim=1)

        pf7 = self.fc7(pf)
        pf7 = self.relu(pf7)
        h_d = self.dropout(pf7)
        h = self.fc8(h_d)  # logits

        if self.with_contrast:
            ff = self.contrast_fc1(pf)
        # ff = self.contrast_relu(ff)
        # ff = self.contrast_dropout(ff)
        # ff = self.contrast_fc2(ff)

        # h = h.squeeze()
        if self.with_contrast:
            return h, ff
        else:
            return h
