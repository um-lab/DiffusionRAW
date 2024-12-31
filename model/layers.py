import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, bias=True, norm_layer=False):
        super(ConvBlock, self).__init__()
        
        if norm_layer:
            bias = False
        self.basicconv = nn.Sequential()
        self.basicconv.add_module(
            'conv', nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias))
        if norm_layer:
            self.basicconv.add_module('bn', nn.BatchNorm2d(out_planes))
        self.basicconv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.basicconv(x)


class Conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=int(in_planes), 
            out_channels=int(out_planes), 
            kernel_size=3, 
            stride=1,
            padding=1,
            bias=True
            )

    def forward(self, x):
        out = self.conv(x)
        return out


class Conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=int(in_planes), 
            out_channels=int(out_planes), 
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=True
            )

    def forward(self, x):
        out = self.conv(x)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return  F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


def downsample(x):
    """Downsample input tensor by a factor of 0.5
    """
    return  F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)