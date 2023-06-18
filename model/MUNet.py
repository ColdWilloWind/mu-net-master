import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

class SE(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channel, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1, stride=1, padding=0)
        self.excitation = nn.Conv2d(in_channel//ratio, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.softmax(out)


class ConvBlock(nn.Module):
    """不带通道注意力机制的卷积块：卷积+归一化+激活函数"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SEBolck(nn.Module):
    """带通道注意力机制的卷积块"""
    def __init__(self, in_channel, out_channel, k, s, p, ratio=16):
        super(SEBolck, self).__init__()
        self.conv_block = ConvBlock(in_channel, out_channel, k, s, p)
        self.se = SE(out_channel, ratio)

    def forward(self, x):
        out = self.conv_block(x)
        se = self.se(out)
        out = out * se
        return out


class ResBlock(nn.Module):
    """残差卷积块"""
    def __init__(self, in_channel, out_channel, kernel_size, strides, padding):
        super(ResBlock, self).__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=strides, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size
                               , stride=strides, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        x = self.bn3(self.conv3(x))
        return F.relu(out + x)


class ResSEConvBlock(nn.Module):
    """带通道注意力机制的残差卷积块"""
    def __init__(self, in_channel, out_channel, k, s, p, ratio=16):
        super(ResSEConvBlock, self).__init__()
        self.conv_block = ResBlock(in_channel, out_channel, k, s, p)
        self.se = SE(out_channel, ratio)

    def forward(self, x):
        out = self.conv_block(x)
        se = self.se(out)
        out = out * se
        return out

class MUNet_512(nn.Module):
    def __init__(self):
        super(MUNet_512, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(512)
        self.route_1 = nn.Sequential(
            ResBlock(2, 64, 7, 1, 3),
            ResSEConvBlock(64, 128, 7, 2, 3),
            ResSEConvBlock(128, 128, 7, 2, 3),
            ResSEConvBlock(128, 128, 7, 2, 3),
            ResSEConvBlock(128, 128, 7, 2, 3),
            ResSEConvBlock(128, 128, 7, 2, 3)
        )
        self.upsample = nn.AdaptiveAvgPool2d(64)
        self.route_2 = nn.Sequential(
            ResBlock(2, 64, 5, 1, 2),
            ResSEConvBlock(64, 128, 5, 2, 2),
            ResSEConvBlock(128, 128, 5, 2, 2),
            ResSEConvBlock(128, 128, 5, 2, 2),
        )
        self.route_3 = nn.Sequential(
            ResSEConvBlock(256, 128, 3, 2, 1),
            ResSEConvBlock(128, 128, 3, 2, 1),
            ResSEConvBlock(128, 128, 3, 2, 1)
        )
        self.fullyconnect = nn.Sequential(
            nn.Conv2d(128, 6, 8, 1, 0),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, 1, 1, 0)
        )
    def forward(self, input):
        input = self.pooling(input)
        x_1 = self.route_1(input)
        x_1 = self.upsample(x_1)
        x_2 = self.route_2(input)
        x_concat = torch.cat([x_1, x_2], dim=1)
        x_3 = self.route_3(x_concat)
        parameters = self.fullyconnect(x_3)
        parameters = parameters.view(-1, 2, 3)
        return parameters

