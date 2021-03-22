"""Construct Residual Network."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class Bottleneck(nn.Module):

  expansion = 4

  def __init__(self, inplanes, planes, stride=1,
               dilation=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes,
                           planes,
                           kernel_size=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(planes, momentum=3e-4)

    self.conv2 = nn.Conv2d(planes,
                           planes,
                           kernel_size=3,
                           stride=stride,
                           padding=dilation,
                           dilation=dilation, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, momentum=3e-4)
    self.conv3 = nn.Conv2d(planes,
                           planes * Bottleneck.expansion,
                           kernel_size=1,
                           bias=False)
    self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion,
                              momentum=3e-4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.dilation = dilation
    self.stride = stride

  def forward(self, x):

    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class conv1(nn.Module):

  def __init__(self):

    super(conv1, self).__init__()
    self.inplanes = 128

    #self.conv1 = nn.Conv2d(3,
    #                       64,
    #                       kernel_size=7,
    #                       stride=2,
    #                       padding=3,
    #                       bias=False)
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64,
                  kernel_size=3,
                  stride=2,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(64, momentum=3e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(64, momentum=3e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False))
    self.bn1 = nn.BatchNorm2d(128, momentum=3e-4)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


  def forward(self, x):

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    return x


class ResnetBackbone(nn.Module):

  def __init__(self, blocks, strides, dilations, config):

    super(ResnetBackbone, self).__init__()

    self.inplanes = 128
    self.conv1 = conv1()

    self.res2 = self._make_layer(
        Bottleneck, 64, blocks[0], stride=strides[0], dilation=dilations[0])
    self.res3 = self._make_layer(
        Bottleneck, 128, blocks[1], stride=strides[1], dilation=dilations[1])

    self.res4 = self._make_layer(
        Bottleneck, 256, blocks[2], stride=strides[2], dilation=dilations[2])
    self.res5 = self._make_layer(
        Bottleneck, 512, blocks[3], stride=strides[3], dilation=dilations[3])

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, _BatchNorm):
        m.weight.data.fill_(1)
        if m.bias is not None:
          m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1, 
                  grids=None):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion, momentum=3e-4))

    layers = []
    if grids is None:
      grids = [1] * blocks

    if dilation == 1 or dilation == 2:
      layers.append(block(self.inplanes, planes, stride, dilation=1, 
                          downsample=downsample,))
    elif dilation == 4:
      layers.append(block(self.inplanes, planes, stride, dilation=2, 
                          downsample=downsample,))
    else:
      raise RuntimeError('=> unknown dilation size: {}'.format(dilation))

    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, 
                          dilation=dilation*grids[i],))

    return nn.Sequential(*layers)

  def forward(self, x):
    conv1 = self.conv1(x)

    res2 = self.res2(conv1)
    res3 = self.res3(res2)
    res4 = self.res4(res3)
    res5 = self.res5(res4)

    return res2, res3, res4, res5
