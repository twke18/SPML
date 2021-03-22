"""Build Spatial Pyramid Pooling Module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):

  def __init__(self, in_channels, out_channels,
               bn=True, relu=True):
    """Build Atrous Spatial Pyramid Module for Deeplab.
    """
    super(ASPP, self).__init__()

    def create_convs(dilation):
      convs = []
      if dilation > 1:
        convs.append(nn.Conv2d(in_channels, out_channels,
                               3, 1,
                               padding=dilation,
                               dilation=dilation,
                               bias=not bn))
      else:
        convs.append(nn.Conv2d(in_channels, out_channels,
                               1, 1, 0, 1, bias=not bn))
      if bn:
        convs.append(nn.BatchNorm2d(out_channels))
      if relu:
        convs.append(nn.ReLU(inplace=True))
      return nn.Sequential(*convs)

    self.aspp_1 = create_convs(6)
    self.aspp_2 = create_convs(12)
    self.aspp_3 = create_convs(18)
    self.aspp_4 = create_convs(24)

  def forward(self, x):
    xs = [self.aspp_1(x), self.aspp_2(x),
          self.aspp_3(x), self.aspp_4(x)]
    #output = torch.cat(xs, dim=1)
    output = sum(xs)
    return output


class PSPP(nn.Module):

  def __init__(self, in_channels, out_channels,
               bn=True, relu=True):
    """Build Pooling Spatial Pyramid Module for PSPNet.
    """
    super(PSPP, self).__init__()

    def create_convs(in_c, out_c, k, size):
      convs = []
      if size:
        convs.append(nn.AdaptiveAvgPool2d(size))
      p = (k - 1) // 2
      convs.append(nn.Conv2d(in_c, out_c, k, 1, p, 1, bias=not bn))
      if bn:
        convs.append(nn.BatchNorm2d(out_c))
      if relu:
        convs.append(nn.ReLU(inplace=True))
      return nn.Sequential(*convs)

    self.pspp_1 = create_convs(in_channels, out_channels, 1, 1)
    self.pspp_2 = create_convs(in_channels, out_channels, 1, 2)
    self.pspp_3 = create_convs(in_channels, out_channels, 1, 3)
    self.pspp_4 = create_convs(in_channels, out_channels, 1, 6)
    self.conv = create_convs(
        in_channels + out_channels * 4, out_channels, 3, None)

  def forward(self, x):
    size = x.shape[-2:]
    x1 = F.interpolate(
        self.pspp_1(x), size=size, mode='bilinear')
    x2 = F.interpolate(
        self.pspp_2(x), size=size, mode='bilinear')
    x3 = F.interpolate(
        self.pspp_3(x), size=size, mode='bilinear')
    x4 = F.interpolate(
        self.pspp_4(x), size=size, mode='bilinear')
    output = torch.cat([x, x1, x2, x3, x4], dim=1)
    output = self.conv(output)

    return output
