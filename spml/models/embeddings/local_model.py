"""Create model for producing pixel location and smoothed RGB features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

import spml.utils.general.common as common_utils
import spml.utils.segsort.common as segsort_common


class GaussianConv2d(nn.Module):

  def __init__(self, in_channels, out_channels, ksize=5):
    """Applies 2-D Gaussian Blur.

    Args:
      in_channels: An integer indicates input channel dimension.
      out_channels: An integer indicates output channel dimension.
      ksize: An integer indicates Gaussian kernel size.
    """

    super(GaussianConv2d, self).__init__()
    weight = (np.arange(ksize, dtype=np.float32) - ksize // 2) ** 2
    weight = np.sqrt(weight[None, :] + weight[:, None])
    weight = np.reshape(weight, (1, 1, ksize, ksize)) / weight.sum()
    self.weight = Parameter(
        torch.Tensor(weight).expand(out_channels, -1, -1, -1))
    self._in_channels = in_channels
    self._out_channels = out_channels

  def forward(self, x):
    with torch.no_grad():
      return F.conv2d(x, self.weight, groups=self._in_channels)


class LocationColorNetwork(nn.Module):

  def __init__(self, use_color=True, use_location=True,
               norm_color=True, smooth_ksize=None):
    """Generates location coordinates and blurred RGB colors.

    Args:
      use_color: enable/disable use_color to output RGB colors.
      use_location: enable/disable use_location to output location
        coordinates.
      norm_color: enable/disable norm_color to normalize RGB colors.
        If True, scale the maximum and minimum value to 1 and -1.
      smooth_ksize: enable/disable smooth_ksize to smooth the RGB
        colors. If True, the Gaussian kernel is set to 3.
    """

    super(LocationColorNetwork, self).__init__()
    self._use_color = use_color
    self._use_location = use_location
    self._norm_color = norm_color
    self._smooth_ksize = smooth_ksize
    if smooth_ksize:
      self.smooth_kernel = GaussianConv2d(3, 3, smooth_ksize)
    else:
      self.smooth_kernel = nn.Identity()

  def __repr__(self):
    return ('LocationColorNetwork(use_color={}, use_location={}'+
            ', smooth_ksize={})').format(
                self._use_color, self._use_location, self._smooth_ksize)

  def forward(self, x, size=None):
    """Genearet location coordinates and color features.

    Args:
      x: A 4-D tensor of shape `[batch_size, channels, height, width]`.
      size: A tuple of integers indicates the output resolution.

    Returns:
      A N-D tensor of shape `[batch_size, out_height, out_width, channels]`.
      For the output channels, the first 2 are the locations and the rest
      are the RGB colors.
    """
    N, C, H, W = x.shape
    if size:
      H, W = size

    features = []

    # Generate location features.
    if self._use_location:
      locations = segsort_common.generate_location_features(
          (H, W), x.device, 'float')
      locations -= 0.5
      locations = locations.unsqueeze(0).expand(N, H, W, 2)
      features.append(locations)

    # Generate color features.
    if self._use_color:
      x = self.smooth_kernel(x)

      if size:
        x = F.interpolate(x, size=size, mode='bilinear')

      colors = x.permute(0, 2, 3, 1).contiguous()
      # Normalize color per data.
      if self._norm_color:
        mean_colors = torch.mean(colors.view(N, -1, C),
                                 dim=1, keepdim=True)
        mean_colors = mean_colors.view(N, 1, 1, C)
        colors = colors - mean_colors

        max_colors, _ = torch.max(
            torch.abs(colors.view(N, -1, C)),
            dim=1, keepdim=True)
        max_colors = max_colors.view(N, 1, 1, C)
        colors = colors / max_colors

      features.append(colors)

    features = torch.cat(features, dim=-1)
    return features
