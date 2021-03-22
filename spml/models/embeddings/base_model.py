"""Build interface for segmentation models."""
# This code is borrowed and modified from:
# https://github.com/uber-research/UPSNet/blob/master/upsnet/models/resnet.py

import warnings

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ResnetBase(nn.Module):

  def name_mapping(self, name, resume=False):
    if resume:
        return name if not name.startswith('module.') else name[len('module.'):]

    if name.startswith('conv1') or name.startswith('bn1'):
        return 'resnet_backbone.conv1.' + name

    return name.replace('layer1', 'resnet_backbone.res2.layers')\
               .replace('layer2', 'resnet_backbone.res3.layers')\
               .replace('layer3', 'resnet_backbone.res4.layers')\
               .replace('layer4', 'resnet_backbone.res5.layers')

  def load_state_dict(self, state_dict, resume=False):

    own_state = self.state_dict()

    for name, param in state_dict.items():
      name = self.name_mapping(name, resume)

      if name not in own_state:
        warnings.warn('unexpected key "{}" in state_dict'.format(name))
        continue

      if isinstance(param, Parameter):
        # backwards compatibility for serialized parameters
        param = param.data

      if own_state[name].shape == param.shape:
        own_state[name].copy_(param)
      else:
        warnings.warn(
            'While copying the parameter named {}, whose dimensions in the'
            'models are {} and whose dimensions in the checkpoint are {}, '
            '...'.format(name, own_state[name].size(), param.size()))

    missing = (set(own_state.keys())
                - set([self.name_mapping(_, resume) for _ in state_dict.keys()]))
    if len(missing) > 0:
        warnings.warn('missing keys in state_dict: "{}"'.format(missing))

  def get_params_lr(self):

    raise NotImplementedError()
