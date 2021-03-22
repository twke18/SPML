# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
import operator
import warnings
from itertools import chain

import torch
from torch.nn.parallel.data_parallel import DataParallel as DP
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply


def _check_balance(device_ids):

  imbalance_warn = """
  There is an imbalance between your GPUs. You may want to exclude GPU {} which
  has less than 75% of the memory or cores of GPU {}. You can do so by setting
  the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
  environment variable."""

  dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

  def warn_imbalance(get_prop):

      values = [get_prop(props) for props in dev_props]
      min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
      max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
      if min_val / max_val < 0.75:
        warnings.warn(imbalance_warn.format(device_ids[min_pos],
                                            device_ids[max_pos]))
        return True

      return False

  if warn_imbalance(lambda props: props.total_memory):
    return
  if warn_imbalance(lambda props: props.multi_processor_count):
    return


class DataParallel(DP):
  r"""Reimplementation of torch.nn.DataParallel, and allows not
  gathering outputs at each gpu.
  """
  # TODO: update notes/cuda.rst when this class handles 8+ GPUs well
  def __init__(self, module, device_ids=None,
               output_device=None, dim=0, gather_output=True):

    super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    if not torch.cuda.is_available():
      self.module = module
      self.device_ids = []
      return

    if device_ids is None:
      device_ids = list(range(torch.cuda.device_count()))
    if output_device is None:
      output_device = device_ids[0]

    self.dim = dim
    self.module = module
    self.device_ids = device_ids
    self.output_device = output_device
    self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))
    self.gather_output = gather_output

    _check_balance(self.device_ids)

    if len(self.device_ids) == 1:
      self.module.cuda(device_ids[0])

  def forward(self, *inputs, **kwargs):

    # If no device ids specified, fall back to single gpu.
    if not self.device_ids:
      return self.module(*inputs, **kwargs)

    for t in chain(self.module.parameters(), self.module.buffers()):
      if t.device != self.src_device_obj:
        raise RuntimeError(
            "module must have its parameters and buffers "
            "on device {} (device_ids[0]) but found one of "
            "them on device: {}".format(self.src_device_obj, t.device))

    # inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
    assert kwargs == {}, 'not implemented'
    kwargs = [{} for _ in range(len(inputs))]

    #if len(self.device_ids) == 1:
    #  return self.module(*inputs[0], **kwargs[0])
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, kwargs)
    if self.gather_output:
      return self.gather(outputs, self.output_device)
    else:
      return outputs
