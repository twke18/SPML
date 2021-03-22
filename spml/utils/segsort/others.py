"""Utility functions.
"""

import os
import glob

import numpy as np
import torch


def load_memory_banks(memory_dir):
  """Return prototypes and labels save in the directory.

  Args:
    memory_dir: A string indicates the directory where 
      prototypes are stored. 
      The dir layout should look like:
      memory_dir --- prototype_1.npy
                 |-- protoytpe_2.npy

  Returns:
    A 2-D float tensor of shape `[num_prototypes, num_channels]`;
    A 1-D long tensor of shape `[num_prototypes]`.
  """
  memory_paths = sorted(glob.glob(os.path.join(
      memory_dir, '*.npy')))
  assert(len(memory_paths) > 0), 'No memory stored in the directory'

  prototypes, prototype_labels = [], []
  for memory_path in memory_paths:
    datas = np.load(memory_path, allow_pickle=True).item()
    prototypes.append(datas['prototype'])
    prototype_labels.append(datas['prototype_label'])

  prototypes = np.concatenate(prototypes, 0)
  prototype_labels = np.concatenate(prototype_labels, 0)

  prototypes = torch.FloatTensor(prototypes)
  prototype_labels = torch.LongTensor(prototype_labels)

  return prototypes, prototype_labels

