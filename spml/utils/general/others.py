"""Utility functions.
"""

import numpy as np
import torch

import spml.data.transforms as transforms


def create_image_pyramid(image_batch, label_batch, scales, is_flip=True):
  """Create pyramid of images and labels in different scales.

  This function generates image and label pyramid by upscaling
  and downscaling the input image and label.

  Args:
    image_batch: A dict with entry `image`, which is a 3-D numpy
      float tensor of shape `[height, width, channels]`.
    label_batch: A dict with entry `semantic_label` and `instance_label`,
      which are 2-D numpy long tensor of shape `[height, width]`.
    scales: A list of floats indicate the scale ratios.
    is_flip: enable/disable flip to augment image & label pyramids
      by horizontally flipping.

  Return:
    A list of tuples of (image, label, {'is_flip': True/False}).
  """
  h, w = image_batch['image'].shape[-2:]
  flips = [True, False] if is_flip else [False]
  batches = []
  for scale in scales:
    for flip in flips:
      img = image_batch['image'].transpose(1, 2, 0)
      sem_lab = label_batch['semantic_label']
      inst_lab = label_batch['instance_label']
      lab = np.stack([sem_lab, inst_lab], axis=2)
      img, lab = transforms.resize(img, lab, scale)
      if flip:
        img = img[:, ::-1, :]
        lab = lab[:, ::-1, :]
      img = img.transpose(2, 0, 1)
      img_batch = {'image': img}
      lab_batch = {'semantic_label': lab[..., 0],
                   'instance_label': lab[..., 1]}
      data_info = {'is_flip': flip}
      batches.append((img_batch, lab_batch, data_info))
  return batches


def prepare_datas_and_labels_mgpu(data_iterator, gpu_ids):
  """Prepare datas and labels for multi-gpu computation.

  Args:
    data_iterator: An Iterator instance of pytorch.DataLoader, which
      return a dictionary of `datas`, `labels`, and a scalar of `index`.
    gpu_ids: A list of scalars indicates the GPU device ids.

  Return:
    A list of tuples of `datas` and `labels`.
  """
  input_batch, label_batch = [], []
  for gpu_id in gpu_ids:
    data, label, index = data_iterator.next()
    for k, v in data.items():
      data[k] = (v if not torch.is_tensor(v)
                  else v.pin_memory().to(gpu_id, non_blocking=True))
    for k, v in label.items():
      label[k] = (v if not torch.is_tensor(v)
                   else v.pin_memory().to(gpu_id, non_blocking=True))
    input_batch.append(data)
    label_batch.append(label)

  return input_batch, label_batch
