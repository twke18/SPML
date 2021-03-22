"""Define utility functions for visualization.
"""

import os

import torch
import torchvision.utils
import torch.nn.functional as F
import scipy.io
import numpy as np

import spml.utils.general.common as common_utils


def write_image_to_tensorboard(writer, images, size, curr_iter, name='image'):
  """Write list of image tensors to tensorboard.

  Args:
    writer: An instance of tensorboardX.SummaryWriter
    images: A list of 4-D tensors of shape
      `[batch_size, channel, height, width]`.
  """
  for ind, image in enumerate(images):
    if image.shape[-2] != size[0] or image.shape[-1] != size[1]:
      image_type = image.dtype
      image = F.interpolate(image.float(), size=size, mode='nearest')
      images[ind] = image.type(image_type)

  images = torch.cat(images, dim=3)
  images = torchvision.utils.make_grid(images, nrow=1)
  writer.add_image(name, images, curr_iter)


def write_scalars_to_tensorboard(writer, scalars, curr_iter):
  """Write dict of scalars to tensorboard.
  """
  for key, value in scalars.items():
    writer.add_scalar(key, value, curr_iter)


def convert_label_to_color(label, color_map):
  """Convert integer label to RGB image.
  """
  n, h, w = label.shape
  rgb = torch.index_select(color_map, 0, label.view(-1)).view(n, h, w, 3)
  rgb = rgb.permute(0, 3, 1, 2)

  return rgb


def load_color_map(color_map_path):
  """Load color map.
  """
  color_map = scipy.io.loadmat(color_map_path)
  color_map = color_map[
    os.path.basename(color_map_path).strip('.mat')]
  color_map = torch.from_numpy((color_map * 255).astype(np.uint8))

  return color_map


def embedding_to_rgb(embeddings, project_type='pca'):
  """Project high-dimension embeddings to RGB colors.

  Args:
    embeddings: A 4-D float tensor with shape
      `[batch_size, embedding_dim, height, width]`.
    project_type: pca | random.

  Returns:
    An N-D float tensor with shape `[batch_size, 3, height, width]`.
  """
  # Transform NCHW to NHWC.
  embeddings = embeddings.permute(0, 2, 3, 1).contiguous()
  embeddings = common_utils.normalize_embedding(embeddings)

  N, H, W, C= embeddings.shape
  if project_type == 'pca':
    rgb = common_utils.pca(embeddings, 3)
  elif project_type == 'random':
    random_inds = torch.randint(0,
                                C,
                                (3,),
                                dtype=tf.long,
                                device=embeddings.device)
    rgb = torch.index_select(embeddings, -1, random_inds)
  else:
    raise NotImplementedError()

  # Normalize per image.
  rgb = rgb.view(N, -1, 3)
  rgb -= torch.min(rgb, 1, keepdim=True)[0]
  rgb /= torch.max(rgb, 1, keepdim=True)[0]
  rgb *= 255
  rgb = rgb.byte()

  # Transform NHWC to NCHW.
  rgb = rgb.view(N, H, W, 3)
  rgb = rgb.permute(0, 3, 1, 2).contiguous()

  return rgb
