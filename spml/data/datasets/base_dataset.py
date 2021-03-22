"""Base classes for Dataset.
"""

import os

import torch
import torch.utils.data
import numpy as np
import PIL.Image as Image
import cv2

import spml.data.transforms as transforms


class ListDataset(torch.utils.data.Dataset):
  """Base class of dataset which takes a file of paired list of
  images, semantic labels and instance labels.
  """

  def __init__(self,
               data_dir,
               data_list,
               img_mean=(0, 0, 0),
               img_std=(1, 1, 1),
               size=None,
               random_crop=False,
               random_scale=False,
               random_mirror=False,
               training=False):
    """Base class for Dataset.

    Args:
      data_dir: A string indicates root directory of images and labels.
      data_list: A list of strings which indicate path of paired images
        and labels. 'image_path semantic_label_path instance_label_path'.
      img_mean: A list of scalars indicate the mean image value per channel.
      img_std: A list of scalars indicate the std image value per channel.
      size: A tuple of scalars indicate size of output image and labels.
        The output resolution remain the same if `size` is None.
      random_crop: enable/disable random_crop for data augmentation.
        If True, adopt randomly cropping as augmentation.
      random_scale: enable/disable random_scale for data augmentation.
        If True, adopt adopt randomly scaling as augmentation.
      random_mirror: enable/disable random_mirror for data augmentation.
        If True, adopt adopt randomly mirroring as augmentation.
      training: enable/disable training to set dataset for training and
        testing. If True, set to training mode.
    """
    self.image_paths, self.semantic_label_paths, self.instance_label_paths = (
      self._read_image_and_label_paths(data_dir, data_list))

    self.training = training
    self.img_mean = img_mean
    self.img_std = img_std
    self.size = size
    self.random_crop = random_crop
    self.random_scale = random_scale
    self.random_mirror = random_mirror

  def eval(self):
    """Set the dataset to evaluation mode.
    """
    self.training = False

  def train(self):
    """Set the dataset to training mode.
    """
    self.training = True

  def _read_image_and_label_paths(self, data_dir, data_list):
    """Parse strings into lists of image, semantic label and
    instance label paths.

    Args:
      data_dir: A string indicates root directory of images and labels.
      data_list: A list of strings which indicate path of paired images
        and labels. 'image_path semantic_label_path instance_label_path'.

    Return:
      Threee lists of file paths.
    """
    images, semantic_labels, instance_labels = [], [], []
    with open(data_list, 'r') as list_file:
      for line in list_file:
        line = line.strip('\n')
        try:
          img, semantic_lab, instance_lab = line.split(' ')
        except:
          img = line
          semantic_lab = instance_lab = None

        images.append(os.path.join(data_dir, img))

        if semantic_lab is not None:
          semantic_labels.append(os.path.join(data_dir, semantic_lab))

        if instance_lab is not None:
          instance_labels.append(os.path.join(data_dir, instance_lab))

    return images, semantic_labels, instance_labels

  def _read_image(self, image_path):
    """Read BGR uint8 image.
    """
    img = np.array(Image.open(image_path).convert(mode='RGB'))
    img = img.astype(np.float32) / 255
    return img

  def _read_label(self, label_path):
    """Read uint8 label.
    """
    return np.array(Image.open(label_path).convert(mode='L'))

  def _get_datas_by_index(self, idx):
    """Return image_path, semantic_label_path, instance_label_path
    by the given index.
    """
    image_path = self.image_paths[idx]
    image = self._read_image(image_path)

    if len(self.semantic_label_paths) > 0:
      semantic_label_path = self.semantic_label_paths[idx]
      semantic_label = self._read_label(semantic_label_path)
    else:
      semantic_label = None

    if len(self.instance_label_paths) > 0:
      instance_label_path = self.instance_label_paths[idx]
      instance_label = self._read_label(instance_label_path)
    else:
      instance_label = None

    return image, semantic_label, instance_label

  def _training_preprocess(self, idx):
    """Data preprocessing for training.
    """
    assert(self.size is not None)
    image, semantic_label, instance_label = self._get_datas_by_index(idx)

    label = np.stack([semantic_label, instance_label], axis=2)

    if self.random_mirror:
      image, label = transforms.random_mirror(image, label)

    if self.random_scale:
      image, label = transforms.random_resize(image, label, 0.5, 1.5)

    if self.random_crop:
      image, label = transforms.random_crop_with_pad(
          image, label, self.size, self.img_mean, 255)

    semantic_label, instance_label = label[..., 0], label[..., 1]

    return image, semantic_label, instance_label

  def _eval_preprocess(self, idx):
    """Data preprocessing for evaluationg.
    """
    image, semantic_label, instance_label = self._get_datas_by_index(idx)

    if self.size is not None:
      image = transforms.resize_with_pad(
          image, self.size, self.img_mean)

      image = image[:self.size[0], :self.size[1], ...]

    return image, semantic_label, instance_label

  def __len__(self):
    """Total number of datas in the dataset.
    """
    return len(self.image_paths)

  def __getitem__(self, idx):
    """Retrive image and label by index.
    """
    if self.training:
      image, semantic_label, instance_label = self._training_preprocess(idx)
    else:
      image, semantic_label, instance_label = self._eval_preprocess(idx)

    image = image - np.array(self.img_mean, dtype=image.dtype)
    image = image / np.array(self.img_std, dtype=image.dtype)

    inputs = {'image': image.transpose(2, 0, 1)}
    labels = {'semantic_label': semantic_label,
              'instance_label': instance_label}

    return inputs, labels, idx

  def _collate_fn_dict_list(self, dict_list):
    """Helper function to collate a list of dictionaries.
    """
    outputs = {}
    for key in dict_list[0].keys():
      values = [d[key] for d in dict_list]
      if values[0] is None:
        values = None
      elif (values[0].dtype == np.uint8
           or values[0].dtype == np.int32
           or values[0].dtype == np.int64):
        values = torch.LongTensor(values)
      elif (values[0].dtype == np.float32
             or values[0].dtype == np.float64):
        values = torch.FloatTensor(values)
      else:
        raise ValueError('Unsupported data type')

      outputs[key] = values

    return outputs

  def collate_fn(self, batch):
    """Customized collate function to group datas into batch.
    """
    images, labels, indices = zip(*batch)

    images = self._collate_fn_dict_list(images)
    labels = self._collate_fn_dict_list(labels)
    indices = torch.LongTensor(indices)

    return images, labels, indices
