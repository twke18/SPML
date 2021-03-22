"""Classes for Dataset with image-level tags.
"""

import cv2
import numpy as np

from spml.data.datasets.base_dataset import ListDataset
import spml.data.transforms as transforms


class ListTagDataset(ListDataset):
  """Class of dataset which takes a file of paired list of 
  images and labels. This class returns image-level tags
  from semantic labels.
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
    """Class for Image-level Tags Dataset.

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
    super(ListTagDataset, self).__init__(
        data_dir,
        data_list,
        img_mean,
        img_std,
        size,
        random_crop,
        random_scale,
        random_mirror,
        training)

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

    if semantic_label is not None:
      cats = np.unique(semantic_label)
      semantic_tags = np.zeros((256, ), dtype=np.uint8)
      semantic_tags[cats] = 1
    else:
      semantic_tags = None

    return image, semantic_label, instance_label, semantic_tags

  def _training_preprocess(self, idx):
    """Data preprocessing for training.
    """
    assert(self.size is not None)
    image, semantic_label, instance_label, semantic_tags = self._get_datas_by_index(idx)

    label = np.stack([semantic_label, instance_label], axis=2)

    if self.random_mirror:
      is_flip = np.random.uniform(0, 1.0) >= 0.5
      if is_flip:
        image = image[:, ::-1, ...]
        label = label[:, ::-1, ...]

    if self.random_scale:
      image, label = transforms.random_resize(image, label, 0.5, 1.5)

    if self.random_crop:
      image, label = transforms.random_crop_with_pad(
          image, label, self.size, self.img_mean, 255)

    semantic_label, instance_label = label[..., 0], label[..., 1]

    return image, semantic_label, instance_label, semantic_tags

  def __getitem__(self, idx):
    """Retrive image and label by index.
    """
    if self.training:
      image, semantic_label, instance_label, semantic_tag = self._training_preprocess(idx)
    else:
      raise NotImplementedError()

    image = image - np.array(self.img_mean, dtype=image.dtype)
    image = image / np.array(self.img_std, dtype=image.dtype)

    inputs = {'image': image.transpose(2, 0, 1)}
    labels = {'semantic_label': semantic_label,
              'instance_label': instance_label,
              'semantic_tag': semantic_tag}

    return inputs, labels, idx


class ListTagClassifierDataset(ListTagDataset):

  def __init__(self,
               data_dir,
               data_list,
               img_mean=(0, 0, 0),
               img_std=(1, 1, 1),
               size=None,
               random_crop=False,
               random_scale=False,
               random_mirror=False,
               random_grayscale=False,
               random_blur=False,
               training=False):
    """Class of Image-level Tags Dataset for training softmax
    classifier, where we introduce more data augmentation.

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
        If True, adopt randomly scaling as augmentation.
      random_mirror: enable/disable random_mirror for data augmentation.
        If True, adopt randomly mirroring as augmentation.
      random_grayscale: enable/disable random_grayscale for data augmentation.
        If True, adopt randomly converting RGB to grayscale as augmentation.
      random_blur: enable/disable random_blur for data augmentation.
        If True, adopt randomly applying Gaussian blur as augmentation.
      training: enable/disable training to set dataset for training and
        testing. If True, set to training mode.
    """
    super(ListTagClassifierDataset, self).__init__(
        data_dir,
        data_list,
        img_mean,
        img_std,
        size,
        random_crop,
        random_scale,
        random_mirror,
        training)
    self.random_grayscale = random_grayscale
    self.random_blur = random_blur

  def _training_preprocess(self, idx):
    """Data preprocessing for training.
    """
    assert(self.size is not None)
    image, semantic_label, instance_label, semantic_tags = self._get_datas_by_index(idx)

    label = np.stack([semantic_label, instance_label], axis=2)

    if self.random_mirror:
      is_flip = np.random.uniform(0, 1.0) >= 0.5
      if is_flip:
        image = image[:, ::-1, ...]
        label = label[:, ::-1, ...]

    if self.random_scale:
      image, label = transforms.random_resize(image, label, 0.5, 2.0)

    if self.random_crop:
      image, label = transforms.random_crop_with_pad(
          image, label, self.size, self.img_mean, 255)

    # Randomly convert RGB to grayscale.
    if self.random_grayscale and np.random.uniform(0, 1.0) < 0.3:
      rgb2gray = np.array([0.3, 0.59, 0.11], dtype=np.float32)
      image = np.sum(
          image * np.reshape(rgb2gray, (1, 1, 3)), axis=-1, keepdims=True)
      image = np.tile(image, (1,1,3))

    # Randomly apply Gaussian blur.
    if self.random_blur and np.random.uniform(0, 1.0) < 0.5:
      sigma = np.random.uniform(0.1, 5)
      w_x, w_y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
      weight = np.exp(- (w_x ** 2 + w_y ** 2) / sigma**2)
      weight = weight / weight.sum()
      image = cv2.filter2D(image, -1, weight)


    semantic_label, instance_label = label[..., 0], label[..., 1]

    return image, semantic_label, instance_label, semantic_tags

