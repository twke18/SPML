"""Utility functions to process images.
"""

import numpy as np
import cv2


def resize(image,
           label,
           ratio,
           image_method='bilinear',
           label_method='nearest'):
  """Rescale image and label to the same size by the specified ratio.
  The aspect ratio is remained the same after rescaling.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.
    ratio: A float/integer indicates the scaling ratio.
    image_method: Image resizing method. bilinear/nearest.
    label_method: Image resizing method. bilinear/nearest.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """
  h, w = image.shape[:2]
  new_h, new_w = int(ratio * h), int(ratio * w)

  inter_image = (cv2.INTER_LINEAR if image_method == 'bilinear'
                  else cv2.INTER_NEAREST)
  new_image = cv2.resize(image, (new_w, new_h), interpolation=inter_image)

  inter_label = (cv2.INTER_LINEAR if label_method == 'bilinear'
                  else cv2.INTER_NEAREST)
  new_label = cv2.resize(label, (new_w, new_h), interpolation=inter_label)

  return new_image, new_label


def random_resize(image,
                  label,
                  scale_min=1.0,
                  scale_max=1.0,
                  image_method='bilinear',
                  label_method='nearest'):
  """Randomly rescale image and label to the same size. The
  aspect ratio is remained the same after rescaling.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.
    scale_min: A float indicates the minimum scaling ratio.
    scale_max: A float indicates the maximum scaling ratio.
    image_method: Image resizing method. bilinear/nearest.
    label_method: Image resizing method. bilinear/nearest.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """
  assert(scale_max >= scale_min)
  ratio = np.random.uniform(scale_min, scale_max)
  return resize(image, label, ratio, image_method, label_method)


def mirror(image, label):
  """Horizontally flipp image and label.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """

  image = image[:, ::-1, ...]
  label = label[:, ::-1, ...]
  return image, label


def random_mirror(image, label):
  """Randomly horizontally flipp image and label.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """
  is_flip = np.random.uniform(0, 1.0) >= 0.5
  if is_flip:
    image, label = mirror(image, label)

  return image, label


def resize_with_interpolation(image, larger_size, method='bilinear'):
  """Rescale image with larger size as `larger_size`. The aspect
  ratio is remained the same after rescaling.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    larger_size: An interger indicates the target size of larger side.
    method: Image resizing method. bilinear/nearest.

  Return:
    A tensor of shape `[new_height, new_width, channels]`.
  """
  h, w = image.shape[:2]
  new_size = float(larger_size)
  ratio = np.minimum(new_size / h, new_size / w)
  new_h, new_w = int(ratio * h), int(ratio * w)

  inter = (cv2.INTER_LINEAR if method == 'bilinear'
                  else cv2.INTER_NEAREST)
  new_image = cv2.resize(image, (new_w, new_h), interpolation=inter)

  return new_image


def resize_with_pad(image, size, image_pad_value=0, pad_mode='left_top'):
  """Upscale image by pad to the width and height.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    size: A tuple of integers indicates the target size.
    image_pad_value: An integer indicates the padding value.
    pad_mode: Padding mode. left_top/center.

  Return:
    A tensor of shape `[new_height, new_width, channels]`.
  """
  h, w = image.shape[:2]
  new_shape = list(image.shape)
  new_shape[0] = h if h > size[0] else size[0]
  new_shape[1] = w if w > size[1] else size[1]
  pad_image = np.zeros(new_shape, dtype=image.dtype)

  if isinstance(image_pad_value, int) or isinstance(image_pad_value, float):
    pad_image.fill(image_pad_value)
  else:
    for ind_ch, val in enumerate(image_pad_value):
      pad_image[:, :, ind_ch].fill(val)

  if pad_mode == 'center':
    s_y = (new_shape[0] - h) // 2
    s_x = (new_shape[1] - w) // 2
    pad_image[s_y:s_y+h, s_x:s_x+w, ...] = image
  elif pad_mode == 'left_top':
    pad_image[:h, :w, ...] = image
  else:
    raise ValueError('Unsupported padding mode')

  return pad_image


def random_crop_with_pad(image,
                         label,
                         crop_size,
                         image_pad_value=0,
                         label_pad_value=255,
                         pad_mode='left_top',
                         return_bbox=False):
  """Randomly crop image and label, and pad them before cropping
  if the size is smaller than `crop_size`.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.
    crop_size: A tuple of integers indicates the cropped size.
    image_pad_value: An integer indicates the padding value.
    label_pad_value: An integer indicates the padding value.
    pad_mode: Padding mode. left_top/center.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """
  image = resize_with_pad(image, crop_size,
                          image_pad_value, pad_mode)
  label = resize_with_pad(label, crop_size,
                          label_pad_value, pad_mode)

  h, w = image.shape[:2]
  start_h = int(np.floor(np.random.uniform(0, h - crop_size[0])))
  start_w = int(np.floor(np.random.uniform(0, w - crop_size[1])))
  end_h = start_h + crop_size[0]
  end_w = start_w + crop_size[1]

  crop_image = image[start_h:end_h, start_w:end_w, ...]
  crop_label = label[start_h:end_h, start_w:end_w, ...]

  if return_bbox:
    bbox = [start_w, start_h, end_w, end_h]
    return crop_image, crop_label, bbox
  else:
    return crop_image, crop_label
