"""Inference script for semantic segmentation by softmax classifier.
"""
from __future__ import print_function, division
import os
import math

import PIL.Image as Image
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

import spml.data.transforms as transforms
import spml.utils.general.vis as vis_utils
from spml.data.datasets.base_dataset import ListDataset
from spml.config.default import config
from spml.config.parse_args import parse_args
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab
from spml.models.predictions.softmax_classifier import softmax_classifier

cudnn.enabled = True
cudnn.benchmark = True


def main():
  """Inference for semantic segmentation.
  """
  # Retreve experiment configurations.
  args = parse_args('Inference for semantic segmentation.')

  # Create directories to save results.
  semantic_dir = os.path.join(args.save_dir, 'semantic_gray')
  semantic_rgb_dir = os.path.join(args.save_dir, 'semantic_color')
  if not os.path.isdir(semantic_dir):
    os.makedirs(semantic_dir)
  if not os.path.isdir(semantic_rgb_dir):
    os.makedirs(semantic_rgb_dir)

  # Create color map.
  color_map = vis_utils.load_color_map(config.dataset.color_map_path)
  color_map = color_map.numpy()

  # Create data loaders.
  test_dataset = ListDataset(
      data_dir=args.data_dir,
      data_list=args.data_list,
      img_mean=config.network.pixel_means,
      img_std=config.network.pixel_stds,
      size=None,
      random_crop=False,
      random_scale=False,
      random_mirror=False,
      training=False)
  test_image_paths = test_dataset.image_paths

  # Create models.
  if config.network.backbone_types == 'panoptic_pspnet_101':
    embedding_model = resnet_101_pspnet(config).cuda()
  elif config.network.backbone_types == 'panoptic_deeplab_101':
    embedding_model = resnet_101_deeplab(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.backbone_types)

  prediction_model = softmax_classifier(config).cuda()
  embedding_model.eval()
  prediction_model.eval()
      
  # Load trained weights.
  model_path_template = os.path.join(args.snapshot_dir, 'model-{:d}.pth')
  save_iter = config.train.max_iteration - 1
  embedding_model.load_state_dict(
      torch.load(model_path_template.format(save_iter))['embedding_model'],
      resume=True)
  prediction_model.load_state_dict(
      torch.load(model_path_template.format(save_iter))['prediction_model'])


  # Start inferencing.
  for data_index in range(len(test_dataset)):
    # Image path.
    image_path = test_image_paths[data_index]
    base_name = os.path.basename(image_path).replace('.jpg', '.png')

    # Image resolution.
    image_batch, _, _ = test_dataset[data_index]
    image_h, image_w = image_batch['image'].shape[-2:]

    # Resize the input image.
    if config.test.image_size > 0:
      image_batch['image'] = transforms.resize_with_interpolation(
          image_batch['image'].transpose(1, 2, 0),
          config.test.image_size,
          method='bilinear').transpose(2, 0, 1)
    resize_image_h, resize_image_w = image_batch['image'].shape[-2:]

    # Crop and Pad the input image.
    image_batch['image'] = transforms.resize_with_pad(
        image_batch['image'].transpose(1, 2, 0),
        config.test.crop_size,
        image_pad_value=0).transpose(2, 0, 1)
    image_batch['image'] = torch.FloatTensor(image_batch['image'][np.newaxis, ...]).cuda()
    pad_image_h, pad_image_w = image_batch['image'].shape[-2:]

    # Create the ending index of each patch.
    stride_h, stride_w = config.test.stride
    crop_h, crop_w = config.test.crop_size
    npatches_h = math.ceil(1.0 * (pad_image_h-crop_h) / stride_h) + 1
    npatches_w = math.ceil(1.0 * (pad_image_w-crop_w) / stride_w) + 1
    patch_ind_h = np.linspace(
        crop_h, pad_image_h, npatches_h, dtype=np.int32)
    patch_ind_w = np.linspace(
        crop_w, pad_image_w, npatches_w, dtype=np.int32)

    # Create place holder for full-resolution embeddings.
    outputs = {}
    with torch.no_grad():
      for ind_h in patch_ind_h:
        for ind_w in patch_ind_w:
          sh, eh = ind_h - crop_h, ind_h
          sw, ew = ind_w - crop_w, ind_w
          crop_image_batch = {
            k: v[:, :, sh:eh, sw:ew] for k, v in image_batch.items()}

          # Feed-forward.
          crop_embeddings = embedding_model(
              crop_image_batch, resize_as_input=True)
          crop_outputs = prediction_model(crop_embeddings)

          for name, crop_out in crop_outputs.items():

            if crop_out is not None:
              if name not in outputs.keys():
                output_shape = list(crop_out.shape)
                output_shape[-2:] = pad_image_h, pad_image_w
                outputs[name] = torch.zeros(output_shape, dtype=crop_out.dtype).cuda()
              outputs[name][..., sh:eh, sw:ew] += crop_out

    # Save semantic predictions.
    semantic_logits = outputs.get('semantic_logit', None)
    if semantic_logits is not None:
      semantic_pred = torch.argmax(semantic_logits, 1)
      semantic_pred = (semantic_pred.view(pad_image_h, pad_image_w)
                                    .cpu()
                                    .data
                                    .numpy()
                                    .astype(np.uint8))
      semantic_pred = semantic_pred[:resize_image_h, :resize_image_w]
      semantic_pred = cv2.resize(
          semantic_pred,
          (image_w, image_h),
          interpolation=cv2.INTER_NEAREST)

      semantic_pred_name = os.path.join(semantic_dir, base_name)
      Image.fromarray(semantic_pred, mode='L').save(semantic_pred_name)

      semantic_pred_rgb = color_map[semantic_pred]
      semantic_pred_rgb_name = os.path.join(semantic_rgb_dir, base_name)
      Image.fromarray(semantic_pred_rgb, mode='RGB').save(
          semantic_pred_rgb_name)

      # Clean GPU memory cache to save more space.
      outputs = {}
      crop_embeddings = {}
      crop_outputs = {}
      torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
