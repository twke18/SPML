"""Inference script for semantic segmentation by softmax classifier.
"""
from __future__ import print_function, division
import os
import math

import PIL.Image as Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import spml.data.transforms as transforms
import spml.utils.general.vis as vis_utils
import spml.utils.general.others as other_utils
from spml.data.datasets.base_dataset import ListDataset
from spml.config.default import config
from spml.config.parse_args import parse_args
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab
from spml.models.predictions.softmax_classifier import softmax_classifier
from spml.models.crf import DenseCRF

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
  os.makedirs(semantic_dir, exist_ok=True)
  os.makedirs(semantic_rgb_dir, exist_ok=True)

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

  # Define CRF.
  postprocessor = DenseCRF(
      iter_max=args.crf_iter_max,
      pos_xy_std=args.crf_pos_xy_std,
      pos_w=args.crf_pos_w,
      bi_xy_std=args.crf_bi_xy_std,
      bi_rgb_std=args.crf_bi_rgb_std,
      bi_w=args.crf_bi_w,)


  # Start inferencing.
  with torch.no_grad():
    for data_index in tqdm(range(len(test_dataset))):
      # Image path.
      image_path = test_image_paths[data_index]
      base_name = os.path.basename(image_path).replace('.jpg', '.png')

      # Image resolution.
      original_image_batch, original_label_batch, _ = test_dataset[data_index]
      image_h, image_w = original_image_batch['image'].shape[-2:]
      batches = other_utils.create_image_pyramid(
          original_image_batch, original_label_batch,
          scales=[0.5, 0.75, 1, 1.25, 1.5],
          is_flip=True)

      semantic_logits = []
      for image_batch, label_batch, data_info in batches:
        resize_image_h, resize_image_w = image_batch['image'].shape[-2:]
        # Crop and Pad the input image.
        image_batch['image'] = transforms.resize_with_pad(
            image_batch['image'].transpose(1, 2, 0),
            config.test.crop_size,
            image_pad_value=0).transpose(2, 0, 1)
        image_batch['image'] = torch.FloatTensor(
            image_batch['image'][np.newaxis, ...]).cuda()
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
        semantic_logit = torch.FloatTensor(
            1, config.dataset.num_classes, pad_image_h, pad_image_w).zero_().to("cuda:0")
        counts = torch.FloatTensor(
            1, 1, pad_image_h, pad_image_w).zero_().to("cuda:0")
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
            semantic_logit[..., sh:eh, sw:ew] += crop_outputs['semantic_logit'].to("cuda:0")
            counts[..., sh:eh, sw:ew] += 1
        semantic_logit /= counts
        semantic_logit = semantic_logit[..., :resize_image_h, :resize_image_w]
        semantic_logit = F.interpolate(
            semantic_logit, size=(image_h, image_w), mode='bilinear')
        semantic_logit = F.softmax(semantic_logit, dim=1)
        semantic_logit = semantic_logit.data.cpu().numpy().astype(np.float32)
        if data_info['is_flip']:
          semantic_logit = semantic_logit[..., ::-1]
        semantic_logits.append(semantic_logit)

      semantic_logits = np.concatenate(semantic_logits, axis=0)
      semantic_prob = np.mean(semantic_logits, axis=0)

      # DenseCRF post-processing.
      image = original_image_batch['image'].astype(np.float32)
      image = image.transpose(1, 2, 0)
      image *= np.reshape(config.network.pixel_stds, (1, 1, 3))
      image += np.reshape(config.network.pixel_means, (1, 1, 3))
      image = image * 255
      image = image.astype(np.uint8)

      semantic_prob = postprocessor(image, semantic_prob)

      semantic_pred = np.argmax(semantic_prob, axis=0).astype(np.uint8)

      # Save semantic predictions.
      semantic_pred_name = os.path.join(semantic_dir, base_name)
      Image.fromarray(semantic_pred, mode='L').save(semantic_pred_name)

      semantic_pred_rgb = color_map[semantic_pred]
      semantic_pred_rgb_name = os.path.join(semantic_rgb_dir, base_name)
      Image.fromarray(semantic_pred_rgb, mode='RGB').save(
          semantic_pred_rgb_name)


if __name__ == '__main__':
  main()
