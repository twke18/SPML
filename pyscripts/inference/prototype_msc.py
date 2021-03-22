"""Inference script for generating memory banks.
"""
from __future__ import print_function, division
import os
import math

import PIL.Image as Image
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import spml.data.transforms as transforms
import spml.utils.general.vis as vis_utils
import spml.utils.general.common as common_utils
import spml.utils.general.others as other_utils
from spml.data.datasets.base_dataset import ListDataset
from spml.config.default import config
from spml.config.parse_args import parse_args
import spml.utils.segsort.common as segsort_common
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab

cudnn.enabled = True
cudnn.benchmark = True


def separate_comma(str_comma):
  ints = [int(i) for i in str_comma.split(',')]
  return ints


def main():
  """Inference for generating memory banks.
  """
  # Retreve experiment configurations.
  args = parse_args('Inference for generating memory banks.')
  config.network.kmeans_num_clusters = separate_comma(args.kmeans_num_clusters)
  config.network.label_divisor = args.label_divisor

  # Create directories to save results.
  prototype_dir = os.path.join(args.save_dir, 'semantic_prototype')
  os.makedirs(prototype_dir, exist_ok=True)

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

  embedding_model = embedding_model.cuda()
  embedding_model.eval()
      
  # Load trained weights.
  model_path_template = os.path.join(args.snapshot_dir, 'model-{:d}.pth')
  save_iter = config.train.max_iteration - 1
  embedding_model.load_state_dict(
      torch.load(model_path_template.format(save_iter))['embedding_model'],
      resume=True)

  # Start inferencing.
  with torch.no_grad():
    for data_index in tqdm(range(len(test_dataset))):
    #for data_index in range(3000):
      # Image path.
      image_path = test_image_paths[data_index]
      base_name = os.path.basename(image_path).replace('.jpg', '.png')

      # Image resolution.
      image_batch, label_batch, _ = test_dataset[data_index]
      image_h, image_w = image_batch['image'].shape[-2:]
      batches = other_utils.create_image_pyramid(
          image_batch, label_batch,
          scales=[0.5, 1, 1.5],
          is_flip=False)

      prototype_results = {'prototype': [], 'prototype_label': []}
      for image_batch, label_batch, data_info in batches:
        resize_image_h, resize_image_w = image_batch['image'].shape[-2:]
        # Crop and Pad the input image.
        image_batch['image'] = transforms.resize_with_pad(
            image_batch['image'].transpose(1, 2, 0),
            config.test.crop_size,
            image_pad_value=0).transpose(2, 0, 1)
        image_batch['image'] = torch.FloatTensor(
            image_batch['image'][np.newaxis, ...]).to("cuda:0")
        pad_image_h, pad_image_w = image_batch['image'].shape[-2:]

        # Create the fake labels where clustering ignores 255.
        fake_label_batch = {}
        for label_name in ['semantic_label', 'instance_label']:
          lab = np.zeros((resize_image_h, resize_image_w),
                         dtype=np.uint8)
          lab = transforms.resize_with_pad(
              lab,
              config.test.crop_size,
              image_pad_value=config.dataset.semantic_ignore_index)

          fake_label_batch[label_name] = torch.LongTensor(
              lab[np.newaxis, ...]).to("cuda:0")

        # Put label batch to gpu 1.
        for k, v in label_batch.items():
          label_batch[k] = torch.LongTensor(v[np.newaxis, ...]).to("cuda:0")

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
        embeddings = {}
        counts = torch.FloatTensor(
            1, 1, pad_image_h, pad_image_w).zero_().to("cuda:0")
        for ind_h in patch_ind_h:
          for ind_w in patch_ind_w:
            sh, eh = ind_h - crop_h, ind_h
            sw, ew = ind_w - crop_w, ind_w
            crop_image_batch = {
              k: v[:, :, sh:eh, sw:ew] for k, v in image_batch.items()}

            # Feed-forward.
            crop_embeddings = embedding_model.generate_embeddings(
                crop_image_batch, resize_as_input=True)

            # Initialize embedding.
            for name in crop_embeddings:
              if crop_embeddings[name] is None:
                continue
              crop_emb = crop_embeddings[name].to("cuda:0")
              if name in ['embedding']:
                crop_emb = common_utils.normalize_embedding(
                    crop_emb.permute(0, 2, 3, 1).contiguous())
                crop_emb = crop_emb.permute(0, 3, 1, 2)
              else:
                continue

              if name not in embeddings.keys():
                embeddings[name] = torch.FloatTensor(
                    1,
                    crop_emb.shape[1],
                    pad_image_h,
                    pad_image_w).zero_().to("cuda:0")
              embeddings[name][:, :, sh:eh, sw:ew] += crop_emb
            counts[:, :, sh:eh, sw:ew] += 1

        for k in embeddings.keys():
          embeddings[k] /= counts

        # KMeans.
        lab_div = config.network.label_divisor
        fake_sem_lab = fake_label_batch['semantic_label']
        fake_inst_lab = fake_label_batch['instance_label']
        clustering_outputs = embedding_model.generate_clusters(
            embeddings.get('embedding', None),
            fake_sem_lab,
            fake_inst_lab)
        embeddings.update(clustering_outputs)

        # Save semantic prototypes.
        prototypes = segsort_common.calculate_prototypes_from_labels(
            embeddings['cluster_embedding'],
            embeddings['cluster_index'])
        _, prototype_labels = (
          segsort_common.find_majority_label_index(
              label_batch['semantic_label'],
              embeddings['cluster_index']))

        prototypes = prototypes.cpu().data.numpy()
        prototype_labels = prototype_labels.cpu().data.numpy()
        prototype_results['prototype'].append(prototypes)
        prototype_results['prototype_label'].append(prototype_labels)

      # Save semantic prototypes.
      prototype_name = os.path.join(
          prototype_dir,
          base_name.replace('.png', '.npy'))

      for k, v in prototype_results.items():
        v = np.concatenate(v, axis=0)
        prototype_results[k] = v
      np.save(prototype_name, prototype_results)



if __name__ == '__main__':
  main()
