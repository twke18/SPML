"""Genereate pseudo labels by nearest neighbor retrievals and CRF.
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
import spml.utils.segsort.others as segsort_others
from spml.data.datasets.base_dataset import ListDataset
from spml.config.default import config
from spml.config.parse_args import parse_args
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab
from spml.models.predictions.segsort import segsort
from spml.models.crf import DenseCRF

cudnn.enabled = True
cudnn.benchmark = True


def separate_comma(str_comma):
  ints = [int(i) for i in str_comma.split(',')]
  return ints


def main():
  """Generate pseudo labels by nearest neighbor retrievals.
  """
  # Retreve experiment configurations.
  args = parse_args('Generate pseudo labels by nearest neighbor retrievals.')
  config.network.kmeans_num_clusters = separate_comma(args.kmeans_num_clusters)
  config.network.label_divisor = args.label_divisor

  # Create directories to save results.
  semantic_dir = os.path.join(args.save_dir, 'semantic_gray')
  semantic_rgb_dir = os.path.join(args.save_dir, 'semantic_color')

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

  if config.network.prediction_types == 'segsort':
    prediction_model = segsort(config)
  else:
    raise ValueError('Not support ' + config.network.prediction_types)

  embedding_model = embedding_model.to("cuda:0")
  prediction_model = prediction_model.to("cuda:0")
  embedding_model.eval()
  prediction_model.eval()
      
  # Load trained weights.
  model_path_template = os.path.join(args.snapshot_dir, 'model-{:d}.pth')
  save_iter = config.train.max_iteration - 1
  embedding_model.load_state_dict(
      torch.load(model_path_template.format(save_iter))['embedding_model'],
      resume=True)
  #prediction_model.load_state_dict(
  #    torch.load(model_path_template.format(save_iter))['prediction_model'])

  # Define CRF.
  postprocessor = DenseCRF(
      iter_max=args.crf_iter_max,
      pos_xy_std=args.crf_pos_xy_std,
      pos_w=args.crf_pos_w,
      bi_xy_std=args.crf_bi_xy_std,
      bi_rgb_std=args.crf_bi_rgb_std,
      bi_w=args.crf_bi_w,)

  # Load memory prototypes.
  semantic_memory_prototypes, semantic_memory_prototype_labels = None, None
  if args.semantic_memory_dir is not None:
    semantic_memory_prototypes, semantic_memory_prototype_labels = (
      segsort_others.load_memory_banks(args.semantic_memory_dir))
    semantic_memory_prototypes = semantic_memory_prototypes.to("cuda:0")
    semantic_memory_prototype_labels = semantic_memory_prototype_labels.to("cuda:0")

    # Remove ignore class.
    valid_prototypes = torch.ne(
        semantic_memory_prototype_labels,
        config.dataset.semantic_ignore_index).nonzero()
    valid_prototypes = valid_prototypes.view(-1)
    semantic_memory_prototypes = torch.index_select(
        semantic_memory_prototypes,
        0,
        valid_prototypes)
    semantic_memory_prototype_labels = torch.index_select(
        semantic_memory_prototype_labels,
        0,
        valid_prototypes)

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
          scales=[0.5, 1, 1.5, 2],
          is_flip=True)

      lab_tags = np.unique(original_label_batch['semantic_label'])
      lab_tags = lab_tags[lab_tags < config.dataset.num_classes]
      label_tags = np.zeros((config.dataset.num_classes,), dtype=np.bool)
      label_tags[lab_tags] = True

      semantic_topks = []
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
        #for k, v in label_batch.items():
        #  label_batch[k] = torch.LongTensor(v[np.newaxis, ...]).to("cuda:0")

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
        fake_sem_lab = fake_label_batch['semantic_label'][..., :resize_image_h, :resize_image_w]
        fake_inst_lab = fake_label_batch['instance_label'][..., :resize_image_h, :resize_image_w]
        embs = embeddings['embedding'][..., :resize_image_h, :resize_image_w]
        clustering_outputs = embedding_model.generate_clusters(
            embs,
            fake_sem_lab,
            fake_inst_lab)
        embeddings.update(clustering_outputs)

        # Generate predictions.
        outputs = prediction_model(
            embeddings,
            {'semantic_memory_prototype': semantic_memory_prototypes,
             'semantic_memory_prototype_label': semantic_memory_prototype_labels},
            with_loss=False, with_prediction=True)
        semantic_topk = common_utils.one_hot(
            outputs['semantic_score'], config.dataset.num_classes).float()
        semantic_topk = torch.mean(semantic_topk, dim=1)
        semantic_topk = semantic_topk.view(resize_image_h, resize_image_w, -1)
        #print(semantic_topk.shape)
        semantic_topk = (
          semantic_topk.data.cpu().numpy().astype(np.float32))
        semantic_topk = cv2.resize(
            semantic_topk, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        if data_info['is_flip']:
          semantic_topk = semantic_topk[:, ::-1]
        semantic_topks.append(semantic_topk)

      # Save semantic predictions.
      semantic_topks = np.stack(semantic_topks, axis=0).astype(np.float32)
      #print(semantic_topks.shape)
      semantic_prob = np.mean(semantic_topks, axis=0)
      semantic_prob = semantic_prob.transpose(2, 0, 1)

      # Normalize prediction.
      C, H, W = semantic_prob.shape
      max_prob = np.max(np.reshape(semantic_prob, (C, -1)), axis=1)
      max_prob = np.maximum(max_prob, 0.15)
      max_prob = np.reshape(max_prob, (C, 1, 1))
      max_prob[~label_tags, :, :] = 1
      semantic_prob = semantic_prob / max_prob

      # DenseCRF post-processing.
      image = original_image_batch['image'].astype(np.float32)
      image = image.transpose(1, 2, 0)
      image *= np.reshape(config.network.pixel_stds, (1, 1, 3))
      image += np.reshape(config.network.pixel_means, (1, 1, 3))
      image = image * 255
      image = image.astype(np.uint8)

      semantic_prob = postprocessor(image, semantic_prob)

      semantic_pred = np.argmax(semantic_prob, axis=0).astype(np.uint8)

      semantic_pred_name = os.path.join(semantic_dir, base_name)
      if not os.path.isdir(os.path.dirname(semantic_pred_name)):
        os.makedirs(os.path.dirname(semantic_pred_name))
      Image.fromarray(semantic_pred, mode='L').save(semantic_pred_name)

      semantic_pred_rgb = color_map[semantic_pred]
      semantic_pred_rgb_name = os.path.join(
          semantic_rgb_dir, base_name)
      if not os.path.isdir(os.path.dirname(semantic_pred_rgb_name)):
        os.makedirs(os.path.dirname(semantic_pred_rgb_name))
      Image.fromarray(semantic_pred_rgb, mode='RGB').save(
          semantic_pred_rgb_name)


if __name__ == '__main__':
  main()
