"""Genereate pseudo labels by random walk and CRF for densepose.
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
from spml.models.embeddings.resnet_pspnet_densepose import resnet_101_pspnet
import spml.models.utils as model_utils
import spml.utils.segsort.common as segsort_common
import spml.utils.general.common as common_utils
from spml.models.crf import DenseCRF

cudnn.enabled = True
cudnn.benchmark = True

WALK_STEPS=6
TH=None


def separate_comma(str_comma):
  ints = [int(i) for i in str_comma.split(',')]
  return ints


def main():
  """Generate pseudo labels by random walk and CRF for DensePose.
  """
  # Retreve experiment configurations.
  args = parse_args('Generate pseudo labels by random walk and CRF for DensePose.')
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

  # Define CRF.
  postprocessor = DenseCRF(
      iter_max=args.crf_iter_max,
      pos_xy_std=args.crf_pos_xy_std,
      pos_w=args.crf_pos_w,
      bi_xy_std=args.crf_bi_xy_std,
      bi_rgb_std=args.crf_bi_rgb_std,
      bi_w=args.crf_bi_w,)

  # Create models.
  if config.network.backbone_types == 'panoptic_pspnet_101':
    embedding_model = resnet_101_pspnet(config).cuda()
  elif config.network.backbone_types == 'panoptic_deeplab_101':
    embedding_model = resnet_101_deeplab(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.backbone_types)

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
      # Image path.
      image_path = test_image_paths[data_index]
      base_name = os.path.basename(image_path).replace('.jpg', '.png')

      # Image resolution.
      image_batch, label_batch, _ = test_dataset[data_index]
      image_h, image_w = image_batch['image'].shape[-2:]

      lab_tags = np.unique(label_batch['semantic_label'])
      lab_tags = lab_tags[lab_tags < config.dataset.num_classes]
      label_tags = np.zeros((config.dataset.num_classes,), dtype=np.bool)
      label_tags[lab_tags] = True
      label_tags = torch.from_numpy(label_tags).cuda()

      # Pad the input image.
      image_batch['image'] = transforms.resize_with_pad(
          image_batch['image'].transpose(1, 2, 0),
          config.test.crop_size,
          image_pad_value=0).transpose(2, 0, 1)
      image_batch['image'] = torch.FloatTensor(
          image_batch['image'][np.newaxis, ...]).cuda()
      pad_image_h, pad_image_w = image_batch['image'].shape[-2:]

      original_semantic_label = label_batch['semantic_label'].copy()
      lab = label_batch['semantic_label']
      lab[lab == 255] = config.dataset.num_classes
      for lab in label_batch.keys():
        label_batch[lab] = torch.LongTensor(
            label_batch[lab][np.newaxis, ...]).cuda()

      embeddings = embedding_model.generate_embeddings(
          image_batch, resize_as_input=True)
      embeddings['embedding'] = F.interpolate(
          embeddings['embedding'],
          size=(pad_image_h//2, pad_image_w//2),
          mode='bilinear')
      embeddings['embedding'] = (
        embeddings['embedding'][:, :, :image_h//2, :image_w//2])

      # Create affinity matrix.
      embs = embeddings['embedding']
      embs = F.interpolate(embs, size=(image_h//8, image_w//8), mode='bilinear')
      embs = embs / torch.norm(embs, dim=1)
      embs_flat = embs.view(embs.shape[1], -1)
      aff = torch.matmul(embs_flat.t(), embs_flat).mul_(5).add_(-5).exp_()

      # Assign unknown labels to nearest neighbor.
      size = embeddings['embedding'].shape[-2:]
      s_lab = common_utils.resize_labels(
          label_batch['semantic_label'], size)
      i_lab = common_utils.resize_labels(
          label_batch['instance_label'], size)
      clusterings = embedding_model.generate_clusters(
          embeddings['embedding'], s_lab, i_lab)

      s_labs, c_inds = segsort_common.prepare_prototype_labels(
          clusterings['cluster_semantic_label'],
          clusterings['cluster_index'],
          clusterings['cluster_semantic_label'].max() + 1)
      embs = clusterings['cluster_embedding']
      protos = segsort_common.calculate_prototypes_from_labels(embs, c_inds)
      s_tags = model_utils.gather_multiset_labels_per_batch_by_nearest_neighbor(
          embs,
          protos,
          s_labs,
          torch.zeros_like(clusterings['cluster_semantic_label']),
          torch.zeros_like(s_labs),
          num_classes=config.dataset.num_classes,
          top_k=1,
          threshold=-1,
          label_divisor=config.network.label_divisor)
      #s_labs = torch.argmax(s_tags, dim=1)
      #semantic_pred = torch.gather(s_labs, 0, c_inds)
      #semantic_pred = s_labs
      s_probs = common_utils.segment_mean(
          s_tags.float(), c_inds)
      s_probs = s_probs / s_probs.sum(dim=1, keepdims=True)
      semantic_probs = torch.index_select(s_probs, 0, c_inds)
      #semantic_pred = torch.argmax(semantic_probs, dim=1)
      #semantic_pred = (semantic_pred.view(image_h, image_w)
      #                              .data.cpu().numpy().astype(np.uint8))
      semantic_probs = semantic_probs.view(1, image_h//2, image_w//2, -1)
      semantic_probs = semantic_probs.permute(0, 3, 1, 2).contiguous()
      semantic_probs = F.interpolate(
          semantic_probs, size=(image_h//8, image_w//8), mode='bilinear')
      max_prob = torch.max(semantic_probs.view(15, -1), dim=1)[0]
      cam_full_arr = semantic_probs / max_prob.view(15, 1, 1)

      cam_shape = cam_full_arr.shape[-2:]
      label_tags = (~label_tags).view(-1, 1, 1).expand(-1, cam_shape[0], cam_shape[1])
      cam_full_arr = cam_full_arr.masked_fill(label_tags, 0)
      if TH is not None:
        cam_full_arr[0] = TH

      # Start random walk.
      aff_mat = aff ** 20

      trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
      for _ in range(WALK_STEPS):
        trans_mat = torch.matmul(trans_mat, trans_mat)

      cam_vec = cam_full_arr.view(15, -1)
      cam_rw = torch.matmul(cam_vec, trans_mat)
      cam_rw = cam_rw.view(15, cam_shape[0], cam_shape[1])

      cam_rw = cam_rw.data.cpu().numpy()
      cam_rw = cv2.resize(cam_rw.transpose(1, 2, 0),
                          dsize=(image_w, image_h),
                          interpolation=cv2.INTER_LINEAR)
      cam_rw_pred = np.argmax(cam_rw, axis=-1).astype(np.uint8)

      # CRF
      image = image_batch['image'].data.cpu().numpy().astype(np.float32)
      image = image[0, :, :image_h, :image_w].transpose(1, 2, 0)
      image *= np.reshape(config.network.pixel_stds, (1, 1, 3))
      image += np.reshape(config.network.pixel_means, (1, 1, 3))
      image = image * 255
      image = image.astype(np.uint8)
      cam_rw = postprocessor(image, cam_rw.transpose(2,0,1))

      cam_rw_pred = np.argmax(cam_rw, axis=0).astype(np.uint8)

      # Ignore regions.
      ignore_mask = original_semantic_label == 255
      cam_rw_pred[ignore_mask] = 255

      # Save semantic predictions.
      semantic_pred = cam_rw_pred

      semantic_pred_name = os.path.join(
          semantic_dir, base_name)
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
