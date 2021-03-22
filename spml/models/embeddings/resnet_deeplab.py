"""Build segmentation model with Deeplab.v2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spml.models.utils as model_utils
import spml.utils.general.common as common_utils
import spml.utils.segsort.common as segsort_common
from spml.models.heads.spp import ASPP
from spml.models.backbones.resnet import ResnetBackbone
from spml.models.embeddings.base_model import ResnetBase
from spml.models.embeddings.local_model import LocationColorNetwork


class ResnetDeeplab(ResnetBase):
    
  def __init__(self, backbone_depth, strides, dilations, config):
    """Build Deeplab.v2 using ResNet as backbone network.

    Args:
      backbone_depth: A list of integers indicate the number
        of residual layers in each block.
      strides: A list of intergers indicate the stride.
      dilations: A list of integers indicate the dilations.
      config: An easydict of the network configurations.
    """

    super(ResnetDeeplab, self).__init__()

    # Build Backbone Network.
    self.resnet_backbone = ResnetBackbone(backbone_depth, strides,
                                          dilations, config)

    # Build Feature Pyramid Network.
    N = len(dilations)
    self.aspp = ASPP(2048,
                     config.network.embedding_dim,
                     bn=False,
                     relu=False)

    # Build Local Feature Network.
    self.lfn = LocationColorNetwork(use_color=False, use_location=True,
                                    norm_color=False, smooth_ksize=None)

    # Parameters for VMF clustering.
    self.label_divisor = config.network.label_divisor
    self.num_classes = config.dataset.num_classes

    self.semantic_ignore_index = config.dataset.semantic_ignore_index

    self.kmeans_num_clusters = config.network.kmeans_num_clusters
    self.kmeans_iterations = config.network.kmeans_iterations

    self.initialize()

  def generate_embeddings(self, datas, targets=None, resize_as_input=False):
    """Feed-forward segmentation model to generate pixel-wise embeddings
    and location & RGB features.

    Args:
      datas: A dict with an entry `image`, which is a 4-D float tensor
        of shape `[batch_size, channels, height, width]`.
      targets: A dict with an entry `semantic_label` and `instance_label`,
        which are 3-D long tensors of shape `[batch_size, height, width]`.
      resize_as_input: enable/disable resize_as_input to upscale the 
        embeddings to the same size as the input image.

    Return:
      A dict with entry `embedding` and `local_feature` of shape
      `[batch_size, channels, height, width]`.
    """

    # Generate embeddings.
    _, _, _, res5 = self.resnet_backbone(datas['image'])

    embeddings = self.aspp(res5)
    embeddings = F.interpolate(embeddings, scale_factor=2, mode='bilinear')

    if resize_as_input:
      input_size = datas['image'].shape[-2:]
      embeddings = F.interpolate(
          embeddings, size=input_size, mode='bilinear')

    size = embeddings.shape[-2:]
    local_features = self.lfn(datas['image'], size=size)

    return {'embedding': embeddings, 'local_feature': local_features}

  def generate_clusters(self, embeddings,
                        semantic_labels,
                        instance_labels,
                        local_features=None):
    """Perform Spherical KMeans clustering within each image.

    Args:
      embeddings: A a 4-D float tensor of shape
        `[batch_size, channels, height, width]`.
      semantic_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      instance_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      local_features: A 4-D float tensor of shape
        `[batch_size, height, width, channels]`.

    Return:
      A dict with entry `cluster_embedding`, `cluster_embedding_with_loc`,
      `cluster_semantic_label`, `cluster_instance_label`, `cluster_index`
      and `cluster_batch_index`.
    """

    if semantic_labels is not None and instance_labels is not None:
      labels = semantic_labels * self.label_divisor + instance_labels
      ignore_index = labels.max() + 1
      labels = labels.masked_fill(
          semantic_labels == self.semantic_ignore_index,
          ignore_index)
    else:
      labels = None
      ignore_index = None

    # Spherical KMeans clustering.
    (cluster_embeddings,
     cluster_embeddings_with_loc,
     cluster_labels,
     cluster_indices,
     cluster_batch_indices) = (
       segsort_common.segment_by_kmeans(
           embeddings,
           labels,
           self.kmeans_num_clusters,
           local_features=local_features,
           ignore_index=ignore_index,
           iterations=self.kmeans_iterations))

    cluster_semantic_labels = cluster_labels // self.label_divisor
    cluster_instance_labels = cluster_labels % self.label_divisor

    outputs = {
      'cluster_embedding': cluster_embeddings,
      'cluster_embedding_with_loc': cluster_embeddings_with_loc,
      'cluster_semantic_label': cluster_semantic_labels,
      'cluster_instance_label': cluster_instance_labels,
      'cluster_index': cluster_indices,
      'cluster_batch_index': cluster_batch_indices,
    }

    return outputs

  def forward(self, datas, targets=None, resize_as_input=None):
    """Generate pixel-wise embeddings and Spherical Kmeans clustering
    within each image.
    """

    targets = targets if targets is not None else {}

    # Generaet embeddings.
    outputs = self.generate_embeddings(datas, targets, resize_as_input)

    # Resize labels to embedding size.
    semantic_labels = targets.get('semantic_label', None)
    if semantic_labels is not None:
      semantic_labels = common_utils.resize_labels(
          semantic_labels, outputs['embedding'].shape[-2:])

    instance_labels = targets.get('instance_label', None)
    if instance_labels is not None:
      instance_labels = common_utils.resize_labels(
          instance_labels, outputs['embedding'].shape[-2:])

    # Generate clusterings.
    cluster_embeddings = self.generate_clusters(
        outputs['embedding'],
        semantic_labels,
        instance_labels,
        outputs['local_feature'])

    outputs.update(cluster_embeddings)

    return outputs

  def initialize(self):
    pass

  def get_params_lr(self):
    """Helper function to adjust learning rate for each sub modules.
    """
    # Specify learning rate for each sub modules.
    ret = []
    resnet_params_name = ['resnet_backbone.res3',
                          'resnet_backbone.res4',
                          'resnet_backbone.res5']
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          resnet_params_name,
          ['weight'])],
      'lr': 1})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          resnet_params_name,
          ['bias'])],
      'lr': 2,
      'weight_decay': 0})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['aspp'],
          ['weight'])],
      'lr': 10})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['aspp'],
          ['bias'])],
      'lr': 20,
      'weight_decay': 0})

    return ret

  def name_mapping(self, name, resume=False):
    if resume:
      return name if not name.startswith('module.') else name[len('module.'):]

    if name.startswith('conv1') or name.startswith('bn1'):
      return 'resnet_backbone.conv1.' + name

    return name.replace('layer1', 'resnet_backbone.res2')\
               .replace('layer2', 'resnet_backbone.res3')\
               .replace('layer3', 'resnet_backbone.res4')\
               .replace('layer4', 'resnet_backbone.res5')

def resnet_101_deeplab(config):
  """Deeplab.v2 with resnet101 backbone.
  """
  return ResnetDeeplab([3, 4, 23, 3], [1, 2, 1, 1], [1, 1, 2, 4], config)


def resnet_50_deeplab(config):
  """Deeplab.v2 with resnet50 backbone.
  """
  return ResnetDeeplab([3, 4, 6, 3], [1, 2, 1, 1], [1, 1, 2, 4], config)
