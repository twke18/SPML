"""Utility functions for defining Pixel-Segment Contrastive Loss.
"""
# This code is borrowed and re-implemented from
# https://github.com/jyhjinghwang/SegSort/blob/master/network/segsort/train_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import spml.utils.segsort.common as segsort_common
import spml.utils.general.common as common_utils


def _calculate_log_likelihood(embeddings,
                              semantic_labels,
                              instance_labels,
                              prototypes,
                              prototype_semantic_labels,
                              concentration,
                              group_mode):
  """Calculates log-likelihood of each pixel belonging to a certain
  prototype.

  This function calculates log-likelihood of each pixel belonging to
  a certain cluster. This log-likelihood is then used as maximum
  likelihood estimation.

  Args:
    embeddings: A 2-D float tensor with shape
      `[num_pixels, embedding_dim]`.
    instance_labels: A 1-D long tensor with length `[num_pixels]`.
      It contains the instance label for each pixel.
    semantic_labels: A 1-D long tensor with length `[num_pixels]`.
      It contains the semantic label for each pixel.
    prototypes: A 2-D float tensor with shape
      `[num_prototypes, embedding_dim]`.
    prototype_semantic_labels: A 1-D long tensor with length
      `[num_prototypes]`. It contains the semantic label for each
      prototype

  Returns:
    log_likelihood: A 1-D float tensor with length `[num_pixels]`.
      It is the negative log-likelihood for each pixel.
  """
  # Compute pixel to all prototypes similarities.
  embeddings = embeddings.view(-1, embeddings.shape[-1])
  prototypes = prototypes.view(-1, prototypes.shape[-1])
  similarities = (torch.mm(embeddings, prototypes.t())
                       .mul_(concentration)
                       .exp_())

  # Extract pixel to self prototype similarities.
  instance_labels = instance_labels.view(-1, 1)
  semantic_labels = semantic_labels.view(-1, 1)
  prototype_semantic_labels = prototype_semantic_labels.view(1, -1)

  pixel_to_prototype_similarities = torch.gather(
      similarities, 1, instance_labels)

  if group_mode == 'segsort+':
    same_semantic_array = torch.eq(semantic_labels, prototype_semantic_labels)
    same_semantic_array = same_semantic_array.float()
    same_semantic_similarities = torch.sum(
        similarities * same_semantic_array, 1, keepdim=True)
    same_semantic_similarities -= pixel_to_prototype_similarities
    numerator = torch.where(
        torch.gt(same_semantic_similarities, 0),
        same_semantic_similarities,
        pixel_to_prototype_similarities)
  else:
    numerator = pixel_to_prototype_similarities

  diff_semantic_array = torch.ne(semantic_labels, prototype_semantic_labels)
  diff_semantic_array = diff_semantic_array.float()
  diff_semantic_similarities = torch.sum(
      similarities * diff_semantic_array, 1, keepdim=True)
  denominator = diff_semantic_similarities.add_(numerator)

  log_likelihood = (numerator / denominator).log_().mul_(-1)

  return log_likelihood


def _one_hot_calculate_log_likelihood(embeddings,
                                      semantic_labels,
                                      instance_labels,
                                      prototypes,
                                      prototype_semantic_labels,
                                      concentration,
                                      group_mode):
  """Calculates log-likelihood of each pixel belonging to a certain
  prototype.
  """
  # Compute pixel to all prototypes similarities.
  embeddings = embeddings.view(-1, embeddings.shape[-1])
  prototypes = prototypes.view(-1, prototypes.shape[-1])
  similarities = (torch.mm(embeddings, prototypes.t())
                       .mul_(concentration)
                       .exp_())

  # Extract pixel to self prototype similarities.
  instance_labels = instance_labels.view(-1, 1)

  pixel_to_prototype_similarities = torch.gather(
      similarities, 1, instance_labels)
  label_affinity = torch.mm(
      semantic_labels.float(),
      prototype_semantic_labels.t().float())

  if group_mode == 'segsort+':
    same_semantic_array = (label_affinity > 0).float()
    same_semantic_similarities = torch.sum(
        similarities * same_semantic_array, 1, keepdim=True)
    same_semantic_similarities -= pixel_to_prototype_similarities
    numerator = torch.where(
        torch.gt(same_semantic_similarities, 0),
        same_semantic_similarities,
        pixel_to_prototype_similarities)
  else:
    numerator = pixel_to_prototype_similarities

  diff_semantic_array = (label_affinity == 0).float()
  diff_semantic_similarities = torch.sum(
      similarities * diff_semantic_array, 1, keepdim=True)
  denominator = diff_semantic_similarities.add_(numerator)

  log_likelihood = (numerator / denominator).log_().mul_(-1)

  return log_likelihood


class SegSortLoss(_Loss):

  def __init__(self,
               concentration=10,
               group_mode='segsort+',
               size_average=None,
               reduce=None,
               reduction='mean'):
    super(SegSortLoss, self).__init__(size_average, reduce, reduction)
    self.concentration = concentration
    self.group_mode = group_mode

  def __repr__(self):
    return 'SegSortLoss(concentration={:.2f}, group_mode={})'.format(
        self.concentration, self.group_mode)

  def forward(self,
              embeddings,
              semantic_labels,
              instance_labels,
              prototypes,
              prototype_semantic_labels,
              prototype_weights=None):
    """Calculates NCA loss given semantic and instance labels.

    Args:
      embeddings: A 2-D float tensor with shape
        `[num_pixels, embedding_dim]`.
      semantic_labels: A 1-D long tensor with length [num_pixels]. It
        contains the semantic label for each pixel.
      instance_labels: A 1-D long tensor with length [num_pixels]. It
        contains the instance label for each pixel.
      prototypes: A 2-D float tensor with shape
        `[num_prototypes, embedding_dim]`.
      prototype_semantic_labels : A 1-D long tensor with length
        `[num_prototypes]`. It contains the semantic label for each
        prototype.

    Returns:
      loss: A float scalar for NCA loss.
    """
    log_likelihood = _calculate_log_likelihood(
        embeddings,
        semantic_labels,
        instance_labels,
        prototypes,
        prototype_semantic_labels,
        self.concentration,
        self.group_mode)

    if self.reduction == 'mean':
      loss = torch.mean(log_likelihood)
    elif self.reduction == 'sum':
      loss = torch.sum(log_likelihood)
    else:
      loss = log_likelihood

    return loss


class SetSegSortLoss(_Loss):

  def __init__(self,
               concentration=10,
               group_mode='segsort+',
               size_average=None,
               reduce=None,
               reduction='mean'):
    super(SetSegSortLoss, self).__init__(size_average, reduce, reduction)
    self.concentration = concentration
    self.group_mode = group_mode

  def __repr__(self):
    return 'SetSegSortLoss(concentration={:.2f}, group_mode={})'.format(
        self.concentration, self.group_mode)

  def forward(self,
              embeddings,
              semantic_labels,
              instance_labels,
              prototypes,
              prototype_semantic_labels,
              prototype_weights=None):
    """Calculates NCA loss given semantic and instance labels.

    Args:
      embeddings: A 2-D float tensor with shape
        `[num_pixels, embedding_dim]`.
      semantic_labels: A 2-D long tensor with length
        [num_pixels, num_classes]. It contains the semantic label
        for each pixel.
      instance_labels: A 1-D long tensor with length [num_pixels]. It
        contains the instance label for each pixel.
      prototypes: A 2-D float tensor with shape
        `[num_prototypes, embedding_dim]`.
      prototype_semantic_labels : A 2-D long tensor with length
        `[num_prototypes, num_classes]`. It contains the semantic
        label for each prototype.

    Returns:
      loss: A float scalar for NCA loss.
    """
    log_likelihood = _one_hot_calculate_log_likelihood(
        embeddings,
        semantic_labels,
        instance_labels,
        prototypes,
        prototype_semantic_labels,
        self.concentration,
        self.group_mode)

    if self.reduction == 'mean':
      loss = torch.mean(log_likelihood)
    elif self.reduction == 'sum':
      loss = torch.sum(log_likelihood)
    else:
      loss = log_likelihood

    return loss
