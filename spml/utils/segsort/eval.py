"""Utility functions for eval.
"""

import torch

import spml.utils.general.common as common_utils


def top_k_ranking(embeddings,
                  labels,
                  prototypes,
                  prototype_labels,
                  top_k=3):
  """Compute top-k accuracy based on embeddings and prototypes
  affinity.

  Args:
    embeddings: An N-D float tensor with last dimension
      as `num_channels`.
    labels: An (N-1)-D long tensor.
    prototypes: A 2-D float tensor with last dimension as
      `num_channels`.
    prototype_labels: A 1-D long tensor.
    top_k: A scalar indicates number of top-ranked retrievals.

  Returns:
    A float scalar indicates accuracy;
    A 2-D long tensor indicates retrieved top-k labels.
  """
  embeddings = embeddings.view(-1, embeddings.shape[-1])
  prototypes = prototypes.view(-1, prototypes.shape[-1])
  feature_affinity = torch.mm(embeddings, prototypes.t())
  top_k_indices = torch.argsort(feature_affinity, 1, descending=True)
  top_k_indices = top_k_indices[:, :top_k].contiguous()
  #top_k_indices = top_k_indices[:, 1:top_k+1].contiguous()

  labels = labels.view(-1, 1)
  prototype_labels = prototype_labels.view(1, -1)
  label_affinity = torch.eq(labels, prototype_labels)

  # Compute top-k accuracy.
  top_k_true_positive = torch.gather(label_affinity, 1, top_k_indices)
  top_k_accuracy = torch.mean(top_k_true_positive.float())

  # Retrieve top-k labels.
  top_k_labels = torch.gather(
      prototype_labels.view(-1),
      0,
      top_k_indices.view(-1))
  top_k_labels = top_k_labels.view(-1, top_k)

  return top_k_accuracy, top_k_labels


def majority_label_from_topk(top_k_labels, num_classes=None):
  """Compute majority label from top-k retrieved labels.
  
  Args:
    top_k_labels: A 2-D long tensor with shape `[num_queries, top_k]`.

  Returns:
    A 1-D long tensor with shape `[num_queries]`.
  """
  one_hot_top_k_labels = common_utils.one_hot(top_k_labels,
                                              num_classes)
  one_hot_top_k_labels = torch.sum(one_hot_top_k_labels,
                                   dim=1)
  majority_labels = torch.argmax(one_hot_top_k_labels, 1)

  return majority_labels
