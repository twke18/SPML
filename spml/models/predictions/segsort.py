"""Define SegSort for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spml.models.utils as model_utils
import spml.utils.general.common as common_utils
import spml.utils.segsort.loss as segsort_loss
import spml.utils.segsort.eval as segsort_eval
import spml.utils.segsort.common as segsort_common


class Segsort(nn.Module):

  def __init__(self, config):
    
    super(Segsort, self).__init__()

    # Define regularization by semantic annotation.
    self.sem_ann_loss = self._construct_loss(
        config.train.sem_ann_loss_types,
        concentration=config.train.sem_ann_concentration)
    self.sem_ann_loss_weight = config.train.sem_ann_loss_weight

    # Define regularization by semantic cooccurrence.
    loss_type = (
      'set_segsort' if config.train.sem_occ_loss_types == 'segsort' else 'none')
    self.sem_occ_loss = self._construct_loss(
        loss_type,
        concentration=config.train.sem_occ_concentration)
    self.sem_occ_loss_weight = config.train.sem_occ_loss_weight

    # Define regularization by low-level image similarity.
    self.img_sim_loss = self._construct_loss(
        config.train.img_sim_loss_types,
        concentration=config.train.img_sim_concentration)
    self.img_sim_loss_weight = config.train.img_sim_loss_weight

    # Define regularization by feature affinity.
    loss_type = (
      'set_segsort' if config.train.feat_aff_loss_types == 'segsort' else 'none')
    self.feat_aff_loss = self._construct_loss(
        config.train.feat_aff_loss_types,
        concentration=config.train.feat_aff_concentration)
    self.feat_aff_loss_weight = config.train.feat_aff_loss_weight

    self.semantic_ignore_index = config.dataset.semantic_ignore_index
    self.num_classes = config.dataset.num_classes
    self.label_divisor = config.network.label_divisor

  def _construct_loss(self, loss_types, **kwargs):

    if loss_types == 'segsort':
      return segsort_loss.SegSortLoss(kwargs['concentration'],
                                      group_mode='segsort+',
                                      reduction='mean')
    elif loss_types == 'set_segsort':
      return segsort_loss.SetSegSortLoss(kwargs['concentration'],
                                         group_mode='segsort+',
                                         reduction='mean')
    elif loss_types == 'none':
      return None
    else:
      raise KeyError('Unsupported loss types: {:s}'.format(loss_types))

  def predictions(self, datas, targets={}):
    """Predict semantic segmentation by Nearest Neighbor Retrievals.
    """

    semantic_pred, semantic_topk = None, None

    # Predict Semantic Segmentation.
    semantic_memory_prototypes = targets.get(
        'semantic_memory_prototype', None)
    semantic_memory_prototype_labels = targets.get(
        'semantic_memory_prototype_label', None)
    semantic_cluster_embeddings = datas.get(
        'cluster_embedding', None)
    semantic_cluster_indices = datas.get(
        'cluster_index', None)
    if (semantic_memory_prototypes is not None
        and semantic_memory_prototype_labels is not None
        and semantic_cluster_embeddings is not None
        and semantic_cluster_indices is not None):
      # Predict semantic labels by retrieving nearest neighbors.
      _, semantic_cluster_indices = torch.unique(
          semantic_cluster_indices, return_inverse=True)
      num_prototypes = semantic_cluster_indices.max() + 1
      semantic_prototypes = (
        segsort_common.calculate_prototypes_from_labels(
            semantic_cluster_embeddings,
            semantic_cluster_indices,
            num_prototypes))
      semantic_prototype_labels = torch.zeros(
          num_prototypes, dtype=torch.long,
          device=semantic_prototypes.device)

      N = semantic_prototype_labels.shape[0]
      device = semantic_prototype_labels.device
      semantic_pred = torch.zeros((N,), dtype=torch.long, device=device)
      semantic_topk = torch.zeros((N, 20), dtype=torch.long, device=device)
      num_groups = min(10, N-1)
      r = N // num_groups
      split_indices = [i * r for i in range(num_groups)] + [N]
      #for i in range(semantic_prototypes.shape[0]):
      for i in range(num_groups):
        st, ed = split_indices[i], split_indices[i+1]
        _, top_k_semantic_labels = segsort_eval.top_k_ranking(
            semantic_prototypes[st:ed,],
            semantic_prototype_labels[st:ed],
            semantic_memory_prototypes,
            semantic_memory_prototype_labels,
            20)
        majority_semantic_labels = segsort_eval.majority_label_from_topk(
            top_k_semantic_labels)
        semantic_pred[st:ed] = majority_semantic_labels
        semantic_topk[st:ed] = top_k_semantic_labels
      semantic_pred = torch.gather(
          semantic_pred, 0, semantic_cluster_indices)
      semantic_topk = torch.index_select(
          semantic_topk, 0, semantic_cluster_indices)

    return semantic_pred, semantic_topk

  def losses(self, datas, targets={}):
    """Compute losses.
    """
    sem_ann_loss = None
    sem_occ_loss = None
    img_sim_loss = None
    sem_ann_acc = None

    # Compute semantic annotation and semantic co-occurrence loss.
    if self.sem_ann_loss is not None or self.sem_occ_loss is not None:
      cluster_indices = datas['cluster_index']
      embeddings = datas['cluster_embedding']
      semantic_labels = datas['cluster_semantic_label']
      batch_indices = datas['cluster_batch_index']

      prototypes = targets['prototype']
      prototype_semantic_labels = targets['prototype_semantic_label']
      prototype_batch_indices = targets['prototype_batch_index']

      # Extract image tags.
      semantic_tags = torch.index_select(
          targets['semantic_tag'][:, 1:self.num_classes],
          0, batch_indices)
      prototype_semantic_tags = (
        targets['prototype_semantic_tag'][:, 1:self.num_classes])

      # Add prototypes in the memory bank.
      memory_prototypes = targets.get(
          'memory_prototype', [])
      memory_prototype_semantic_labels = targets.get(
          'memory_prototype_semantic_label', [])
      memory_prototype_batch_indices = targets.get(
          'memory_prototype_batch_index', [])
      memory_prototype_semantic_tags = targets.get(
          'memory_prototype_semantic_tag', [])
      if (memory_prototypes
          and memory_prototype_semantic_labels
          and memory_prototype_semantic_tags
          and memory_prototype_batch_indices):
        prototypes = [prototypes]
        prototypes += memory_prototypes
        prototype_semantic_labels = [prototype_semantic_labels]
        prototype_semantic_labels += memory_prototype_semantic_labels
        memory_prototype_semantic_tags = [
          lab[:, 1:self.num_classes]
            for lab in memory_prototype_semantic_tags]
        prototype_semantic_tags = [prototype_semantic_tags]
        prototype_semantic_tags += memory_prototype_semantic_tags
        prototype_batch_indices = [prototype_batch_indices]
        prototype_batch_indices += memory_prototype_batch_indices
        prototypes = torch.cat(prototypes, dim=0)
        prototype_semantic_labels = torch.cat(
            prototype_semantic_labels, dim=0)
        prototype_semantic_tags = torch.cat(
            prototype_semantic_tags, dim=0)
        prototype_batch_indices = torch.cat(
            prototype_batch_indices, dim=0)

      pixel_inds = (semantic_labels < self.num_classes).nonzero().view(-1)
      proto_inds = (prototype_semantic_labels < self.num_classes).nonzero().view(-1)
      c_inds = torch.arange(
          prototypes.shape[0], dtype=torch.long,
          device=prototypes.device)
      c_inds = c_inds.masked_fill(
          prototype_semantic_labels >= self.num_classes,
          c_inds.max() + 1)
      _, c_inds = torch.unique(c_inds, return_inverse=True)
      new_cluster_indices = torch.gather(
          c_inds, 0, cluster_indices)

      sem_ann_loss = self.sem_ann_loss (
          torch.index_select(embeddings, 0, pixel_inds),
          torch.index_select(semantic_labels, 0, pixel_inds),
          torch.index_select(new_cluster_indices, 0, pixel_inds),
          torch.index_select(prototypes, 0, proto_inds),
          torch.index_select(prototype_semantic_labels, 0, proto_inds))
      sem_ann_loss *= self.sem_ann_loss_weight

      sem_occ_loss = self.sem_occ_loss(
          embeddings,
          semantic_tags,
          cluster_indices,
          prototypes,
          prototype_semantic_tags)
      sem_occ_loss *= self.sem_occ_loss_weight

      sem_ann_acc, _ = segsort_eval.top_k_ranking(
          prototypes,
          prototype_semantic_labels,
          prototypes,
          prototype_semantic_labels,
          5)

    # Compute low-level image similarity loss.
    if self.img_sim_loss is not None:
      cluster_indices = datas['cluster_index']
      embeddings = datas['cluster_embedding_with_loc']
      instance_labels = datas['cluster_instance_label']
      batch_indices = datas['cluster_batch_index']

      img_sim_loss = []
      for batch_ind in torch.unique(batch_indices):
        batch_mask = batch_indices == batch_ind
        inds = batch_mask.nonzero().view(-1)
        embs = torch.index_select(embeddings, 0, inds)
        labs = torch.index_select(instance_labels, 0, inds)
        c_inds = torch.index_select(cluster_indices, 0, inds)
        p_labs, c_inds = segsort_common.prepare_prototype_labels(
            labs, c_inds, labs.max() + 1)
        protos = (
          segsort_common.calculate_prototypes_from_labels(embs, c_inds))
        img_sim_loss.append(self.img_sim_loss(
            embs, labs, c_inds, protos, p_labs))
      img_sim_loss = sum(img_sim_loss) / len(img_sim_loss)
      img_sim_loss *= self.img_sim_loss_weight

    return sem_ann_loss, sem_occ_loss, img_sim_loss, sem_ann_acc

  def forward(self, datas, targets=None,
              with_loss=True, with_prediction=False):
    """Compute loss and predictions.
    """
    targets = targets if targets is not None else {}
    outputs = {}

    if with_prediction:
      # Predict semantic and instance labels.
      semantic_pred, semantic_score = self.predictions(datas, targets)

      outputs.update({'semantic_prediction': semantic_pred,
                      'semantic_score': semantic_score,})

    if with_loss:
      sem_ann_loss, sem_occ_loss, img_sim_loss, sem_ann_acc = (
          self.losses(datas, targets))

      outputs.update(
          {'sem_ann_loss': sem_ann_loss,
           'sem_occ_loss': sem_occ_loss,
           'img_sim_loss': img_sim_loss,
           'accuracy': sem_ann_acc})

    return outputs

  def get_params_lr(self):
    """Helper function to adjust learning rate for each sub modules.
    """
    # Specify learning rate for each sub modules.
    ret = []

    return ret


def segsort(config):
  """Non-paramteric prototype predictor.
  """
  return Segsort(config)
