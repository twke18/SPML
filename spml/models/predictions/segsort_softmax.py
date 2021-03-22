"""Define SegSort with Softmax Classifier for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spml.models.utils as model_utils
import spml.utils.general.common as common_utils
import spml.utils.segsort.loss as segsort_loss
import spml.utils.segsort.eval as segsort_eval
import spml.utils.segsort.common as segsort_common


class SegsortSoftmax(nn.Module):

  def __init__(self, config):
    
    super(SegsortSoftmax, self).__init__()

    # Softmax classifier head.
    self.semantic_classifier = nn.Sequential(
        nn.Conv2d(config.network.embedding_dim,
          config.network.embedding_dim*2,
          #kernel_size=1,
          kernel_size=3,
          padding=1,
          stride=1,
          bias=False),
        nn.BatchNorm2d(config.network.embedding_dim*2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.75),
        nn.Conv2d(config.network.embedding_dim*2,
          config.dataset.num_classes,
          kernel_size=1,
          stride=1,
          bias=True))
    self.softmax_loss = nn.CrossEntropyLoss(
        ignore_index=config.dataset.semantic_ignore_index)

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
    """Predict semantic segmentation by Softmax Classifier.
    """

    # Predict semantic labels.
    embeddings = datas['embedding']
    embeddings = (
      embeddings / torch.norm(embeddings, dim=1, keepdim=True))
    semantic_logits = self.semantic_classifier(embeddings)

    semantic_pred = torch.argmax(semantic_logits, dim=1)


    return semantic_pred, semantic_logits

  def losses(self, datas, targets={}):
    """Compute losses.
    """
    sem_ann_loss = None
    sem_occ_loss = None
    img_sim_loss = None
    sem_ann_acc = None

    # Compute softmax loss.
    embeddings = datas['embedding'].detach()
    embeddings = (
      embeddings / torch.norm(embeddings, dim=1, keepdim=True))
    semantic_logits = self.semantic_classifier(embeddings)

    semantic_labels = targets.get('semantic_label', None)
    semantic_logits = F.interpolate(
        semantic_logits,
        size=semantic_labels.shape[-2:],
        mode='bilinear')
    semantic_labels = semantic_labels.masked_fill(
        semantic_labels >= self.num_classes, self.semantic_ignore_index)

    #prob = F.softmax(semantic_logits, dim=1)
    #mask = torch.max(prob, dim=1)[0] <= 0.5
    #semantic_labels = semantic_labels.masked_fill(
    #    mask, self.semantic_ignore_index)
    semantic_labels = semantic_labels.squeeze_(1).long()

    sem_ann_loss = self.softmax_loss(semantic_logits, semantic_labels)


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

      sem_ann_loss += self.sem_ann_loss (
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
      semantic_pred, semantic_logits = self.predictions(datas, targets)

      outputs.update({'semantic_prediction': semantic_pred,
                      'semantic_logit': semantic_logits,})

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
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['semantic_classifier'],
          ['weight'])],
      'lr': 10})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['semantic_classifier'],
          ['bias'])],
      'lr': 20,
      'weight_decay': 0})

    return ret


def segsort(config):
  """Paramteric prototype predictor.
  """
  return SegsortSoftmax(config)
