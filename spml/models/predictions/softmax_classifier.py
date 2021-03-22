"""Define Softmax Classifier for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spml.models.utils as model_utils


class SoftmaxClassifier(nn.Module):

  def __init__(self, config):
    super(SoftmaxClassifier, self).__init__()
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
        nn.Dropout(p=0.65),
        nn.Conv2d(config.network.embedding_dim*2,
          config.dataset.num_classes,
          kernel_size=1,
          stride=1,
          bias=True))
    self.semantic_loss = nn.CrossEntropyLoss(
        ignore_index=config.dataset.semantic_ignore_index)
    self.ignore_index = config.dataset.semantic_ignore_index
    self.num_classes = config.dataset.num_classes


  def forward(self, datas, targets=None):
    """Predict semantic segmenation and loss.

    Args:
      datas: A dict with an entry `embedding`, which is a 4-D float
        tensor of shape `[batch_size, num_channels, height, width]`.
      targets: A dict with an entry `semantic_label`, which is a 3-D
        long tensor of shape `[batch_size, height, width]`.

    Return:
      A dict of tensors and scalars.
    """
    targets = targets if targets is not None else {}

    # Predict semantic labels.
    semantic_embeddings = datas['embedding']
    semantic_embeddings = (
      semantic_embeddings / torch.norm(semantic_embeddings, dim=1, keepdim=True))
    semantic_logits = self.semantic_classifier(semantic_embeddings)

    # Compute semantic loss.
    semantic_loss, semantic_acc = None, None
    semantic_labels = targets.get('semantic_label', None)
    if semantic_labels is not None:
      # Upscale logits.
      semantic_logits = F.interpolate(
          semantic_logits,
          size=semantic_labels.shape[-2:],
          mode='bilinear')
      semantic_pred = torch.argmax(semantic_logits, dim=1)
      # Rescale labels to the same size as logits.
      #n, h, w = semantic_labels.shape
      #semantic_labels = F.interpolate(
      #    semantic_labels.view(n, 1, h, w).float(),
      #    size=semantic_embeddings.shape[-2:],
      #    mode='nearest')
      semantic_labels = semantic_labels.masked_fill(
          semantic_labels >= self.num_classes, self.ignore_index)
      semantic_labels = semantic_labels.squeeze_(1).long()

      semantic_loss = self.semantic_loss(semantic_logits, semantic_labels)
      semantic_acc = torch.eq(semantic_pred, semantic_labels)
      valid_pixels = torch.ne(semantic_labels,
                              self.ignore_index)
      semantic_acc = torch.masked_select(semantic_acc, valid_pixels).float().mean()
    else:
      semantic_pred = torch.argmax(semantic_logits, dim=1)

    outputs = {'semantic_prediction': semantic_pred,
               'semantic_logit': semantic_logits,
               'sem_ann_loss': semantic_loss,
               'accuracy': semantic_acc,}

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


def softmax_classifier(config):
  """Pixel semantic segmentation model.
  """
  return SoftmaxClassifier(config)
