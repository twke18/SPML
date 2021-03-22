"""Utility scripts for models.
"""

import torch
import torch.nn.functional as F
import torch.nn.parallel.scatter_gather as scatter_gather

import spml.utils.segsort.common as segsort_common
import spml.utils.general.common as common_utils


def get_params(model, prefixs, suffixes, exclude=None):
  """Retrieve all the trainable parameters in the model.

  Args:
    model: A Torch.Module object indicates the training model.
    prefixs: A list of strings indicates the layer prefixes.
    suffixes: A list of strings indicates included parameter names.
    exclude: A list of strings indicates excluded parameter names.

  Return:
    An enumerator of retrieved parameters.
  """
  for name, module in model.named_modules():
    for prefix in prefixs:
      if name == prefix:
        for n, p in module.named_parameters():
          n = '.'.join([name, n])
          if type(exclude) == list and n in exclude:
            continue
          if type(exclude) == str and exclude in n:
            continue

          for suffix in suffixes:
            if ((n.split('.')[-1].startswith(suffix) or n.endswith(suffix))
                 and p.requires_grad):
              yield p
        break


def gather_clustering_and_update_prototypes(embeddings,
                                            embeddings_with_loc,
                                            cluster_indices,
                                            batch_indices,
                                            semantic_labels,
                                            instance_labels,
                                            anchor_device=None):
  """Gather per-image clustering results from each gpus, and
  calculate segment prototypes and labels. Then, segment prototypes
  and labels are distributed to each gpu.

  Args:
    embeddings: A list of float tensors indicate pixel-wise embeddings.
    embeddings_with_loc: A list of float tensors indicate pixel-wise
      embeddings with location coordinates.
    cluster_indices: A list of 1-D long tensors indicate per-image 
      KMeans clustering indices.
    batch_indices: A list of 1-D long tensors indicate the batch index
      of input images.
    semantic_labels: A list of long tensor indicate the semantic labels.
    instance_labels: A list of long tensor indicate the instance labels.
    anchor_device: An integer indicates the GPU index to handle gathering.

  Returns:
    prototypes: A list of 2-D float tensors indicate the segment 
      prototypes, which are duplicated on each GPU.
    prototypes_with_loc: A list of 2-D float tensors indicate the
      segment prototypes and location coordinates, which are duplicated
      on each GPU.
    prototype_semantic_labels: A list of 1-D long tensors indicate the
      segment semantic labels, which are duplicated on each GPU.
    prototype_instance_labels: A list of 1-D long tensors indicate the
      segment instance labels, which are duplicated on each GPU.
    prototype_batch_indices: A list of 1-D long tensors indicate the
      segment batch indices, which are duplicated on each GPU.
    update_cluster_indices: A list of 1-D long tensors indicate the 
      calibrated per-image clustering indices, which are duplicated
      on each GPU.
  """
  split_sections = [c_ind.shape[0] for c_ind in cluster_indices]
  devices = [c_ind.device for c_ind in cluster_indices]
  if anchor_device is None:
    anchor_device = devices[0]

  # Gather all datas from each gpu to the specified one.
  embeddings = scatter_gather.gather(embeddings, anchor_device)
  embeddings_with_loc = scatter_gather.gather(embeddings_with_loc,
                                              anchor_device)
  cluster_indices = scatter_gather.gather(cluster_indices, anchor_device)
  batch_indices = scatter_gather.gather(batch_indices, anchor_device)
  semantic_labels = scatter_gather.gather(semantic_labels, anchor_device)
  instance_labels = scatter_gather.gather(instance_labels, anchor_device)

  # Remap cluster indices by batch index.
  divisor = cluster_indices.max() + 1
  cluster_indices = batch_indices * divisor + cluster_indices
  _, cluster_indices = torch.unique(cluster_indices, return_inverse=True)

  # Combine semantic, instance and batch labels.
  lab_div = max([instance_labels.max() + 1, semantic_labels.max() + 1])
  labels = (batch_indices * lab_div**2
              + semantic_labels * lab_div
              + instance_labels)

  # Compute prototypes and labels.
  prototype_labels, updated_cluster_indices = (
    segsort_common.prepare_prototype_labels(
        labels, cluster_indices, labels.max() + 1))
  prototype_batch_indices = prototype_labels // lab_div**2
  prototype_semantic_labels = (prototype_labels  % lab_div**2) // lab_div
  prototype_instance_labels = prototype_labels % lab_div

  prototypes = segsort_common.calculate_prototypes_from_labels(
      embeddings, updated_cluster_indices)
  prototypes_with_loc = segsort_common.calculate_prototypes_from_labels(
      embeddings_with_loc, updated_cluster_indices)

  # Re-allocate to each gpu.
  updated_cluster_indices = torch.split(updated_cluster_indices,
                                        split_sections)
  updated_cluster_indices = [
    c_ind.to(d) for c_ind, d in zip(updated_cluster_indices, devices)]
  prototypes = [prototypes.to(d) for d in devices]
  prototypes_with_loc = [prototypes_with_loc.to(d) for d in devices]
  prototype_semantic_labels = [prototype_semantic_labels.to(d) for d in devices]
  prototype_instance_labels = [prototype_instance_labels.to(d) for d in devices]
  prototype_batch_indices = [prototype_batch_indices.to(d) for d in devices]

  return prototypes, prototypes_with_loc,\
         prototype_semantic_labels, prototype_instance_labels,\
         prototype_batch_indices, updated_cluster_indices


def gather_and_update_datas(datas, anchor_device=None):
  """Gather datas from each gpus, and distribute to each gpu.

  Args:
    datas: A list of float tensors.
    anchor_device: An integer indicates the GPU index to handle gathering.

  Return:
    A list of tensors, which are duplicated on each GPU.
  """
  devices = [d.device for d in datas]
  if anchor_device is None:
    anchor_device = devices[0]

  # Gather all datas from each gpu to the first one.
  datas = scatter_gather.gather(datas, anchor_device)

  # Re-allocate to each gpu.
  datas = [datas.to(d) for d in devices]

  return datas


def gather_multiset_labels_per_batch_by_nearest_neighbor(
    embeddings,
    prototypes,
    semantic_prototype_labels,
    batch_embedding_labels,
    batch_prototype_labels,
    num_classes=21,
    top_k=3,
    threshold=0.95,
    label_divisor=255):
  """Assigned labels for unlabelled pixels by nearest-neighbor
  labeled segments, which is useful in feature affinity regularization.

  Args:
    embeddings: A float tensor indicates pixel-wise embeddings, whose
      last dimension denotes feature channels.
    prototypes: A 2-D float tensor indicates segment prototypes
      of shape `[num_segments, channels]`.
    semantic_prototype_labels: A 1-D float tensor indicates segment
      semantic labels of shape `[num_segments]`.
    batch_embedding_labels: A 1-D long tensor indicates pixel-wise
      batch indices, which should include the same number of pixels
      as `embeddings`.
    batch_prototype_labels: A 1-D long tensor indicates segment
      batch indices, which should include the same number of segments
      as `prototypes`.
    num_classes: An integer indicates the number of semantic categories.
    top_k: An integer indicates top-K retrievals.
    threshold: A float indicates the confidence threshold.
    label_divisor: An integer indicates the ignored label index.

  Return:
    A 2-D long tensor of shape `[num_pixels, num_classes]`. If entry i's
    value is 1, the nearest-neighbor segment is of category i.
  """

  embeddings = embeddings.view(-1, embeddings.shape[-1])
  prototypes = prototypes.view(-1, embeddings.shape[-1])
  N, C = embeddings.shape

  # Compute distance and retrieve nearest neighbors.
  batch_affinity = torch.eq(batch_embedding_labels.view(-1, 1),
                            batch_prototype_labels.view(1, -1))
  valid_prototypes = (semantic_prototype_labels < num_classes).view(1, -1)
  label_affinity = batch_affinity & valid_prototypes

  dists = torch.mm(embeddings, prototypes.t())
  min_dist = dists.min()
  dists = torch.where(label_affinity, dists, min_dist - 1)
  nn_dists, nn_inds = torch.topk(dists, top_k, dim=1)
  setsemantic_labels = torch.gather(
      semantic_prototype_labels.view(1, -1).expand(N, -1),
      1, nn_inds)
  setsemantic_labels = setsemantic_labels.masked_fill(
      nn_dists < threshold, num_classes)
  setbatch_labels = torch.gather(
      batch_prototype_labels.view(1, -1).expand(N, -1),
      1, nn_inds)

  # Remove ignored cluster embeddings.
  setsemantic_labels_2d = common_utils.one_hot(
      setsemantic_labels, num_classes+1)
  setsemantic_labels_2d = torch.sum(setsemantic_labels_2d, dim=1)
  setsemantic_labels_2d = (setsemantic_labels_2d > 0).long()
  setsemantic_labels_2d = setsemantic_labels_2d[:, :num_classes]

  return setsemantic_labels_2d
