"""Utility functions for all.
"""
# This code is borrowed and re-implemented from
# https://github.com/jyhjinghwang/SegSort/blob/master/network/segsort/common_utils.py

import torch

import spml.utils.general.common as common_utils


def calculate_prototypes_from_labels(embeddings,
                                     labels,
                                     max_label=None):
  """Calculates prototypes from labels.

  This function calculates prototypes (mean direction) from embedding
  features for each label. This function is also used as the m-step in
  k-means clustering.

  Args:
    embeddings: A 2-D or 4-D float tensor with feature embedding in the
      last dimension (embedding_dim).
    labels: An N-D long label map for each embedding pixel.
    max_label: The maximum value of the label map. Calculated on-the-fly
      if not specified.

  Returns:
    A 2-D float tensor with shape `[num_prototypes, embedding_dim]`.
  """
  embeddings = embeddings.view(-1, embeddings.shape[-1])

  if max_label is None:
    max_label = labels.max() + 1
  prototypes = torch.zeros((max_label, embeddings.shape[-1]),
                           dtype=embeddings.dtype,
                           device=embeddings.device)
  labels = labels.view(-1, 1).expand(-1, embeddings.shape[-1])
  prototypes = prototypes.scatter_add_(0, labels, embeddings)
  prototypes = common_utils.normalize_embedding(prototypes)

  return prototypes


def find_nearest_prototypes(embeddings, prototypes):
  """Finds the nearest prototype for each embedding pixel.

  This function calculates the index of nearest prototype for each
  embedding pixel. This function is also used as the e-step in k-means
  clustering.

  Args:
    embeddings: An N-D float tensor with embedding features in the last
      dimension (embedding_dim).
    prototypes: A 2-D float tensor with shape
      `[num_prototypes, embedding_dim]`.

  Returns:
    A 1-D long tensor with length `[num_pixels]` containing the index
    of the nearest prototype for each pixel.
  """
  embeddings = embeddings.view(-1, prototypes.shape[-1])
  similarities = torch.mm(embeddings, prototypes.t())

  return torch.argmax(similarities, 1)


def kmeans_with_initial_labels(embeddings,
                               initial_labels,
                               max_label=None,
                               iterations=10):
  """Performs the von-Mises Fisher k-means clustering with initial
  labels.

  Args:
    embeddings: A 2-D float tensor with shape
      `[num_pixels, embedding_dim]`.
    initial_labels: A 1-D long tensor with length [num_pixels].
      K-means clustering will start with this cluster labels if
      provided.
    max_label: An integer for the maximum of labels.
    iterations: Number of iterations for the k-means clustering.

  Returns:
    A 1-D long tensor of the cluster label for each pixel.
  """
  if max_label is None:
    max_label = initial_labels.max() + 1

  labels = initial_labels
  for _ in range(iterations):
    # M-step of the vMF k-means clustering.
    prototypes = calculate_prototypes_from_labels(
        embeddings, labels, max_label)
    # E-step of the vMF k-means clustering.
    labels = find_nearest_prototypes(embeddings, prototypes)

  return labels


def kmeans(embeddings, num_clusters, iterations=10):
  """Performs the von-Mises Fisher k-means clustering.

  Args:
    embeddings: A 4-D float tensor with shape
      `[batch, height, width, embedding_dim]`.
    num_clusters: A list of 2 integers for number of clusters in
      y and x axes.
    iterations: Number of iterations for the k-means clustering.

  Returns:
    A 3-D long tensor of the cluster label for each pixel with shape
    `[batch, height, width]`.
  """
  shape = embeddings.shape
  labels = initialize_cluster_labels(
      num_clusters, [shape[1], shape[2]])

  embeddings = embeddings.view(-1, shape[3])
  labels = labels.view(-1)

  labels = kmeans_with_initial_labels(
      embeddings, labels, iterations=iterations)

  labels = labels.view(shape[0], shape[1], shape[2])

  return labels


def initialize_cluster_labels(num_clusters,
                              img_dimensions,
                              device):
  """Initializes uniform cluster labels for an image.

  This function is used to initialize cluster labels that uniformly
  partition a 2-D image.

  Args:
    num_clusters: A list of 2 integers for number of clusters in y
      and x axes.
    img_dimensions: A list of 2 integers for image's y and x dimension.

  Returns:
    A 2-D long tensor with shape specified by img_dimension.
  """
  y_labels = torch.linspace(
      0, num_clusters[0] - 1, img_dimensions[0], device=device).round_().long()
  x_labels = torch.linspace(
      0, num_clusters[1] - 1, img_dimensions[1], device=device).round_().long()
  y_labels = y_labels.view(-1, 1)
  x_labels = x_labels.view(1, -1)
  labels = y_labels + (y_labels.max() + 1) * x_labels

  return labels


def generate_location_features(img_dimensions,
                               device,
                               feature_type='int'):
  """Calculates location features for an image.

  This function generates location features for an image. The 2-D
  location features range from -1 to 1 for y and x axes each.

  Args:
    img_dimensions: A list of 2 integers for image's y and x dimension.
    feature_type: The data type of location features, integer or float.

  Returns:
    A 3-D float/long tensor with shape
    `[img_dimension[0], img_dimension[1], 2]`.

  Raises:
    ValueError: Type of location features is neither 'int' nor 'float'.
  """
  if feature_type == 'int':
    y_features = torch.arange(img_dimensions[0], device=device)
    x_features = torch.arange(img_dimensions[1], device=device)
  elif feature_type == 'float':
    y_features = torch.linspace(0, 1, img_dimensions[0], device=device)
    x_features = torch.linspace(0, 1, img_dimensions[1], device=device)
  else:
    raise ValueError(
        'Type of location features should be either int or float.')

  y_features, x_features = torch.meshgrid(y_features, x_features)

  location_features = torch.stack([y_features, x_features], dim=2)

  return location_features


def prepare_prototype_labels(semantic_labels,
                             instance_labels,
                             offset=256):
  """Prepares prototype labels from semantic and instance labels.

  This function generates unique prototype labels from semantic and
  instance labels. Note that instance labels sometimes can be cluster
  labels.

  Args:
    semantic_labels: A 1-D long tensor for semantic labels.
    instance_labels: A 1-D long tensor for instance labels.
    offset: An integer for instance offset.

  Returns:
    prototype_labels: A 1-D long tensor for the semantic labels of
      prototypes with length as the number of unique instances.
    unique_instance_labels: A 1-D long tensor for unique instance
      labels with the same length as the input semantic labels.
  """
  panoptic_labels = semantic_labels + instance_labels * offset
  prototype_panoptic_labels, unique_instance_labels = torch.unique(
      panoptic_labels, return_inverse=True)

  prototype_semantic_labels = prototype_panoptic_labels % offset

  return prototype_semantic_labels, unique_instance_labels


def find_majority_label_index(semantic_labels, cluster_labels):
  """Finds indices of pixels that belong to their majority
  label in a cluster.

  Args:
    semantic_labels: An N-D long tensor for semantic labels.
    cluster_labels: An N-D long tensor for cluster labels.

  Returns:
    select_pixel_indices: An 2-D long tensor for indices of pixels
      that belong to their majority label in a cluster.
    majority_semantic_labels: A 1-D long tensor for the semantic
      label for each cluster with length `[num_clusters]`.
  """
  semantic_labels = semantic_labels.view(-1)
  cluster_labels = cluster_labels.view(-1)
  num_clusters = cluster_labels.max() + 1
  num_classes = semantic_labels.max() + 1

  #one_hot_semantic_labels = common_utils.one_hot(
  #    semantic_labels, semantic_labels.max() + 1).float()
  #one_hot_cluster_labels = common_utils.one_hot(
  #    cluster_labels, cluster_labels.max() + 1).float()

  #accumulate_semantic_labels = torch.mm(one_hot_cluster_labels.t(),
  #                                      one_hot_semantic_labels)
  one_hot_semantic_labels = common_utils.one_hot(
      semantic_labels, num_classes)
  accumulate_semantic_labels = torch.zeros(
      (num_clusters, num_classes),
      dtype=torch.long,
      device=semantic_labels.device)
  accumulate_semantic_labels = accumulate_semantic_labels.scatter_add_(
      0,
      cluster_labels.view(-1, 1).expand(-1, num_classes),
      one_hot_semantic_labels)
  majority_semantic_labels = torch.argmax(accumulate_semantic_labels, 1)

  cluster_semantic_labels = torch.gather(
      majority_semantic_labels,
      0,
      cluster_labels)
  select_pixel_indices = torch.eq(cluster_semantic_labels,
                                  semantic_labels)
  select_pixel_indices = select_pixel_indices.nonzero()

  return select_pixel_indices, majority_semantic_labels


def segment_by_kmeans(embeddings,
                      labels=None,
                      num_clusters=[5,5],
                      cluster_indices=None,
                      local_features=None,
                      ignore_index=None,
                      iterations=10):
  """Segment image into prototypes by Spherical KMeans Clustering.

  This function conducts Spherical KMeans Clustering within
  each image.

  Args:
    embeddings: A 4-D float tensor of shape
      `[batch_size, num_channels, height, width]`.
    num_clusters: A list of two integers indicate number of cluster
      for height and width.
    kmeans_iterations: An integer indicates number of iteration for
      kmeans clustering.
    label_divisor: An integer indicates the offset between semantic
      and instance labels.
    labels: A 3-D long tensor of shape
      `[batch_size, height, width]`.
    cluster_indices: A 3-D long tensor of shape
      `[batch_size, height, width]`.
    location_features: A 4-D float tensor of shape
      `[batch_size, height, width, 2]`.
    ignore_index: An integer denotes index of ignored class.

  Returns:
    prototypes: A 2-D float tensor of shape `[num_prototypes, embedding_dim]`.
    prototype_panoptic_labels: A 1-D long tensor.
    prototype_batch_labels: A 1-D long tensor.
    cluster_labels: A 1-D long tensor.
  """
  # Convert embeddings from NCHW to NHWC.
  embeddings = embeddings.permute(0, 2, 3, 1).contiguous()
  N, H, W, C = embeddings.shape

  # L-2 normalize the embeddings.
  embeddings = common_utils.normalize_embedding(embeddings)

  # Generate location features.
  if local_features is None:
    local_features = generate_location_features(
        (H, W), device=embeddings.device, feature_type='float')
    local_features -= 0.5
    local_features = local_features.view(1, H, W, 2).expand(N, H, W, 2)

  # Create initial cluster labels.
  if cluster_indices is None:
    cluster_indices = initialize_cluster_labels(
        num_clusters, (H, W), device=embeddings.device)
    cluster_indices = cluster_indices.view(1, H, W).expand(N, H, W)

  # Extract semantic and instance labels from panoptic labels.
  if labels is None:
    labels = torch.zeros((N, H, W),
                         dtype=torch.long,
                         device=embeddings.device)

  # Perform KMeans clustering per image.
  gathered_datas = {'labels': [],
                    'cluster_indices': [],
                    'batch_indices': [],
                    'embeddings': [],
                    'embeddings_with_loc': []}
  for batch_index in range(N):
    # Prepare datas for each image.
    cur_labels = labels[batch_index].view(-1)
    cur_cluster_indices = cluster_indices[batch_index].view(-1)
    _, cur_cluster_indices = torch.unique(
        cur_cluster_indices, return_inverse=True)

    cur_num_clusters = cur_cluster_indices.max() + 1

    cur_embeddings = embeddings[batch_index].view(-1, C)
    cur_local_features = (
      local_features[batch_index].view(-1, local_features.shape[-1]))
    cur_embeddings_with_loc = torch.cat(
        [cur_embeddings, cur_local_features], -1)
    cur_embeddings_with_loc = common_utils.normalize_embedding(
        cur_embeddings_with_loc)

    # Remove ignore label.
    if ignore_index is not None:
      valid_pixel_indices = torch.ne(cur_labels, ignore_index)
      valid_pixel_indices = valid_pixel_indices.nonzero().view(-1)
      cur_labels = torch.index_select(
          cur_labels, 0, valid_pixel_indices)
      cur_cluster_indices = torch.index_select(
          cur_cluster_indices, 0, valid_pixel_indices)
      cur_embeddings = torch.index_select(
          cur_embeddings, 0, valid_pixel_indices)
      cur_embeddings_with_loc = torch.index_select(
          cur_embeddings_with_loc, 0, valid_pixel_indices)

    # KMeans clustering.
    if cur_embeddings.shape[0] > 0:
      cur_cluster_indices = kmeans_with_initial_labels(
          cur_embeddings_with_loc,
          cur_cluster_indices,
          cur_num_clusters,
          iterations)

    # Small hack to solve issue of batch index for multi-gpu.
    gpu_id = cur_cluster_indices.device.index
    _batch_index = batch_index + (N * gpu_id)

    # Add offset to labels to separate different images.
    cur_batch_indices = torch.zeros_like(cur_cluster_indices)
    cur_batch_indices.fill_(_batch_index)

    # Gather from each image.
    gathered_datas['labels'].append(cur_labels)
    gathered_datas['cluster_indices'].append(cur_cluster_indices)
    gathered_datas['batch_indices'].append(cur_batch_indices)
    gathered_datas['embeddings'].append(cur_embeddings)
    gathered_datas['embeddings_with_loc'].append(cur_embeddings_with_loc)

  # Concat results from each images.
  labels = torch.cat(gathered_datas['labels'], 0)
  embeddings = torch.cat(gathered_datas['embeddings'], 0)
  embeddings_with_loc = torch.cat(gathered_datas['embeddings_with_loc'], 0)
  cluster_indices = torch.cat(gathered_datas['cluster_indices'], 0)
  batch_indices = torch.cat(gathered_datas['batch_indices'], 0)

  # Partition segments by image.
  lab_div = cluster_indices.max() + 1
  cluster_indices = batch_indices * lab_div + cluster_indices
  _, cluster_indices = torch.unique(
      cluster_indices, return_inverse=True)

  # Partition segments by ground-truth labels.
  _, cluster_indices = prepare_prototype_labels(
        labels, cluster_indices, labels.max() + 1)

  return embeddings, embeddings_with_loc,\
         labels, cluster_indices, batch_indices



