"""Script for training pixel-wise embeddings by pixel-segment
contrastive learning loss.
"""

from __future__ import print_function, division
import os

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.parallel.scatter_gather as scatter_gather
import tensorboardX
from tqdm import tqdm

from lib.nn.parallel.data_parallel import DataParallel
from lib.nn.optimizer import SGD
from lib.nn.sync_batchnorm.batchnorm import convert_model
from lib.nn.sync_batchnorm.replicate import patch_replication_callback
from spml.config.default import config
from spml.config.parse_args import parse_args
import spml.utils.general.train as train_utils
import spml.utils.general.vis as vis_utils
import spml.utils.general.others as other_utils
import spml.models.utils as model_utils
from spml.data.datasets.list_tag_dataset import ListTagDataset
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab
#from spml.models.predictions.segsort import segsort
from spml.models.predictions.segsort_softmax import segsort
from spml.models.predictions.softmax_classifier import softmax_classifier

torch.cuda.manual_seed_all(235)
torch.manual_seed(235)

cudnn.enabled = True
cudnn.benchmark = True


def main():
  """Training for pixel-wise embeddings by pixel-segment
  contrastive learning loss.
  """
  # Retreve experiment configurations.
  args = parse_args('Training for pixel-wise embeddings.')

  # Retrieve GPU informations.
  device_ids = [int(i) for i in config.gpus.split(',')]
  gpu_ids = [torch.device('cuda', i) for i in device_ids]
  num_gpus = len(gpu_ids)

  # Create logger and tensorboard writer.
  summary_writer = tensorboardX.SummaryWriter(logdir=args.snapshot_dir)
  color_map = vis_utils.load_color_map(config.dataset.color_map_path)

  model_path_template = os.path.join(args.snapshot_dir,
                                     'model-{:d}.pth')
  optimizer_path_template = os.path.join(args.snapshot_dir,
                                         'model-{:d}.state.pth')

  # Create data loaders.
  train_dataset = ListTagDataset(
      data_dir=args.data_dir,
      data_list=args.data_list,
      img_mean=config.network.pixel_means,
      img_std=config.network.pixel_stds,
      size=config.train.crop_size,
      random_crop=config.train.random_crop,
      random_scale=config.train.random_scale,
      random_mirror=config.train.random_mirror,
      training=True)

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=config.train.batch_size,
      shuffle=config.train.shuffle,
      num_workers=num_gpus * config.num_threads,
      drop_last=False,
      collate_fn=train_dataset.collate_fn)

  # Create models.
  if config.network.backbone_types == 'panoptic_pspnet_101':
    embedding_model = resnet_101_pspnet(config).cuda()
  elif config.network.backbone_types == 'panoptic_deeplab_101':
    embedding_model = resnet_101_deeplab(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.backbone_types)

  if config.network.prediction_types == 'segsort':
    prediction_model = segsort(config).cuda()
  elif config.network.prediction_types == 'softmax_classifier':
    prediction_model = softmax_classifier(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.prediction_types)

  # Use synchronize batchnorm.
  if config.network.use_syncbn:
    embedding_model = convert_model(embedding_model).cuda()
    prediction_model = convert_model(prediction_model).cuda()
      
  # Use customized optimizer and pass lr=1 to support different lr for
  # different weights.
  optimizer = SGD(
      embedding_model.get_params_lr() + prediction_model.get_params_lr(),
      lr=1,
      momentum=config.train.momentum,
      weight_decay=config.train.weight_decay)
  optimizer.zero_grad()

  # Load pre-trained weights.
  curr_iter = config.train.begin_iteration
  if config.train.resume:
    model_path = model_path_template.fromat(curr_iter)
    print('Resume training from {:s}'.format(model_path))
    embedding_model.load_state_dict(
        torch.load(model_path)['embedding_model'],
        resume=True)
    prediction_model.load_state_dict(
        torch.load(model_path)['prediction_model'],
        resume=True)
    optimizer.load_state_dict(torch.load(
        optimizer_path_template.format(curr_iter)))
  elif config.network.pretrained:
    print('Loading pre-trained model: {:s}'.format(config.network.pretrained))
    embedding_model.load_state_dict(torch.load(config.network.pretrained))
  else:
    print('Training from scratch')

  # Distribute model weights to multi-gpus.
  embedding_model = DataParallel(embedding_model,
                                 device_ids=device_ids,
                                 gather_output=False)
  prediction_model = DataParallel(prediction_model,
                                  device_ids=device_ids,
                                  gather_output=False)
  if config.network.use_syncbn:
    patch_replication_callback(embedding_model)
    patch_replication_callback(prediction_model)

  for module in embedding_model.modules():
    if isinstance(module, _BatchNorm) or isinstance(module, _ConvNd):
      print(module.training, module)
  print(embedding_model)
  print(prediction_model)

  # Create memory bank.
  memory_banks = {}

  # start training
  train_iterator = train_loader.__iter__()
  iterator_index = 0
  pbar = tqdm(range(curr_iter, config.train.max_iteration))
  for curr_iter in pbar:
    # Check if the rest of datas is enough to iterate through;
    # otherwise, re-initiate the data iterator.
    if iterator_index + num_gpus >= len(train_loader):
        train_iterator = train_loader.__iter__()
        iterator_index = 0

    # Feed-forward.
    image_batch, label_batch = other_utils.prepare_datas_and_labels_mgpu(
        train_iterator, gpu_ids)
    iterator_index += num_gpus

    # Generate embeddings, clustering and prototypes.
    embeddings = embedding_model(*zip(image_batch, label_batch))

    # Synchronize cluster indices and computer prototypes.
    c_inds = [emb['cluster_index'] for emb in embeddings]
    cb_inds = [emb['cluster_batch_index'] for emb in embeddings]
    cs_labs = [emb['cluster_semantic_label'] for emb in embeddings]
    ci_labs = [emb['cluster_instance_label'] for emb in embeddings]
    c_embs = [emb['cluster_embedding'] for emb in embeddings]
    c_embs_with_loc = [emb['cluster_embedding_with_loc']
                         for emb in embeddings]
    (prototypes, prototypes_with_loc,
     prototype_semantic_labels, prototype_instance_labels,
     prototype_batch_indices, cluster_indices) = (
      model_utils.gather_clustering_and_update_prototypes(
          c_embs, c_embs_with_loc,
          c_inds, cb_inds,
          cs_labs, ci_labs,
          'cuda:{:d}'.format(num_gpus-1)))

    for i in range(len(label_batch)):
      label_batch[i]['prototype'] = prototypes[i]
      label_batch[i]['prototype_with_loc'] = prototypes_with_loc[i]
      label_batch[i]['prototype_semantic_label'] = prototype_semantic_labels[i]
      label_batch[i]['prototype_instance_label'] = prototype_instance_labels[i]
      label_batch[i]['prototype_batch_index'] = prototype_batch_indices[i]
      embeddings[i]['cluster_index'] = cluster_indices[i]

    semantic_tags = model_utils.gather_and_update_datas(
        [lab['semantic_tag'] for lab in label_batch],
        'cuda:{:d}'.format(num_gpus-1))
    for i in range(len(label_batch)):
      label_batch[i]['semantic_tag'] = semantic_tags[i]
      label_batch[i]['prototype_semantic_tag'] = torch.index_select(
          semantic_tags[i],
          0,
          label_batch[i]['prototype_batch_index'])

    # Add memory bank to label batch.
    for k in memory_banks.keys():
      for i in range(len(label_batch)):
        assert(label_batch[i].get(k, None) is None)
        label_batch[i][k] = [m.to(gpu_ids[i]) for m in memory_banks[k]]

    # Compute loss.
    outputs = prediction_model(*zip(embeddings, label_batch))
    outputs = scatter_gather.gather(outputs, gpu_ids[0])
    losses = []
    for k in ['sem_ann_loss', 'sem_occ_loss', 'img_sim_loss', 'feat_aff_loss']:
      loss = outputs.get(k, None)
      if loss is not None:
        outputs[k] = loss.mean()
        losses.append(outputs[k])
    loss = sum(losses)
    acc = outputs['accuracy'].mean()

    # Write to tensorboard summary.
    writer = (summary_writer if curr_iter % config.train.tensorboard_step == 0
               else None)
    if writer is not None:
      summary_vis = []
      summary_val = {}
      # Gather labels to cpu.
      cpu_label_batch = scatter_gather.gather(label_batch, -1)
      summary_vis.append(vis_utils.convert_label_to_color(
          cpu_label_batch['semantic_label'], color_map))
      summary_vis.append(vis_utils.convert_label_to_color(
          cpu_label_batch['instance_label'], color_map))

      # Gather outputs to cpu.
      vis_names = ['embedding']
      cpu_embeddings = scatter_gather.gather(
          [{k: emb.get(k, None) for k in vis_names} for emb in embeddings],
          -1)
      for vis_name in vis_names:
        if cpu_embeddings.get(vis_name, None) is not None:
          summary_vis.append(vis_utils.embedding_to_rgb(
              cpu_embeddings[vis_name], 'pca'))

      val_names = ['sem_ann_loss', 'sem_occ_loss',
                   'img_sim_loss', 'feat_aff_loss',
                   'accuracy']
      for val_name in val_names:
        if outputs.get(val_name, None) is not None:
          summary_val[val_name] = outputs[val_name].mean().to('cpu')

      vis_utils.write_image_to_tensorboard(summary_writer,
                                           summary_vis,
                                           summary_vis[-1].shape[-2:],
                                           curr_iter)
      vis_utils.write_scalars_to_tensorboard(summary_writer,
                                             summary_val,
                                             curr_iter)

    # Backward propogation.
    if config.train.lr_policy == 'step':
      lr = train_utils.lr_step(config.train.base_lr,
                               curr_iter,
                               config.train.decay_iterations,
                               config.train.warmup_iteration)
    else:
      lr = train_utils.lr_poly(config.train.base_lr,
                               curr_iter,
                               config.train.max_iteration,
                               config.train.warmup_iteration)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step(lr)

    # Update memory banks.
    with torch.no_grad():
      for k in label_batch[0].keys():
        if 'prototype' in k and 'memory' not in k:
          memory = label_batch[0][k].clone().detach()
          memory_key = 'memory_' + k
          if memory_key not in memory_banks.keys():
            memory_banks[memory_key] = []
          memory_banks[memory_key].append(memory)
          if len(memory_banks[memory_key]) > config.train.memory_bank_size:
            memory_banks[memory_key] = memory_banks[memory_key][1:]

      # Update batch labels.
      for k in ['memory_prototype_batch_index']:
        memory_labels = memory_banks.get(k, None)
        if memory_labels is not None:
          for i, memory_label in enumerate(memory_labels):
            memory_labels[i] += config.train.batch_size * num_gpus

    # Snapshot the trained model.
    if ((curr_iter+1) % config.train.snapshot_step == 0
         or curr_iter == config.train.max_iteration - 1):
      model_state_dict = {
        'embedding_model': embedding_model.module.state_dict(),
        'prediction_model': prediction_model.module.state_dict()}
      torch.save(model_state_dict,
                 model_path_template.format(curr_iter))
      torch.save(optimizer.state_dict(),
                 optimizer_path_template.format(curr_iter))

    # Print loss in the progress bar.
    line = 'loss = {:.3f}, acc = {:.3f}, lr = {:.6f}'.format(
        loss.item(), acc.item(), lr)
    pbar.set_description(line)


if __name__ == '__main__':
  main()
