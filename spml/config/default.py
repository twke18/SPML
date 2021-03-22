"""Default configuration for SPML."""

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()
config.embedding_model = ''
config.prediction_model = ''
config.gpus = ''
config.num_threads = 4

## Parameters for network.
config.network = edict()
# Backbone network.
config.network.pixel_means = np.array((0.485, 0.456, 0.406))
config.network.pixel_stds = np.array((0.229, 0.224, 0.225))
config.network.pretrained = ''
config.network.use_syncbn = False
config.network.backbone_types = ''
# Spatial Pooling Pyramid module.
config.network.aspp_feature_dim = 512
config.network.pspp_feature_dim = 512
config.network.embedding_dim = 128
config.network.label_divisor = 255
# Spherical KMeans.
config.network.kmeans_iterations = 10
config.network.kmeans_num_clusters = [5, 5]

## Parameters for dataset.
config.dataset = edict()
config.dataset.data_dir = ''
config.dataset.train_data_list = ''
config.dataset.test_data_list = ''
config.dataset.color_map_path = ''
config.dataset.num_classes = 0
config.dataset.semantic_ignore_index = 255

## Parameters for training.
config.train = edict()
# Data processing.
config.train.lr_policy = 'step'
config.train.random_mirror = True
config.train.random_scale = True
config.train.random_crop = True
config.train.shuffle = True
config.train.resume = False
config.train.begin_iteration = 0
config.train.max_iteration = 0
config.train.warmup_iteration = 0
config.train.decay_iterations = [0]
config.train.snapshot_step = 0
config.train.tensorboard_step = 0
config.train.base_lr = 1e-3
config.train.weight_decay = 5e-3
config.train.momentum = 0.9
config.train.batch_size = 0
config.train.crop_size = [0, 0]
config.train.memory_bank_size = 0
# Losses.
config.train.sem_ann_loss_types = 'none'
config.train.sem_occ_loss_types = 'none'
config.train.img_sim_loss_types = 'none'
config.train.feat_aff_loss_types = 'none'
config.train.sem_ann_concentration = 0
config.train.sem_occ_concentration = 0
config.train.img_sim_concentration = 0
config.train.feat_aff_concentration = 0
config.train.sem_ann_loss_weight= 0.0
config.train.sem_occ_loss_weight= 0.0
config.train.img_sim_loss_weight= 0.0
config.train.feat_aff_loss_weight= 0.0

## Parameters for testing.
config.test = edict()
# Data Processing.
config.test.scales = [0]
config.test.image_size = 0
config.test.crop_size = [0, 0]
config.test.stride = [0, 0]


def update_config(config_file):

  exp_config = None
  with open(config_file) as f:
    exp_config = edict(yaml.load(f))
    for k, v in exp_config.items():
      # update default config.
      if k in config:
        if isinstance(v, dict):
          if k == 'train':
            if 'base_lr' in v:
              v['base_lr'] = float(v['base_lr'])
            if 'weight_decay' in v:
              v['weight_decay'] = float(v['weight_decay'])
          for vk, vv in v.items():
            config[k][vk] = vv
        else:
          config[k] = v
      # insert new config.
      else:
        config[k] = v
