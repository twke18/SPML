#!/bin/bash
# This script is used for training, inference and benchmarking
# the baseline method with SPML on PASCAL VOC 2012 using 
# point annotations. Users could also modify from this
# script for their use case.
#
# Usage:
#   # From SPML/ directory.
#   source bashscripts/voc12/train_spml_point.sh
#
#

# Set up parameters for network.
BACKBONE_TYPES=panoptic_deeplab_101
EMBEDDING_DIM=64

# Set up parameters for training.
PREDICTION_TYPES=segsort
TRAIN_SPLIT=train+
GPUS=0,1,2,3
LR_POLICY=poly
USE_SYNCBN=true
SNAPSHOT_STEP=30000
MAX_ITERATION=30000
WARMUP_ITERATION=100
LR=3e-3
WD=5e-4
BATCH_SIZE=4
CROP_SIZE=512
MEMORY_BANK_SIZE=2
KMEANS_ITERATIONS=10
KMEANS_NUM_CLUSTERS=6
SEM_ANN_LOSS_TYPES=segsort # segsort / none
SEM_OCC_LOSS_TYPES=segsort # segsort / none
IMG_SIM_LOSS_TYPES=segsort # segsort / none
FEAT_AFF_LOSS_TYPES=none # segsort / none
SEM_ANN_CONCENTRATION=6
SEM_OCC_CONCENTRATION=8
IMG_SIM_CONCENTRATION=16
FEAT_AFF_CONCENTRATION=0
SEM_ANN_LOSS_WEIGHT=0.1
SEM_OCC_LOSS_WEIGHT=0.1
IMG_SIM_LOSS_WEIGHT=0.1
FEAT_AFF_LOSS_WEIGHT=0.0

# Set up parameters for inference.
INFERENCE_SPLIT=val
INFERENCE_IMAGE_SIZE=512
INFERENCE_CROP_SIZE_H=512
INFERENCE_CROP_SIZE_W=512
INFERENCE_STRIDE=512

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/voc12_point/${BACKBONE_TYPES}_${PREDICTION_TYPES}/p${CROP_SIZE}_dim${EMBEDDING_DIM}_nc${KMEANS_NUM_CLUSTERS}_ki${KMEANS_ITERATIONS}_lr${LR}_syncbn_${USE_SYNCBN}_bs${BATCH_SIZE}_gpu${GPUS}_it${MAX_ITERATION}_wd${WD}_memsize${MEMORY_BANK_SIZE}_losstype${SEM_ANN_LOSS_TYPES}_${SEM_OCC_LOSS_TYPES}_${IMG_SIM_LOSS_TYPES}_${FEAT_AFF_LOSS_TYPES}_lossconc${SEM_ANN_CONCENTRATION}_${SEM_OCC_CONCENTRATION}_${IMG_SIM_CONCENTRATION}_${FEAT_AFF_CONCENTRATION}_lossw${SEM_ANN_LOSS_WEIGHT}_${SEM_OCC_LOSS_WEIGHT}_${IMG_SIM_LOSS_WEIGHT}_${FEAT_AFF_LOSS_WEIGHT}
echo ${SNAPSHOT_DIR}

# Set up the procedure pipeline.
IS_CONFIG_EMB=1
IS_TRAIN_EMB=1
IS_CONFIG_CLASSIFIER=1
IS_ANNOTATION_1=1
IS_TRAIN_CLASSIFIER_1=1
IS_INFERENCE_CLASSIFIER_1=1
IS_BENCHMARK_CLASSIFIER_1=1

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory and file list.
DATAROOT=/home/twke/data/VOCdevkit
PRETRAINED=./snapshots/imagenet/trained/resnet-101-cuhk.pth
TRAIN_DATA_LIST=datasets/voc12/scribble_${TRAIN_SPLIT}_d6_0.0_hed.txt
TEST_DATA_LIST=datasets/voc12/panoptic_${INFERENCE_SPLIT}.txt
MEMORY_DATA_LIST=datasets/voc12/panoptic_${TRAIN_SPLIT}_hed.txt


# Build configuration file for training embedding network.
if [ ${IS_CONFIG_EMB} -eq 1 ]; then
  if [ ! -d ${SNAPSHOT_DIR} ]; then
    mkdir -p ${SNAPSHOT_DIR}
  fi

  sed -e "s/TRAIN_SPLIT/${TRAIN_SPLIT}/g"\
    -e "s/BACKBONE_TYPES/${BACKBONE_TYPES}/g"\
    -e "s/PREDICTION_TYPES/${PREDICTION_TYPES}/g"\
    -e "s/EMBEDDING_MODEL/${EMBEDDING_MODEL}/g"\
    -e "s/PREDICTION_MODEL/${PREDICTION_MODEL}/g"\
    -e "s/EMBEDDING_DIM/${EMBEDDING_DIM}/g"\
    -e "s/GPUS/${GPUS}/g"\
    -e "s/BATCH_SIZE/${BATCH_SIZE}/g"\
    -e "s/LABEL_DIVISOR/2048/g"\
    -e "s/USE_SYNCBN/${USE_SYNCBN}/g"\
    -e "s/LR_POLICY/${LR_POLICY}/g"\
    -e "s/SNAPSHOT_STEP/${SNAPSHOT_STEP}/g"\
    -e "s/MAX_ITERATION/${MAX_ITERATION}/g"\
    -e "s/WARMUP_ITERATION/${WARMUP_ITERATION}/g"\
    -e "s/LR/${LR}/g"\
    -e "s/WD/${WD}/g"\
    -e "s/MEMORY_BANK_SIZE/${MEMORY_BANK_SIZE}/g"\
    -e "s/KMEANS_ITERATIONS/${KMEANS_ITERATIONS}/g"\
    -e "s/KMEANS_NUM_CLUSTERS/${KMEANS_NUM_CLUSTERS}/g"\
    -e "s/TRAIN_CROP_SIZE/${CROP_SIZE}/g"\
    -e "s/TEST_SPLIT/${INFERENCE_SPLIT}/g"\
    -e "s/TEST_IMAGE_SIZE/${INFERENCE_IMAGE_SIZE}/g"\
    -e "s/TEST_CROP_SIZE_H/${INFERENCE_CROP_SIZE_H}/g"\
    -e "s/TEST_CROP_SIZE_W/${INFERENCE_CROP_SIZE_W}/g"\
    -e "s/TEST_STRIDE/${INFERENCE_STRIDE}/g"\
    -e "s#PRETRAINED#${PRETRAINED}#g"\
    -e "s/SEM_ANN_LOSS_TYPES/${SEM_ANN_LOSS_TYPES}/g"\
    -e "s/SEM_OCC_LOSS_TYPES/${SEM_OCC_LOSS_TYPES}/g"\
    -e "s/IMG_SIM_LOSS_TYPES/${IMG_SIM_LOSS_TYPES}/g"\
    -e "s/FEAT_AFF_LOSS_TYPES/${FEAT_AFF_LOSS_TYPES}/g"\
    -e "s/SEM_ANN_CONCENTRATION/${SEM_ANN_CONCENTRATION}/g"\
    -e "s/SEM_OCC_CONCENTRATION/${SEM_OCC_CONCENTRATION}/g"\
    -e "s/IMG_SIM_CONCENTRATION/${IMG_SIM_CONCENTRATION}/g"\
    -e "s/FEAT_AFF_CONCENTRATION/${FEAT_AFF_CONCENTRATION}/g"\
    -e "s/SEM_ANN_LOSS_WEIGHT/${SEM_ANN_LOSS_WEIGHT}/g"\
    -e "s/SEM_OCC_LOSS_WEIGHT/${SEM_OCC_LOSS_WEIGHT}/g"\
    -e "s/IMG_SIM_LOSS_WEIGHT/${IMG_SIM_LOSS_WEIGHT}/g"\
    -e "s/FEAT_AFF_LOSS_WEIGHT/${FEAT_AFF_LOSS_WEIGHT}/g"\
    configs/voc12_template.yaml > ${SNAPSHOT_DIR}/config_emb.yaml

  cat ${SNAPSHOT_DIR}/config_emb.yaml
fi


# Train for the embedding.
if [ ${IS_TRAIN_EMB} -eq 1 ]; then
  python3 pyscripts/train/train.py\
    --data_dir ${DATAROOT}\
    --data_list ${TRAIN_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --cfg_path ${SNAPSHOT_DIR}/config_emb.yaml

  python3 pyscripts/inference/prototype.py\
    --data_dir ${DATAROOT}\
    --data_list ${MEMORY_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${TRAIN_SPLIT}\
    --kmeans_num_clusters 12,12\
    --label_divisor 2048\
    --cfg_path ${SNAPSHOT_DIR}/config_emb.yaml

  python3 pyscripts/inference/inference.py\
    --data_dir ${DATAROOT}\
    --data_list ${TEST_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}\
    --semantic_memory_dir ${SNAPSHOT_DIR}/stage1/results/${TRAIN_SPLIT}/semantic_prototype\
    --kmeans_num_clusters 12,12\
    --label_divisor 2048\
    --cfg_path ${SNAPSHOT_DIR}/config_emb.yaml

  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}/semantic_gray\
    --gt_dir ${DATAROOT}/VOC2012/segcls\
    --num_classes 21

  python3 pyscripts/inference/inference_softmax.py\
    --data_dir ${DATAROOT}\
    --data_list ${TEST_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --save_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}_softmax\
    --cfg_path ${SNAPSHOT_DIR}/config_emb.yaml

  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred_dir ${SNAPSHOT_DIR}/stage1/results/${INFERENCE_SPLIT}_softmax/semantic_gray\
    --gt_dir ${DATAROOT}/VOC2012/segcls\
    --num_classes 21

fi


# Build configuration file for training softmax classifier.
if [ ${IS_CONFIG_CLASSIFIER} -eq 1 ]; then

  sed -e "s/TRAIN_SPLIT/${TRAIN_SPLIT}/g"\
    -e "s/BACKBONE_TYPES/${BACKBONE_TYPES}/g"\
    -e "s/PREDICTION_TYPES/softmax_classifier/g"\
    -e "s/EMBEDDING_MODEL/${EMBEDDING_MODEL}/g"\
    -e "s/PREDICTION_MODEL/softmax_classifier/g"\
    -e "s/EMBEDDING_DIM/${EMBEDDING_DIM}/g"\
    -e "s/GPUS/0/g"\
    -e "s/BATCH_SIZE/16/g"\
    -e "s/LABEL_DIVISOR/2048/g"\
    -e "s/LR_POLICY/${LR_POLICY}/g"\
    -e "s/SNAPSHOT_STEP/${SNAPSHOT_STEP}/g"\
    -e "s/MAX_ITERATION/4000/g"\
    -e "s/WARMUP_ITERATION/${WARMUP_ITERATION}/g"\
    -e "s/LR/${LR}/g"\
    -e "s/WD/${WD}/g"\
    -e "s/KMEANS_ITERATIONS/0/g"\
    -e "s/KMEANS_NUM_CLUSTERS/1/g"\
    -e "s/TRAIN_CROP_SIZE/${CROP_SIZE}/g"\
    -e "s/TEST_SPLIT/${INFERENCE_SPLIT}/g"\
    -e "s/TEST_IMAGE_SIZE/${INFERENCE_IMAGE_SIZE}/g"\
    -e "s/TEST_CROP_SIZE_H/${INFERENCE_CROP_SIZE_H}/g"\
    -e "s/TEST_CROP_SIZE_W/${INFERENCE_CROP_SIZE_W}/g"\
    -e "s/TEST_STRIDE/${INFERENCE_STRIDE}/g"\
    -e "s#PRETRAINED#${SNAPSHOT_DIR}\/stage1\/model-$(($MAX_ITERATION-1)).pth#g"\
    configs/voc12_template.yaml > ${SNAPSHOT_DIR}/config_classifier.yaml

  cat ${SNAPSHOT_DIR}/config_classifier.yaml
fi


# Generate pseudo labels from CAM.
if [ ${IS_ANNOTATION_1} -eq 1 ]; then

  python3 pyscripts/inference/pseudo_softmaxrw_crf.py\
    --data_dir ${DATAROOT}\
    --data_list ${TRAIN_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/stage1\
    --save_dir ${SNAPSHOT_DIR}/stage1/pseudo_labels/${TRAIN_SPLIT}_cam_rw\
    --kmeans_num_clusters 1,1\
    --label_divisor 2048\
    --crf_iter_max 10\
    --crf_pos_w 3\
    --crf_pos_xy_std 1\
    --crf_bi_w 4\
    --crf_bi_xy_std 67\
    --crf_bi_rgb_std 3\
    --cfg_path ${SNAPSHOT_DIR}/config_emb.yaml

  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred_dir ${SNAPSHOT_DIR}/stage1/pseudo_labels/${TRAIN_SPLIT}_cam_rw/semantic_gray\
    --gt_dir ${DATAROOT}/VOC2012/segcls\
    --num_classes 21

  sed -e "s#scribble_annotation\/VOC2012\/dilate_6_0\.0\/segcls#`pwd`\/${SNAPSHOT_DIR}\/stage1\/pseudo_labels\/${TRAIN_SPLIT}_cam_rw\/semantic_gray#g"\
    -e "s#VOC2012#${DATAROOT}\/VOC2012#g"\
    ${TRAIN_DATA_LIST} > ${SNAPSHOT_DIR}/stage1/pseudo_labels/${TRAIN_SPLIT}_cam_rw/list.txt

fi


# Train softmax classifier while fix embedding network.
if [ ${IS_TRAIN_CLASSIFIER_1} -eq 1 ]; then
  python3 pyscripts/train/train_classifier.py\
    --data_dir ${HOME}\
    --data_list ${SNAPSHOT_DIR}/stage1/pseudo_labels/${TRAIN_SPLIT}_cam_rw/list.txt\
    --snapshot_dir ${SNAPSHOT_DIR}/softmax_classifier_stage1\
    --cfg_path ${SNAPSHOT_DIR}/config_classifier.yaml
fi


# Inference. 
if [ ${IS_INFERENCE_CLASSIFIER_1} -eq 1 ]; then
  python3 pyscripts/inference/inference_softmax.py\
    --data_dir ${DATAROOT}\
    --data_list ${TEST_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}/softmax_classifier_stage1\
    --save_dir ${SNAPSHOT_DIR}/softmax_classifier_stage1/results/${INFERENCE_SPLIT}\
    --crf_iter_max 10\
    --crf_pos_w 3\
    --crf_pos_xy_std 1\
    --crf_bi_w 4\
    --crf_bi_xy_std 67\
    --crf_bi_rgb_std 3\
    --cfg_path ${SNAPSHOT_DIR}/config_classifier.yaml
fi


# Benchmark.
if [ ${IS_BENCHMARK_CLASSIFIER_1} -eq 1 ]; then
  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred_dir ${SNAPSHOT_DIR}/softmax_classifier_stage1/results/${INFERENCE_SPLIT}/semantic_gray\
    --gt_dir /home/twke/data/VOCdevkit/VOC2012/segcls\
    --num_classes 21
fi
