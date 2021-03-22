import os
import argparse

from PIL import Image
import numpy as np


def parse_args():

  parser = argparse.ArgumentParser(
    description='Benchmark segmentation predictions'
  )
  parser.add_argument('--pred_dir', type=str, default='',
                      help='/path/to/prediction.')
  parser.add_argument('--gt_dir', type=str, default='',
                      help='/path/to/ground-truths')
  parser.add_argument('--inst_dir', type=str, default='',
                      help='/path/to/instance-mask')
  parser.add_argument('--num_classes', type=int, default=21,
                      help='number of segmentation classes')
  parser.add_argument('--string_replace', type=str, default=',',
                      help='replace the first string with the second one')

  return parser.parse_args()


def iou_stats(pred, target, num_classes=21, background=0):
  """Computes statistics of true positive (TP), false negative (FN) and
  false positive (FP).

  Args:
    pred: A numpy array.
    target: A numpy array which should be in the same size as pred.
    num_classes: A number indicating the number of valid classes.
    background: A number indicating the class index of the back ground.

  Returns:
    Three num_classes-D vector indicating the statistics of (TP+FN), (TP+FP)
    and TP across each class.
  """
  # Set redundant classes to background.
  locs = np.logical_and(target > -1, target < num_classes)

  # true positive + false negative
  tp_fn, _ = np.histogram(target[locs],
                          bins=np.arange(num_classes+1))
  # true positive + false positive
  tp_fp, _ = np.histogram(pred[locs],
                          bins=np.arange(num_classes+1))
  # true positive
  tp_locs = np.logical_and(locs, pred == target)
  tp, _ = np.histogram(target[tp_locs],
                       bins=np.arange(num_classes+1))

  return tp_fn, tp_fp, tp


def main():

  args = parse_args()


  assert(os.path.isdir(args.pred_dir))
  assert(os.path.isdir(args.gt_dir))
  print(args.pred_dir)
  iou = np.zeros(args.num_classes, dtype=np.float64)
  ninst = np.zeros(args.num_classes, dtype=np.float64)
  for dirpath, dirnames, filenames in os.walk(args.pred_dir):
    for filename in filenames:
      predname = os.path.join(dirpath, filename)
      gtname = predname.replace(args.pred_dir, args.gt_dir)
      instname = predname.replace(args.pred_dir, args.inst_dir)
      if args.string_replace != '':
        stra, strb = args.string_replace.split(',')
        gtname = gtname.replace(stra, strb)
        instname = instname.replace(stra, strb)

      pred = np.asarray(
          Image.open(predname).convert(mode='L'),
          dtype=np.uint8)
      gt = np.asarray(
          Image.open(gtname).convert(mode='L'),
          dtype=np.uint8)
      inst = np.asarray(
          Image.open(instname).convert(mode='P'),
          dtype=np.uint8)

      # Compute true-positive, false-positive
      # and false-negative
      _tp_fn, _tp_fp, _tp = iou_stats(
          pred,
          gt,
          num_classes=args.num_classes,
          background=0)

      # Compute num. of instances per class
      inst_inds = np.unique(inst)
      ninst_ = np.zeros(args.num_classes, dtype=np.float64)
      for i in range(inst_inds.size):
        if i < 255:
          inst_ind = inst_inds[i]
          inst_mask = inst == inst_ind
          seg_mask = gt[inst_mask]
          npixel, _ = np.histogram(
              seg_mask, bins=args.num_classes,
              range=(0, args.num_classes-1)) # num. pixel per class of this inst.
          cls = np.argmax(npixel)
          ninst_[cls] += 1
                                

      iou_ = _tp/(_tp_fn + _tp_fp - _tp + 1e-12)
      iou += iou_*ninst_
      ninst += ninst_

  iou /= ninst+1e-12
  iou *= 100

  if args.num_classes == 15:
    # MSCOCO-Densepose
    class_names = ['Background', 'Torso', 'R. Hand', 'L. Hand',
                   'L. Foot', 'R. Foot', 'R. Thigh', 'L. Thigh',
                   'R. Leg', 'L. Leg', 'L. Arm' ,'R. Arm',
                   'L. Forearm', 'R. Forearm', 'Head']
  elif args.num_classes == 21:
    # VOC12
    class_names = ['Background', 'Aero', 'Bike', 'Bird', 'Boat',
                   'Bottle', 'Bus', 'Car', 'Cat', 'Chair','Cow',
                   'Table', 'Dog', 'Horse' ,'MBike', 'Person',
                   'Plant', 'Sheep', 'Sofa', 'Train', 'TV']
  else:
    raise NotImplementedError()


  for i in range(args.num_classes):
    if i >= len(class_names):
      break
    print('class {:10s}: {:02d}, acc: {:4.4f}%'.format(
        class_names[i], i, iou[i]))
  mean_iou = iou.sum() / args.num_classes
  print('mean IOU: {:4.4f}%'.format(mean_iou))

if __name__ == '__main__':
  main()
