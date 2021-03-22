"""Utility functions for training.
"""

import torch
import torch.nn.functional as F


def lr_poly(base_lr, curr_iter, max_iter, warmup_iter=0, power=0.9):
  """Polynomial-decay learning rate policy.

  Args:
    base_lr: A scalar indicates initial learning rate.
    curr_iter: A scalar indicates current iteration.
    max_iter: A scalar indicates maximum iteration.
    warmup_iter: A scalar indicates the number of iteration
      before which the learning rate is not adjusted.
    power: A scalar indicates the decay power.

  Return:
    A scalar indicates the current adjusted learning rate.
  """
  if curr_iter < warmup_iter:
    alpha = curr_iter / warmup_iter

    return min(base_lr * (1 / 10.0 * (1 - alpha) + alpha),
               base_lr * ((1 - float(curr_iter) / max_iter)**(power)))
  return base_lr * ((1 - float(curr_iter) / max_iter)**(power))


def get_step_index(curr_iter, decay_iters):
  """Get step when the learning rate is decayed.
  """
  for idx, decay_iter in enumerate(decay_iters):
    if curr_iter < decay_iter:
      return idx
  return len(decay_iters)


def lr_step(base_lr, curr_iter, decay_iters, warmup_iter=0):
  """Stepwise exponential-decay learning rate policy.

  Args:
    base_lr: A scalar indicates initial learning rate.
    curr_iter: A scalar indicates current iteration.
    decay_iter: A list of scalars indicates the numbers of
      iteration when the learning rate is decayed.
    warmup_iter: A scalar indicates the number of iteration
      before which the learning rate is not adjusted.

  Return:
    A scalar indicates the current adjusted learning rate.
  """
  if curr_iter < warmup_iter:
    alpha = curr_iter / warmup_iter
    return base_lr * (1 / 10.0 * (1 - alpha) + alpha)
  else:
    return base_lr * (0.1 ** get_step_index(curr_iter, decay_iters))
