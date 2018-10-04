from thad import *

import numpy as np
import torch

class AdamOptimizer(object):
  def __init__(self, dim, dtype=torch.FloatTensor):
    self.dim = dim
    self.grad = torch.zeros(dim).type(dtype)
    self.param = torch.zeros(dim).type(dtype)
    self.gmean = torch.zeros(dim).type(dtype)
    self.gvar = torch.zeros(dim).type(dtype)
    self.iter_ct = 0

  def step(self, params, grads, grad_scale=None, l2_reg=None, grad_l2_clip=None, step_size=0.001, beta1=0.9, beta2=0.999, epsilon=1.0e-8):
    assert self.iter_ct >= 0
    norm_1 = float(1.0 / (1.0 - np.power(beta1, self.iter_ct + 1)))
    norm_2 = float(1.0 / (1.0 - np.power(beta2, self.iter_ct + 1)))

    assert self.dim == serialize_vars(grads, self.grad)
    assert self.dim == serialize_vars(params, self.param)

    if grad_scale is not None:
      self.grad.mul_(grad_scale)
    if l2_reg is not None:
      self.grad.add_(l2_reg * self.param)
    if grad_l2_clip is not None:
      g = torch.norm(self.grad)
      w = np.minimum(grad_clip, g) / g
      self.grad.mul_(w)

    self.gmean.add_((1.0 - beta1) * (self.grad - self.gmean))
    self.gvar.add_((1.0 - beta2) * (torch.mul(self.grad, self.grad) - self.gvar))
    self.param.add_(-step_size * (self.gmean * norm_1) / ((self.gvar * norm_2).sqrt() + epsilon))

    assert self.dim == deserialize_vars(params, self.param)

    self.iter_ct += 1
