from thad import *

import torch as th

class ParamsBuilder(object):
  def __init__(self):
    self._keys = []
    self._kvs = {}

  def get_var(self, key, init_fn):
    if key not in self._kvs:
      self._keys.append(key)
      self._kvs[key] = (var(init_fn()), init_fn)
    v, _ = self._kvs[key]
    return v

  def build(self):
    param_vars = []
    param_init_fns = []
    for key in reversed(self._keys):
      v, init_fn = self._kvs[key]
      param_vars.append(v)
      param_init_fns.append(init_fn)
    return param_vars, param_init_fns

def const_init(data, dtype=th.FloatTensor):
  def init_fn():
    return th.from_numpy(data).type(dtype)
  return init_fn

def zeros_init(dim, dtype=th.FloatTensor):
  def init_fn():
    return th.zeros(*dim).type(dtype)
  return init_fn

def uniform_init(dim, lo, hi, dtype=th.FloatTensor):
  def init_fn():
    x = np.random.uniform(low=lo, high=hi, size=dim).astype(np.float32)
    return th.from_numpy(x).type(dtype)
  return init_fn

def normal_init(dim, mean, std, dtype=th.FloatTensor):
  def init_fn():
    x = np.random.normal(mean, std, size=dim).astype(np.float32)
    return th.from_numpy(x).type(dtype)
  return init_fn

def orth_normal_init(dim, orth_axis, dtype=th.FloatTensor):
  def init_fn():
    x = np.random.normal(0.0, 1.0, size=dim).astype(np.float32)
    x /= np.linalg.norm(x, axis=orth_axis, keepdims=True)
    return th.from_numpy(x).type(dtype)
  return init_fn

def normal_linear_init(kernel_dim, mean, std, dtype=th.FloatTensor):
  def init_fn():
    x = np.random.normal(loc=mean, scale=std, size=kernel_dim).astype(np.float32)
    return th.from_numpy(x).type(dtype)
  return init_fn

def xavier_linear_init(kernel_dim, dtype=th.FloatTensor):
  def init_fn():
    half_width = np.sqrt(6.0 / float(kernel_dim[0] + kernel_dim[1]))
    x = np.random.uniform(low=-half_width, high=half_width, size=kernel_dim).astype(np.float32)
    return th.from_numpy(x).type(dtype)
  return init_fn
