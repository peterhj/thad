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

def zeros_init(dim, dtype=th.FloatTensor):
  def init_fn():
    return th.zeros(*dim).type(dtype)
  return init_fn

def normal_linear_init(kernel_dim, mean, std, dtype=th.FloatTensor):
  def init_fn():
    x = np.random.normal(loc=mean, scale=std, size=kernel_dim).astype(np.float32)
    return th.from_numpy(x).type(dtype)
  return init_fn
