import numpy as np
import torch as th
from torch.autograd import Variable, grad

from functools import reduce
from operator import mul

def var(data):
  return Variable(data, requires_grad=True, volatile=False)

def buf2var(buf):
  return var(th.from_numpy(buf))

def zeros_var(shape, dtype=None):
  return buf2var(np.zeros(shape, dtype=dtype))

def const_var(data):
  return Variable(data, requires_grad=False, volatile=False)

def const_var_as(v):
  return const_var(th.zeros(v.data.size()).type_as(v.data))

def fix_var(v):
  w = v.detach()
  return w

def clone_var(v):
  w = v.detach()
  w.requires_grad = True
  return w

def duplicate_vars(vars_):
  dup_vars_ = []
  for v in vars_:
    y = th.zeros(v.data.size()).type_as(v.data)
    dup_vars_.append(var(y))
  return dup_vars_

def copy_vars(dst_vars_, src_vars_):
  for w, v in zip(dst_vars_, src_vars_):
    w.data.copy_(v.data)

def average_vars(avg_rate, dst_vars_, src_vars_):
  for w, v in zip(dst_vars_, src_vars_):
    w.data.add_(avg_rate * (v.data - w.data))

def initialize_vars(vars_, init_fns):
  for v, init_fn in zip(vars_, init_fns):
    init_val = init_fn()
    assert v.size() == init_val.size(), "initialize_vars: size mismatch: var: {} init val: {}".format(v.size(), init_val.size())
    v.data.copy_(init_val)

def flat_count_vars(vars_):
  offset = 0
  for v in vars_:
    flat_len = reduce(mul, list(v.data.size()), 1)
    offset += flat_len
  return offset

def deserialize_vars(vars_, src_data):
  offset = 0
  for v in vars_:
    v_dim = v.data.size()
    flat_len = reduce(mul, list(v_dim), 1)
    v.data.resize_(flat_len)
    v.data.copy_(src_data[offset:offset+flat_len])
    v.data.resize_(v_dim)
    offset += flat_len
  return offset

def serialize_vars(vars_, dst_data):
  offset = 0
  for v in vars_:
    v_dim = v.data.size()
    flat_len = reduce(mul, list(v_dim), 1)
    v.data.resize_(flat_len)
    dst_data[offset:offset+flat_len].copy_(v.data)
    v.data.resize_(v_dim)
    offset += flat_len
  return offset

def adjoint(sink_var, vars_, sink_grad=None):
  if sink_grad is None:
    sink_grad = var(th.ones(*sink_var.data.size()).type_as(sink_var.data))
  return grad([sink_var], vars_, grad_outputs=[sink_grad], retain_graph=True, create_graph=True, only_inputs=True)
