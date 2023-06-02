# Python program to move a tensor from CPU to GPU

# when Pytorch comes first, the thing works, else gives a segmentation fault

import numpy as np

import torch

import ctypes

# create a tensor
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("Tensor:", x[1], "\n")

addr = id(x)
print("Tensor CPU mem address", addr, "\n")

a = ctypes.cast(x.storage().data_ptr(), ctypes.py_object).value
