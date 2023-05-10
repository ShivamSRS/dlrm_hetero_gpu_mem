# Python program to move a tensor from CPU to GPU

# when Pytorch comes first, the thing works, else gives a segmentation fault

import numpy as np

import torch

import ctypes

# create a tensor
x = torch.tensor([1.0,2.0,3.0,4.0])
print("Tensor:", x[1] , "\n")

addr = id(x)
print("Tensor CPU mem address", addr , "\n")

a = ctypes.cast(addr, ctypes.py_object).value

# print(type(a))
# print("Value fetched using memory address - ", a)

# check tensor device (cpu/cuda)
print("Tensor device:", x.device, "\n")

# Move tensor from CPU to GPU
# check CUDA GPU is available or not
print("CUDA GPU:", torch.cuda.is_available(), "\n")
if torch.cuda.is_available():
   x = x.to("cuda:0")
   # or x=x.to("cuda")
print(x, "\n")

addr = id(x)
print("tensor GPU mem address", addr, "\n")

a = ctypes.cast(addr, ctypes.py_object).value
print("Value fetched using memory address - ", a)

# now check the tensor device
print("Tensor device:", x.device, "\n")


### variable declaration

# start with 100 zeroes
arr = np.zeros((100,)).astype(int)

# change 25 of them to 1s
arr[:25] = 1

# shuffle the array to put all the elements in random positions
# np.random.shuffle(arr)

# reshape to final desired shape
arr = arr.reshape((10,10))
  
# display variable
print("Actual value -", arr)
  
# get the memory address of the python object 
# for variable
x = id(arr[1][1])

# display memory address
print("Memory address - ", x)
  
# get the value through memory address
a = ctypes.cast(x, ctypes.py_object).value
  
# display
print("Value fetched using memory address - ", a)

# Datatypes
print("Arr type - " , type(arr))
print("id arr[1][1] type - " , type(x))
print("ctype cast datatype - " , type(a))





##############################################################################
#### GPU to CPU and vice versa #######################
##############################################################################

# print("Memory address - " , type(a))
# print("Memory address - " , type(a))
# print("Memory address - " , type(a))




# print(arr[1][1])

# print(id(arr[1][1]))
# print(hex(id(arr[1][1])))

# print(id(arr))
# print(hex(id(arr)))

# numpy IsADirectoryErro
# memory address ka tensor
# non zero elements ki memory addresses ka ek tensor
# tensor ko gpu me daal
# cuda tensor jop GPU me hau BUT 