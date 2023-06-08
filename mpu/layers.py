# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch import nn
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility


def _initialize_affine_weight(weight, output_size, input_size,
                              per_partition_size, partition_dim, init_method,
                              stride=1, return_master_weight=False,device_i=None):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        if init_method is None:
            init_method = init.xavier_normal_
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=weight.dtype,
                                requires_grad=False)
    if device_i is not None:
        master_weight = master_weight.to(torch.device(device_i))
    if init_method is None:
        init_method = init.xavier_normal_

    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = per_partition_size#divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]
    if len(my_weight_list) != 0:
        with torch.no_grad():
            print(f"my_weight: {my_weight_list}")
            print(f"my_weight length: {len(my_weight_list)}")
            print(f"my_weight_out: {weight}")
            torch.cat(my_weight_list, dim=partition_dim, out=weight)
        if return_master_weight:
            return master_weight
        return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_model_parallel_rank(),
                get_model_parallel_world_size())
        self.num_embeddings_per_partition = self.vocab_end_index - \
                                            self.vocab_start_index

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition,
                                             self.embedding_dim))
        self.weight.model_parallel = True
        # And initialize.
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_):
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class PipeParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_,
                 keep_master_weight_for_test=True):
        super(PipeParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        print("going inside embedding")
        self.embedding_dim_per_partition = self.embedding_dim

        #if torch.distributed.get_rank() == 0:  # Store on GPU 0, if process rank is 0
        #self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim_per_partition).cuda(0))
        self.weight = torch.Tensor(self.num_embeddings, self.embedding_dim_per_partition)
        self.weight = self.weight.to(torch.device('cuda:0'))

        self.weight.model_parallel = False
        #else:
        #    self.register_parameter('weight', None)
            #self.weight.model_parallel = False
        #print(torch.cuda.memory_summary())
        print(f"weight: {self.weight}")
        #print(f"embed_dim: {self.embedding_dim_per_partition}")
        #print(f"num_embed: {self.num_embeddings}")
        #print(f"embed_dim: {self.embedding_dim}")



        if torch.distributed.get_rank() == 0:  # Initialize on GPU 0, if process rank is 0
            self.master_weight= _initialize_affine_weight(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.embedding_dim_per_partition, 1, init_method,
                stride=1, return_master_weight=keep_master_weight_for_test,device_i='cuda:0')
            print(f"PipeParallelEmbedding weight parallel: {self.master_weight}")

        print("going outside embedding")

    def forward(self, input_):
        input_ = input_.to(torch.device('cuda:0'))  # Move input to GPU 0
        output = F.embedding(input_, self.weight,
                             self.padding_idx, self.max_norm,
                             self.norm_type, self.scale_grad_by_freq,
                             self.sparse)
        output = output.to(torch.device('cuda:1'))  # Move output to GPU 1
        return output

class Linear_GPU_to_GPU(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=True, from_gpu=0,to_gpu=1,master_weight=None):
        super(Linear_GPU_to_GPU, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        self.weight = Parameter(torch.Tensor(self.output_size,
                                             self.input_size))
        self.weight.model_parallel = False
        print("going inside linear 1")
        #if torch.distributed.get_rank() == from_gpu:
        self.weight = Parameter(torch.Tensor(self.weight).cuda(from_gpu))
        if master_weight is not None:
            master_weight = Parameter(torch.Tensor(master_weight).cuda(from_gpu))
            self.weight = master_weight
            print(f"GPUGPU Linear weight early: {self.weight}")
        #else:
        #    self.register_parameter('weight', None)
        if bias: #and torch.distributed.get_rank() == from_gpu:
            self.bias = Parameter(torch.Tensor(self.output_size)).cuda(from_gpu)
            self.bias.model_parallel = False
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        if torch.distributed.get_rank() == from_gpu  and len(self.weight.size()) > 0:
            print(f"gpu GPU weight: {self.weight}")
            # Initialize weight.
            self.master_weight = _initialize_affine_weight(
                self.weight, self.output_size, self.input_size,
                self.output_size, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test,device_i='cuda:0')
            print(f"gpu GPU master weight: {self.master_weight}")
        else:
            self.master_weight = self.weight
        print("going out of linear 1")

    def forward(self, input_,from_gpu=0,to_gpu=1):
        input_ = input_.to(torch.device(f"cuda:{from_gpu}"))  # Move input to GPU 0
        self.weight = torch.nn.Parameter(self.weight.to(torch.device(f"cuda:{from_gpu}")))#Parameter(torch.Tensor(self.weight)).cuda(from_gpu)#self.weight.to(torch.device(f"cuda:{from_gpu}"))
        self.bias = torch.nn.Parameter(self.bias.to(torch.device(f"cuda:{from_gpu}")))#Parameter(torch.Tensor(self.bias)).cuda(from_gpu)#self.bias.to(torch.device(f"cuda:{from_gpu}"))
        output = F.linear(input_, self.weight, self.bias)
        output = output.to(torch.device(f"cuda:{from_gpu}"))
        #output = output.to(torch.device(f"cuda:{to_gpu}"))  # Move output to GPU 1
        return output

class Linear_GPU_to_CPU(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=True,from_gpu=1,master_weight=None):
        super(Linear_GPU_to_CPU, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        self.weight = Parameter(torch.Tensor(self.output_size,
                                             self.input_size))
        self.weight.model_parallel = False
        print("going inside linear 2")
        #if torch.distributed.get_rank() == from_gpu:
        self.weight = Parameter(torch.Tensor(self.weight).cuda(from_gpu))
        if master_weight is not None:
            master_weight = Parameter(torch.Tensor(master_weight).cuda(from_gpu))
            self.weight = master_weight
            print(f"gpucpu weight early: {self.weight}")
        #else:
            #self.register_parameter('weight', None)
        if bias:# and torch.distributed.get_rank() == from_gpu:
            self.bias = Parameter(torch.Tensor(self.output_size)).cuda(from_gpu)
            self.bias.model_parallel = False
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        if torch.distributed.get_rank() == from_gpu and len(self.weight.size()) > 0:
        # Initialize weight.
            print(f"gpucpu weight: {self.weight}")
            self.master_weight = _initialize_affine_weight(
                self.weight, self.output_size, self.input_size,
                self.output_size, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test,device_i='cuda:1')
            print(f"gpu CPUUU master weight: {self.master_weight}")

        print("going out of linear 2")


    def forward(self, input_,from_gpu=1):
        input_ = input_.to(torch.device(f"cuda:{from_gpu}"))  # Move input to GPU 1
        self.weight = torch.nn.Parameter(self.weight.to(torch.device(f"cuda:{from_gpu}")))#self.weight.to(
            #torch.device(f"cuda:{from_gpu}"))  # Parameter(torch.Tensor(self.weight)).cuda(from_gpu)
        self.bias = torch.nn.Parameter(self.bias.to(torch.device(f"cuda:{from_gpu}")))#self.bias.to(torch.device(f"cuda:{from_gpu}"))  # Parameter(torch.Tensor(self.bias)).cuda(from_gpu)
        output = F.linear(input_, self.weight.t(), self.bias.expand(input_.size(0), self.weight.t().size(1)))
        output = output.to(torch.device('cpu'))  # Move output to GPU 1
        return output

class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_,
                 keep_master_weight_for_test=False):
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set some detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        world_size = get_model_parallel_world_size()
        self.embedding_dim_per_partition = divide(self.embedding_dim,
                                                  world_size)

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim_per_partition))
        self.weight.model_parallel = True
        # And initialize.
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.embedding_dim_per_partition, 1, init_method,
            stride=1, return_master_weight=False)

    def forward(self, input_):
        # print("INSDIE PARALL EMBEDDING shape of input before",input_.shape)
        input_parallel = copy_to_model_parallel_region(input_)
        # print("INSDIE PARALL EMBEDDING shape of input after",input_.shape)
        output_parallel = F.embedding(input_parallel, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # print("INSDIE PARALL EMBEDDING shape of op prll b4 gather",input_.shape)
        output = gather_from_model_parallel_region(output_parallel)
        # print("INSDIE PARALL EMBEDDING shape of op prll after gather",input_.shape)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition,
                                             self.input_size))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            self.bias.model_parallel = True
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.output_size_per_partition, 0, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.
        # print("INSDIE COLMPRLL b4 cpying",input_.shape)
        input_parallel = copy_to_model_parallel_region(input_)
        # print("INSDIE COLMPRLL after cpying",input_parallel.shape)
        # exit()
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size,
                                             self.input_size_per_partition))
        self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.input_size_per_partition, 1, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        # Set up backprop all-reduce.

        if self.input_is_parallel:
            # print("input is prll",input_.shape)
            input_parallel = input_
        else:
            # print("make it parallel",input_.shape)
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        # print("matrix mul in rowparallel",input_parallel.shape,self.weight.shape)
        output_parallel = F.linear(input_parallel, self.weight)
        # print("now all reduce ",output_parallel.shape)
        # exit()
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        # print("all reduce is done",output_.shape)
        # exit()
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


class ParallelMLP(torch.nn.Module):
    """MLP for XLNet.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    # changed init method from a compulosory to optional arg
    def __init__(self, hidden_size, output_dropout_prob, init_method=None,
                 output_layer_init_method=None):
        super(ParallelMLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4 * hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)
        self.relu = nn.ReLU(inplace=True)
        # print("h to 4h and 4h to h parallel MLP")

    def forward(self, hidden_states):
        # [b, s, 4hp]
        # print("going to inter layer h to 4h",hidden_states.shape)
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # print("out of inter layer h to 4h")
        ####added dropout here  and removed dropout on output changed act to relu
        # print("going to relu")
        intermediate_parallel = self.relu(intermediate_parallel)
        # print("out of relu")
        # print("going to dropout")
        intermediate_parallel = self.dropout(intermediate_parallel)
        # print("out of dropout going to layer  4h to h")
        # [b, s, h]
        # exit()
        output = self.dense_4h_to_h(intermediate_parallel)

        # print("##############","out of inter layer 4h to h","#############",sep="\n")
        return output
