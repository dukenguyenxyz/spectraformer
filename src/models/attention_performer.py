# https://github.com/lucidrains/performer-pytorch
# MIT License

# Copyright (c) 2020 Phil Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import math

class PerformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.rp_dim = config["rp_dim"]
        self.kernel_type = config["kernel_type"]        
        self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, feat=config["feat"], kernel_fn = self.kernel_type)


    def forward(self, Q, K, V, mask):
        return self.attn_fn(
            Q / math.sqrt(math.sqrt(self.head_dim)),
            K / math.sqrt(math.sqrt(self.head_dim)) * mask[:, None, :, None],
            V * mask[:, None, :, None])

    def extra_repr(self):
        return f'rp_dim={self.rp_dim}, kernel_type={self.kernel_type}'


################################################################

import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager

from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from performer_pytorch.reversible import ReversibleSequence, SequentialSequence

from models.component_functions import get_rfs_torch, optimize_opt_pos_params_torch, optimize_geom_params_torch, get_params_posrf, get_params_trigrf # oprf_compfunc

from models.weight_matrices_approx import gaussian_matrix, gaussian_orthogonal_random_matrix, scrf_matrix, sorf_matrix, rom_matrix, sgq_matrix, qmc_matrix, mm_matrix
from models.weight_matrices_learner import FastFoodRandomFeatures, SmoothedFastFoodRandomFeatures

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v.type_as(k))
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)
    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out

# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v):
    k_cumsum = k.cumsum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...nde', k, v)
    context = context.cumsum(dim=-3)
    out = torch.einsum('...nde,...nd,...n->...ne', context, q, D_inv)
    return out

def calculate_psi_star(X, Y):
    """
    Calculate the diagonal elements psi_{l,l}^* for SADERFs.
    
    Args:
    - X (torch.Tensor): The tensor containing x(i) vectors, shape (L, d)
    - Y (torch.Tensor): The tensor containing y(j) vectors, shape (L, d)
    
    Returns:
    - torch.Tensor: The tensor containing the diagonal elements psi_{l,l}^*, shape (d,)
    """
    # Squaring each element in X and Y
    X_squared = X.pow(2)
    Y_squared = Y.pow(2)
    
    # Summing over the L dimension (summing each feature across all vectors)
    sum_X_squared = X_squared.sum(dim=2)  # Sum over the rows for each column (feature)
    sum_Y_squared = Y_squared.sum(dim=2)  # Sum over the rows for each column (feature)
    
    # Calculating the psi_{l,l}^* for each feature
    psi_star = (sum_Y_squared / sum_X_squared).pow(1/4)
    
    return torch.diag_embed(psi_star, offset=0)

class FastAttention(nn.Module):
    def __init__(self, dim_heads, n= None, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, no_projection = False, feat = 'orf'):
        # print('attention performer hERE')

        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.feat = feat
        
        comp_functions = {
            'trigrf': get_rfs_torch, # trigrf_compfunc,
            # 'nrff': get_rfs_torch,
            'posrf': get_rfs_torch, # posrf_compfunc,
            'oprf': get_rfs_torch, # oprf_compfunc,
            'gerf': get_rfs_torch,
            'saderf': get_rfs_torch,

        }
        
        self.optimize_params = {
            'trigrf': get_params_trigrf,
            'posrf': get_params_posrf,
            'oprf': optimize_opt_pos_params_torch,
        }
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Component function is callable
        if callable(kernel_fn):
            pass
        elif isinstance(kernel_fn, str) and kernel_fn in comp_functions.keys():
            self.rf_type = kernel_fn
            self.get_query = True
            kernel_fn = comp_functions[kernel_fn]
        else:
            raise Exception(f'Kernel function {kernel_fn} is of invalid type {type(kernel_fn)}')
        self.kernel_fn = kernel_fn

        if self.rf_type in ['gerf', 'saderf']:
            self.gerf_params = None

        if feat in ['gaus', 'orf', 'scrf', 'sorf', 'rom', 'sgq', 'qmc', 'mm']:
            self.non_matrx_linear_op = False
            if feat =='gaus':
                self.create_projection = partial(gaussian_matrix, nb_rows = self.nb_features, nb_columns = dim_heads)
            elif feat == 'orf':
                self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling, qr_uniform_q = qr_uniform_q)
            elif feat == 'scrf':
                self.create_projection = partial(scrf_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
            elif feat == 'sorf':
                self.create_projection = partial(sorf_matrix, nb_rows = self.nb_features, nb_columns = dim_heads)
            elif feat == 'rom':
                self.create_projection = partial(rom_matrix, nb_rows = self.nb_features, nb_columns = dim_heads)
            elif feat == 'sgq':
                # GQ is bad so we won't use it cus slow
                self.create_projection = partial(sgq_matrix, nb_rows = self.nb_features, nb_columns = dim_heads)
            elif feat == 'qmc':
                self.create_projection = partial(qmc_matrix, nb_rows = self.nb_features, nb_columns = dim_heads)
            elif feat == 'mm':
                self.create_projection = partial(mm_matrix, nb_rows = self.nb_features, nb_columns = dim_heads)
            # else:
            #     raise Exception(f"Feat type '{feat}' is not valid") 
            
            projection_matrix = self.create_projection()
            self.register_buffer('projection_matrix', projection_matrix)
        elif feat in ['fastfood_learnable', 'fastfood_fixed']:
            self.non_matrx_linear_op = True
            if feat == 'fastfood_fixed':
                self.projection_matrix_learner = SmoothedFastFoodRandomFeatures(dim_heads, learn_S=False, learn_G_B=False)
            else:
                self.projection_matrix_learner = SmoothedFastFoodRandomFeatures(dim_heads, learn_S=True, learn_G_B=True)

            self.projection_matrix_learner.new_feature_map(device, torch.float64)
            
        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    def set_gerf_params(self, q):
        b, h, j, d = q.shape
        self.gerf_params = (nn.Parameter(torch.zeros(h,j, device=q.device).uniform_(-0.124999, 0.124999)), 1)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:            
            if self.rf_type in ['gerf', 'saderf']:
                if not self.gerf_params:
                    self.set_gerf_params(q)
                rf_params = self.gerf_params
            else:
                rf_params = self.optimize_params[self.rf_type](q, k)
            
            if self.rf_type == 'saderf':
                psi = calculate_psi_star(q, k)
            else:
                psi = None
            
            if self.non_matrx_linear_op:
                create_kernel = partial(self.kernel_fn, rf_type = self.rf_type, query_dim = q.shape[-1], non_matrx_linear_op=True, d = self.dim_heads, projection_matrix = self.projection_matrix_learner, rf_params=rf_params, device = device, psi=psi)
            else:
                create_kernel = partial(self.kernel_fn, rf_type = self.rf_type, query_dim = q.shape[-1], non_matrx_linear_op=False, d = self.dim_heads, projection_matrix = self.projection_matrix, rf_params=rf_params, device = device, psi=psi)

            # poisrf or geomrf
            if self.rf_type.endswith('+'):
                c = torch.minimum(torch.min(q, dim=0)[0], torch.min(k, dim=0)[0]) - 1e-8
                q -= c
                k -= c

            if self.get_query:
                q = create_kernel(q, is_query = True)
                k = create_kernel(k, is_query = False)
            else:
                q = create_kernel(q)
                k = create_kernel(k)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)        

        return out