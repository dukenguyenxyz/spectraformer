import torch
import scipy
import warnings
import numpy as np 

from torch.nn import init
from math import sqrt, log
from scipy.stats import chi 
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F

gelu = lambda x: F.gelu(x) # GELU 
pelu = lambda x: F.elu(x) + 1 # Positive ELU 

# from .hadamard import hadamard_transform_torch, hadamard_transform_cuda
from .hadamard import hadamard_transform

class FeatureMap(nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device, dtype):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner

class FastFoodRandomFeatures(FeatureMap): 
    """
    Random Fastfood features for the RBF kernel according to [1].

    [1]: "Fastfood - Approximating Kernel Expansions in Loglinear Time" 
    by Quoc Le, Tamas Sarlos and Alexander Smola.

    Arguments
    ---------
        query_dimensions: int 
            The input query dimensions.
        softmax_temp: float 
            The temerature for the Gaussian kernel approximation 
            exp(-t * ||x-y||^2). (default: 1/sqrt(query_dimensions))
    """

    def __init__(self, query_dimensions, n_samples=None, softmax_temp=None, 
                 learn_S=False, learn_G_B=False): 
        super(FastFoodRandomFeatures, self).__init__(query_dimensions)

        # Check Fastfood condition 
        if n_samples is not None: 
            if n_samples < query_dimensions: 
                raise RuntimeError(('The dimension of the feature map must be '
                                    'greater or equal than the input dimension.'))

        self.learn_S = learn_S
        self.learn_G_B = learn_G_B
        self.n_samples = n_samples or query_dimensions # Currently ignored! 
        self.softmax_temp = softmax_temp or 1/sqrt(query_dimensions)

        # Declare structured matrices 
        self.P = None 

        if self.learn_G_B:
            self.B = Parameter(torch.Tensor(self.query_dims)) 
            self.G = Parameter(torch.Tensor(self.query_dims)) 
            init.normal_(self.B, std=sqrt(1./self.query_dims))
            init.normal_(self.G, std=sqrt(1./self.query_dims))
        else: 
            self.B = None 
            self.G = None 

        if self.learn_S: 
            self.S = Parameter(torch.Tensor(self.query_dims)) 
            init.normal_(self.S, std=sqrt(1./self.query_dims))
        else: 
            self.S = None 

    def new_feature_map(self, device, dtype): 
        # Permutation matrix P 
        self.P = torch.randperm(
            self.query_dims, 
            device=device 
        )

        if not self.learn_G_B:
            # Binary scaling matrix B 
            self.B = torch.tensor(
                np.random.choice([-1.0, 1.0], 
                    size=self.query_dims
                ),
                dtype=dtype, 
                device=device, 
                requires_grad=True
            )

            # Gaussian scaling matrix G 
            self.G = torch.zeros(
                self.query_dims, 
                dtype=dtype,
                device=device
            )
            self.G.normal_()

        if not self.learn_S: 
            # Scaling matrix S
            self.S = torch.tensor(
                chi.rvs( 
                    df=self.query_dims, 
                    size=self.query_dims
                ), 
                dtype=dtype,
                device=device 
            ) / torch.norm(self.G)
    
    # def get_weight_matrix(self, x):
    #     # Original shape
    #     x_shape = x.shape
    #     # x = x * sqrt(self.softmax_temp)
        
    #     # print("X, SHAPE", x_shape)
    #     # print("QUERY DIM", self.query_dims)
                
    #     # Reshape for Fastfood
    #     # x = x.view(-1, self.query_dims)
    #     x = x.reshape(-1, self.query_dims)

    #     # Fastfood multiplication
    #     Bx = x * self.B
    #     HBx = hadamard_transform(Bx) 
    #     PHBx = HBx[:,self.P]
    #     GPHBx = PHBx * self.G
    #     HGPHBx = hadamard_transform(GPHBx)
    #     SHGPHBx = HGPHBx * self.S

    #     # Normalize and recover original shape
    #     # Vx = (sqrt(1.0/self.query_dims) * SHGPHBx).view(x_shape)
    #     # Vx = SHGPHBx.view(x_shape)
    #     Vx = SHGPHBx.reshape(x_shape)

    #     return Vx

    def get_weight_matrix(self):        
        return self.S * self.G * self.B
        

    def forward(self, x): 
        """
        Compute the FastFood feature map for the given input. 

        Arguments: 
        ----------
        x : (N, L, H, D)
            The input tensor.
        """
        # Original shape
        # X IS ACTUALLY  (N, H, L, D)
        # BUT THIS WONT AFFECT ANYTHING
        # x = torch.swapaxes(x, 1, 2)
        
        x_shape = x.shape
        x = x * sqrt(self.softmax_temp)
        
        # print("X, SHAPE", x_shape)
        # print("QUERY DIM", self.query_dims)
                
        # Reshape for Fastfood
        # x = x.view(-1, self.query_dims)
        x = x.reshape(-1, self.query_dims)

        # Fastfood multiplication
        Bx = x * self.B
        HBx = hadamard_transform(Bx) 
        PHBx = HBx[:,self.P]
        GPHBx = PHBx * self.G
        HGPHBx = hadamard_transform(GPHBx)
        SHGPHBx = HGPHBx * self.S

        # Normalize and recover original shape
        # Vx = (sqrt(1.0/self.query_dims) * SHGPHBx).view(x_shape)
        Vx = (sqrt(1.0/self.query_dims) * SHGPHBx).reshape(x_shape)

        return Vx
        # # Feature map
        # phi = torch.cat([torch.cos(Vx), torch.sin(Vx)], dim=-1)
        # phi = sqrt(1.0/self.query_dims) * phi
        # return 
        
class SmoothedFastFoodRandomFeatures(FastFoodRandomFeatures):
    """Simply add a constant value to the dot product in order to avoid
    possible numerical instabilities when the feature map is slightly
    negative.

    Implements K(x, y) = exp(-|x-y|^2) + s.

    Arguments
    ---------
        query_dimensions: int, 
            The input query dimensions.
        softmax_temp: float 
            The temerature for the Gaussian kernel approximation 
            exp(-t * ||x-y||^2). (default: 1/sqrt(query_dimensions))
        smoothing: float
            The smoothing parameter to add to the dot product.
    """
    def __init__(self, query_dimensions, n_samples=None, softmax_temp=None, 
                 learn_S=False, learn_G_B=False, smoothing=1.0):
        super(SmoothedFastFoodRandomFeatures, self).__init__(
            query_dimensions,
            n_samples=None, # Currently ignored! 
            softmax_temp=softmax_temp, 
            learn_S=learn_S, 
            learn_G_B=learn_G_B
        )
        self.smoothing = smoothing

    def forward(self, x):
        y = super().forward(x)
        smoothing = torch.full(
            y.shape[:-1] + (1,),
            self.smoothing,
            dtype=y.dtype,
            device=y.device
        )
        return torch.cat([y, smoothing], dim=-1)

class FastFoodPositiveFeatures(FastFoodRandomFeatures):
    def __init__(self, query_dimensions, n_samples=None, softmax_temp=None, 
                 learn_S=False, learn_G_B=False):
        super(SmoothedFastFoodRandomFeatures, self).__init__(
            query_dimensions,
            n_samples=None, # Currently ignored! 
            softmax_temp=softmax_temp, 
            learn_S=learn_S, 
            learn_G_B=learn_G_B
        )

    def forward(self, x): 
        """
        Compute the FastFood feature map for the given input. 

        Arguments: 
        ----------
        x : (N, L, H, D)
            The input tensor.
        """        
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)

        # Reshape for Fastfood
        x_shape = x.shape
        # x = x.view(-1, self.query_dims)
        x = x.reshape(-1, self.query_dims)

        # Fastfood multiplication
        Bx = x * self.B
        HBx = hadamard_transform(Bx) 
        PHBx = HBx[:,self.P]
        GPHBx = PHBx * self.G
        HGPHBx = hadamard_transform(GPHBx)
        SHGPHBx = HGPHBx * self.S

        # Normalize and recover original shape
        # Vx = (sqrt(1.0/self.query_dims) * SHGPHBx).view(x_shape)
        Vx = (sqrt(1.0/self.query_dims) * SHGPHBx).reshape(x_shape)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared + 0.5 * log(self.query_dims)

        return torch.exp(Vx - offset)
    
class GenerativeRandomFourierFeatures(FeatureMap): 
    """
    """
    def __init__(self, query_dimensions, noise_dims, 
                 n_dims=None, softmax_temp=None, 
                 redraw=1, deterministic_eval=False): 
        super(GenerativeRandomFourierFeatures, self).__init__(
            query_dimensions
        )

        self.noise_dims = noise_dims
        self.n_dims = n_dims or query_dimensions
        self.softmax_temp = softmax_temp or 1/sqrt(query_dimensions)
        self.redraw = redraw
        self.deterministic_eval = deterministic_eval
        
        # Generator network 
        self.generator = GeneratorBlock(
            noise_dims, 
            query_dimensions, 
            hidden_act='leaky', 
            output_act='tanh'
        )    

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            'omega',
            torch.zeros(self.n_dims//2, self.noise_dims[0])
        )

        # Buffer for storing the counter 
        self.register_buffer(
            '_calls', 
            torch.tensor(-1, dtype=torch.int)
        )

    def new_feature_map(self, device, dtype):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1

        if (self._calls % self.redraw) != 0:
            return

        omega = torch.zeros(
            self.n_dims//2,
            self.noise_dims[0],
            dtype=dtype,
            device=device
        )
        omega.normal_()

        self.register_buffer('omega', omega)

    def forward(self, x): 
        # Run the generator 
        omega = self.generator(self.omega)

        # Scale input 
        x = x * sqrt(self.softmax_temp)

        # Compute feature map 
        u = torch.matmul(
            x.unsqueeze(-2), 
            omega.transpose(0,1)
        ).squeeze(-2)

        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * sqrt(2/self.n_dims)

class SmoothedGenerativeRandomFourierFeatures(GenerativeRandomFourierFeatures):
    """
    """
    def __init__(self, query_dimensions, noise_dims, n_dims=None, 
                 softmax_temp=None, smoothing=1.0, redraw=1, 
                 deterministic_eval=False):
        super(SmoothedGenerativeRandomFourierFeatures, self).__init__(
            query_dimensions, noise_dims=noise_dims,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp, redraw=redraw, 
            deterministic_eval=deterministic_eval
        )
        self.smoothing = smoothing

    def forward(self, x):
        y = super().forward(x)
        smoothing = torch.full(
            y.shape[:-1] + (1,),
            self.smoothing,
            dtype=y.dtype,
            device=y.device
        )
        return torch.cat([y, smoothing], dim=-1)

class GenerativePositiveRandomFeatures(GenerativeRandomFourierFeatures): 
    """
    """
    def __init__(self, query_dimensions, noise_dims, n_dims=None, 
                 softmax_temp=None, stabilize=False, redraw=1, 
                 deterministic_eval=False): 
        super(GenerativePositiveRandomFeatures, self).__init__(
            query_dimensions, noise_dims=noise_dims,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp, redraw=redraw,
            deterministic_eval=deterministic_eval
        )

        self.stabilize = stabilize
        
        # Generator network 
        self.generator = GeneratorBlock(
            noise_dims, 
            query_dimensions, 
            hidden_act='leaky', 
            output_act='tanh'
        )    

    def forward(self, x): 
        # Scale input 
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)

        # Run the generator 
        omega = self.generator(self.omega)

        # Compute feature map 
        u = torch.matmul(
            x.unsqueeze(-2), 
            omega.transpose(0,1)
        ).squeeze(-2)

        # Compute the exponential offset 
        offset = norm_x_squared + 0.5 * log(self.n_dims)

        if self.stabilize:
            self._check_sequence_length(norm_x_squared)
            offset = offset + norm_x_squared.max(1, keepdim=True)[0]

        return torch.exp(u - offset)