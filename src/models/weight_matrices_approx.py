import torch
import math
import chaospy
import scipy
from scipy.stats import qmc
import numpy as np
from .hadamard import hadamard_transform
from torch.distributions import multivariate_normal

def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
    # MARK: change cols (float 64.0) to int(cols)
    unstructured_block = torch.randn((int(cols), int(cols)), device = device)
    q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()

def gaussian_matrix(nb_rows, nb_columns):
    generator = multivariate_normal.MultivariateNormal(torch.zeros(nb_columns), torch.eye(nb_columns))
    return torch.stack([generator.sample() for _ in range(nb_rows)])

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q[:int(remaining_rows)])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, int(nb_columns)), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

def scrf_matrix(nb_rows, nb_columns, scaling=0, device=None):
    k = round(nb_columns / nb_rows)
    W = torch.zeros((nb_rows, nb_columns), dtype=torch.float, device=device)
    for i in range(k):
        w = torch.randn((nb_rows, 1), dtype=torch.float, device=device)
        F = torch.fft.fft(torch.eye(nb_rows, dtype=torch.cfloat, device=device))  # Use torch.eye and fft
        A = F.conj().transpose(0, 1)  # Transpose for complex conjugate
        B = torch.diag(torch.fft.fft(w, dim=0).squeeze())  # FFT along the right dimension and create diagonal
        C = 1/nb_rows * A @ B @ F

        # Insert the real part of C into W, handling the slicing to place each block
        W[:, (i * nb_rows):((i + 1) * nb_rows)] = C.real
    # Ensure W has exactly D columns and return the real part
    W = W[:, :nb_columns]

    if scaling == 0:
        multiplier = torch.randn((nb_rows, int(nb_columns)), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    return W

def sorf_matrix(nb_rows, nb_columns, device=None, scaling=True):
    # s = row, d = column
    n = nb_columns
    if nb_rows > nb_columns:
        n = nb_rows

    Ds = 2 * (torch.rand(n, 3, device=device) < 0.5).float() - 1
    
    hd0 = hadamard_transform(torch.diag(Ds[:, 0], diagonal=0))
    hd1 = hadamard_transform(torch.diag(Ds[:, 1], diagonal=0))   
    hd2 = hadamard_transform(torch.diag(Ds[:, 2], diagonal=0))

    matrx = hd0 @ hd1 @ hd2
    if scaling:
        matrx = torch.sqrt(torch.tensor(nb_columns, device=device)) * matrx

    choice = np.random.choice(list(range(n)), nb_rows, replace=False)    
    return matrx[choice, :nb_columns]

# this is specifically the S-Hybrid one
def rom_matrix(nb_rows, nb_columns, device=None, scaling=True):
    # https://proceedings.neurips.cc/paper_files/paper/2017/file/bf8229696f7a3bb4700cfddef19fa23f-Paper.pdf
    # s = row, d = column
    n = nb_columns
    if nb_rows > nb_columns:
        n = nb_rows
    Ds = 2 * (torch.rand(n, 2, device=device) < 0.5).float() - 1
    unif_circle_real = torch.zeros(n, device=device, dtype=torch.float).uniform_(0, math.pi)
    unif_circle_img = torch.zeros(n, device=device, dtype=torch.float).uniform_(0, math.pi)
    unif_circle = unif_circle_real + 1j * unif_circle_img

    sdu = hadamard_transform(torch.diag(unif_circle, diagonal=0))
    sdr1 = hadamard_transform(torch.diag(Ds[:, 0], diagonal=0))   
    sdr2 = hadamard_transform(torch.diag(Ds[:, 1], diagonal=0))

    matrx = sdu @ sdr1.type(torch.complex64) @ sdr2.type(torch.complex64)
    if scaling:
        matrx = torch.sqrt(torch.tensor(nb_columns, device=device)) * matrx

    choice = np.random.choice(list(range(n)), nb_rows, replace=False)    
    return matrx[choice, :nb_columns]

def gq_matrix(nb_rows, nb_columns, gamma=1, deg=8, device=None):
    # raise Exception('SGQ not yet supported')
    d = nb_rows
    D = nb_columns
    points, weights = np.polynomial.hermite.hermgauss(deg)
    points = torch.tensor(points, dtype=torch.float, device=device)
    weights = torch.tensor(weights, device=device)
    points = points * math.sqrt(2) / gamma
    weights = weights / torch.sqrt(torch.tensor(np.pi, device=device))
    
    rand_tensor = torch.rand(D, d, device=device)
    cumulative_weights = torch.cat((torch.tensor([0.0], device=device), torch.cumsum(weights, dim=0)))
    
    # discretize
    # This function discretizes the input tensor using the cumulative weights
    def discretize(input_tensor, bins):
        discretized = torch.zeros_like(input_tensor, dtype=torch.float, device=device)
        for i in range(1, len(bins)):
            mask = (input_tensor >= bins[i - 1]) & (input_tensor < bins[i])
            discretized += mask * (i - 1)
        return discretized
    
    W = discretize(rand_tensor, cumulative_weights)

    for k in range(torch.min(W).to(torch.int).item(), torch.max(W).to(torch.int).item() + 1):
        W[(W==k)] = points[k]
    return W.T

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

from torch.quasirandom import SobolEngine
from torch import distributions

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def qmc_matrix(nb_rows, nb_columns, seq='sobol', device=None):
    # d : dim of X
    dimension = next_power_of_2(nb_rows)

    if seq=='sobol':
        sampler = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=None)
        p = sampler.draw(nb_columns)
        
    else:
        raise Exception(f'{seq} is an invalid method. Only Sobol is supported for torch')
    n = distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    W = n.icdf(p).T[:, :nb_columns].to(torch.float)
    return W

def mm_matrix(nb_rows, nb_columns, gamma=1, device=None):
    # Step 2: Draw nb_rows nb_columns-dimensional uniform samples
    u = torch.rand(nb_rows, nb_columns, device=device)

    # Step 3: Generate nb_rows nb_columns-dimensional normal samples by the inverse transform
    # Assuming u is uniform [0, 1], use the inverse CDF (quantile function) of the normal distribution

    # these two methods are equivalent
    # Method 1
    # n = distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    # w = (gamma**(-1)) * n.icdf(u)

    # Method 2
    w = (gamma**(-1)) * torch.erfinv(2 * u - 1) * torch.sqrt(torch.tensor(2, device=device))
    
    # Step 4: Compute the sample mean and the square root of the sample covariance matrix
    w_mean = w.mean(dim=0)
    w_centered = w - w_mean
    covariance = w_centered.t().mm(w_centered) / nb_rows
    # covariance = torch.cov(w_centered)
    
    A = torch.linalg.cholesky(covariance + 1e-6 * torch.eye(nb_columns, device=device))  # Adding a small value for numerical stability

    # Step 5: Generate the truly uncorrelated standard normal samples by moment matching
    A_inv = torch.linalg.inv(A)
    w_hat = A_inv.mm(w_centered.t()).t()

    # Step 6: Generate the desired samples
    w_tilde = gamma * w_hat

    return w_tilde[:nb_rows,:nb_columns]

def sgq_matrix(nb_rows, nb_columns, gamma=1, deg=8, device=None):
    # https://chaospy.readthedocs.io/en/master/user_guide/fundamentals/quadrature_integration.html
    # https://people.math.sc.edu/Burkardt/py_src/sparse_grid/sparse_grid.html
    # https://github.com/rncarpio/py_tsg?tab=readme-ov-file
    # https://pypi.org/project/smolyak/
    # https://github.com/mfouesneau/sparsegrid/tree/master
    # https://github.com/EconForge/Smolyak/blob/master/smolyak/grid.py

    d = nb_rows
    D = nb_columns
    points, weights = np.polynomial.hermite.hermgauss(deg)

    distribution = chaospy.Normal(0,1)
    points, weights = chaospy.generate_quadrature(1, distribution, sparse=True)
    points = points[0]
    
    points = torch.tensor(points, dtype=torch.float, device=device)
    weights = torch.tensor(weights, device=device)
    points = points * math.sqrt(2) / gamma
    weights = weights / torch.sqrt(torch.tensor(np.pi, device=device))
    
    rand_tensor = torch.rand(D, d, device=device)
    cumulative_weights = torch.cat((torch.tensor([0.0], device=device), torch.cumsum(weights, dim=0)))
    
    # discretize
    # This function discretizes the input tensor using the cumulative weights
    def discretize(input_tensor, bins):
        discretized = torch.zeros_like(input_tensor, dtype=torch.float, device=device)
        for i in range(1, len(bins)):
            mask = (input_tensor >= bins[i - 1]) & (input_tensor < bins[i])
            discretized += mask * (i - 1)
        return discretized
    
    W = discretize(rand_tensor, cumulative_weights)

    for k in range(torch.min(W).to(torch.int).item(), torch.max(W).to(torch.int).item() + 1):        
        W[(W==k)] = points[k]
    return W.T