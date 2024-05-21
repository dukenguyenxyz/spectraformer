import torch
import torch.optim as optim
from torch import nn
from einops import rearrange, repeat

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def derf_compfunc():
    pass

def get_params_trigrf(x,y):
    b, h, j, d = x.shape
    return (torch.zeros(h,j, device=x.device), -1)

def get_params_posrf(x,y):
    b, h, j, d = x.shape
    return (torch.zeros(h,j, device=x.device), 1)

def optimize_geom_params_torch(x_, y_, steps=100):
    x = x_
    y = y_
    
    device = x.device

    b, h, s, d = x.shape
    stat = (torch.abs(x).mean(dim=0) * torch.abs(y).mean(dim=0))
    eps = torch.tensor(1e-8, device=device)

    p = torch.zeros(h, s, device=device, dtype=torch.float).uniform_(1e-8, 1 - 1e-8).unsqueeze(-1).requires_grad_()
    p_copy = p.detach().clone().squeeze(-1)
    # p = torch.full((h, s), 0.5, requires_grad=True)

    sub_optimizer = optim.AdamW([p], lr=1e-8) # 
    # print("GRAD", p.requires_grad)

    def closure():
        sub_optimizer.zero_grad()
        # Ensure p is broadcastable with stat; might need to unsqueeze to match dimensions
        # p_unsq = p.unsqueeze(-1)  # Adding dimension for d; [h, s, 1]
        bessel_args = (2 * torch.pow(1 - p, -0.5) * stat)  # Ensuring broadcastability
        # Using torch.i0e for Bessel function approximation; correct for missing torch.special.bessel_j0
        # Would be torch.special.bessel_j0 for version > 1.8
        loss = (-(torch.log1p(torch.special.bessel_j0(bessel_args)) + bessel_args).sum() - torch.log1p(p).sum() * d)
        loss.backward()
        return loss
    
    for step in range(steps):
        sub_optimizer.step(closure)

        # Optionally clamp p values within bounds after each update step
        with torch.no_grad():
            p.clamp_(min=eps, max=1 - eps)
    
    p_val = p.detach().squeeze(-1)
    return (p_val,)

def optimize_opt_pos_params_torch(x_, y_):
    x = x_
    y = y_
    
    d = x.shape[-1]

    x_stat = (torch.norm(x, p=2, dim=-1, keepdim=True) ** 2).mean(dim=0)
    y_stat = (torch.norm(y, p=2, dim=-1, keepdim=True) ** 2).mean(dim=0)
    
    xy_stat = (x.mean(dim=0) * y.mean(dim=0)).sum(dim=-1)
    xy_sum = torch.squeeze(x_stat + y_stat, dim=-1)
    full_xy_stat = 2 * xy_stat + xy_sum

    a = 2 * full_xy_stat
    b = 2 * full_xy_stat + d
    c = -d
    rho = (torch.sqrt(b * b - 4 * a * c) - b) / (2 * a)
    A = (1 - 1 / rho) / 8    
    return (A, 1)

# Do GERF and then ASDERF
def get_rfs_torch(data, *, d=None, rf_type=None, rf_params=None, non_matrx_linear_op=None, query_dim=None, projection_matrix=None, is_query=None, normalize_data=True, eps=1e-4, device = None, psi=None):
    omega=projection_matrix
    if omega is None:
        raise Exception('Omega is none')

    d_ = d
    on_second_arg = not is_query    
    
    b, h, j, d = data.shape
    
    norms = torch.norm(data, p=2, dim=-1, keepdim=True)

    if rf_type in ['nrff', 'trigrf']:
        # data = data.type(torch.float64)
        if rf_type == 'nrff':
            raise Exception('NRFF not yet supported')
            data = data / norms

        if non_matrx_linear_op:
            data_dash = omega(data).type_as(data)            
        else:
            projection = repeat(omega, 'j d -> b h j d', b = b, h = h)
            data_dash = torch.einsum('...id,...jd->...ij', data.type_as(projection), projection)
                        
            if torch.is_complex(data_dash):
                data_dash = torch.real(data_dash)

            data_dash.type_as(data)
            
        data_dash = torch.cat([torch.cos(data_dash), torch.sin(data_dash)], axis=-1)
        data_renormalizer = torch.max(data_dash, dim=-1, keepdim=True).values
        norms = norms - data_renormalizer
        norms = torch.exp(norms)
        data_prime = data_dash * norms

        return data_prime.type_as(data)
    
    elif rf_type in ['posrf', 'oprf', 'gerf', 'saderf']:
        A, s = rf_params
        
        if rf_type == 'gerf':
            A = torch.clamp(A, max=0.124999)
        
        C = -(s + 1) / 2
        D_arg = torch.pow(1 - 4 * A, 1 / 4)
        D_arg_dir = D_arg / torch.abs(D_arg)
        D_dir = torch.pow(D_arg_dir, d_)

        if s == 1:
            B = torch.sqrt(1 - 4 * A)
        else:
            B = torch.sqrt(4 * A - 1 + 0j)

        if rf_type == 'saderf':
            if on_second_arg:
                psi = torch.linalg.inv(psi)
            data = data @ psi
        
        if non_matrx_linear_op:
            data_dash = omega(data).type_as(data)            
            omega_weights = omega.get_weight_matrix()
            omega_norms = torch.unsqueeze(torch.norm(omega_weights, p=2, dim=-1, keepdim=True), dim=0).type_as(data)

        else:
            projection = repeat(omega, 'j d -> b h j d', b = b, h = h)
            data_dash = torch.einsum('...id,...jd->...ij', data.type_as(projection), projection)

            if torch.is_complex(data_dash):
                data_dash = torch.real(data_dash)
                
            data_dash = data_dash.type_as(data)

            omega_norms = rearrange(torch.norm(omega, p=2, dim=1, keepdim=True), 'a b -> b a')
        
        sB = B
        if on_second_arg:
            sB = s * B

        eq_a1 = torch.unsqueeze(A, -1) @ (omega_norms ** 2)
        eq_a2 = repeat(eq_a1, 'h j d -> b h j d', b=b)

        # f here is number of feature
        eq_b1 = repeat(sB, 'h j -> b h j f', b=b, f=data_dash.shape[-1])
        eq_b2 = eq_b1 * data_dash
        eq_c = C * norms ** 2
        # print(a.shape,b.shape,c.shape)
        data_prime1 = eq_a2 + eq_b2 + eq_c

        if torch.is_complex(data_prime1):
            rfs_max = torch.max(data_prime1.real, dim=-1, keepdim=True).values
        else:
            rfs_max = torch.max(data_prime1, dim=-1, keepdim=True).values
            
        data_prime1 = data_prime1 - rfs_max
        
        data_prime2 = repeat(D_dir, 'h j -> b h j f', b=b, f=data_dash.shape[-1]) * torch.exp(data_prime1)

        if torch.is_complex(data_prime2):
            data_prime2 = torch.cat([data_prime2.real, data_prime2.imag], axis=-1)

    elif rf_type.startswith('geomrf'):        
        if non_matrx_linear_op:
            ratio = query_dim * -0.5
        else:
            ratio = (projection_matrix.shape[0] ** -0.5)

        p, = rf_params
        log_facts = torch.lgamma(omega + 1).sum(dim=1, keepdims=True)

        projection = repeat(omega, 'j d -> b h j d', b = b, h = h).type_as(data)

        rf_signs = (-torch.sign(data) + 1) / 2
        rf_signs = torch.einsum('...id,...jd->...ij', rf_signs, projection).type(torch.int) % 2
        rf_signs = -rf_signs * 2 + 1

        p1 = repeat(p, 'h j -> b h j d', b = b, d = d).type_as(data)
        
        data2 = torch.log1p(torch.abs(data)) - torch.log1p(1 - p1) / 2

        data_dash = torch.einsum('...id,...jd->...ij', data2, projection)
        
        p2 = repeat(p, 'h j -> b h j d', b = b, d = data_dash.shape[-1]).type_as(data)
        
        log_facts = repeat(torch.squeeze(log_facts, dim=-1), 'd -> b h j d', b=b, h=h, j=j)

        data_prime = data_dash - torch.log1p(p2) * d / 2 - norms ** 2 / 2 - log_facts / 2

        rfs_max = torch.max(data_prime, dim=-1, keepdim=True).values
        data_prime = data_prime - rfs_max
        data_prime2 = ratio * rf_signs * torch.exp(data_prime)

    return data_prime2.type_as(data) + eps