"""
    SE2
    ===

    Various tools related to SE(2)
"""

import torch

class SE2():
    """"""
    def __init__(self):
        self.dim = 3
    
    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1`.
        """
        g = torch.zeros_like(g_2)
        x_1 = g_1[..., 0]
        y_1 = g_1[..., 1]
        θ_1 = g_1[..., 2]
        cos = torch.cos(θ_1)
        sin = torch.sin(θ_1)
        x_2 = g_2[..., 0]
        y_2 = g_2[..., 1]
        θ_2 = g_2[..., 2]

        g[..., 0] = x_1 + cos * x_2 - sin * y_2
        g[..., 1] = y_1 + sin * x_2 + cos * y_2
        g[..., 2] = θ_1 + θ_2
        return g
    
    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1^-1`.
        """
        g = torch.zeros_like(g_2)
        x_1 = g_1[..., 0]
        y_1 = g_1[..., 1]
        θ_1 = g_1[..., 2]
        cos = torch.cos(θ_1)
        sin = torch.sin(θ_1)
        x_2 = g_2[..., 0]
        y_2 = g_2[..., 1]
        θ_2 = g_2[..., 2]

        g[..., 0] = cos * (x_2 - x_1) + sin * (y_2 - y_1)
        g[..., 1] = -sin * (x_2 - x_1) + cos * (y_2 - y_1)
        g[..., 2] = θ_2 - θ_1
        return g
    
    def log(self, g):
        """
        Lie group logarithm of `g`, i.e. `A` in Lie algebra such that `exp(A) = g`.
        """
        A = torch.zeros_like(g)
        θ = g[..., 2]
        
        parallel = θ == 0.
        # Maybe more stable?
        # ε = 1e-8
        # parallel = θ.abs() < ε

        A[parallel, 0] = g[parallel, 0]
        A[parallel, 1] = g[parallel, 1]

        x_not_par = g[~parallel, 0]
        y_not_par = g[~parallel, 1]
        θ_not_par = g[~parallel, 2]
        tan = torch.tan(θ_not_par/2.)
        A[~parallel, 0] = θ_not_par/2. * (y_not_par + x_not_par / tan)
        A[~parallel, 1] = θ_not_par/2. * (-x_not_par + y_not_par / tan)
        A[~parallel, 2] = θ_not_par
        return A
    
    def exp(self, A):
        """
        Lie group exponential of `A`, i.e. `g` in Lie group such that `exp(A) = g`.
        """
        g = torch.zeros_like(A)
        c3 = A[..., 2]
        parallel = c3 == 0.
        g[parallel, 0] = A[parallel, 0]
        g[parallel, 1] = A[parallel, 1]

        c1_not_par = A[~parallel, 0]
        c2_not_par = A[~parallel, 1]
        c3_not_par = A[~parallel, 2]
        cos = torch.cos(c3_not_par/2.)
        sin = torch.sin(c3_not_par/2.)
        g[~parallel, 0] = (c1_not_par * cos - c2_not_par * sin) * sin / (c3_not_par/2.)
        g[~parallel, 1] = (c1_not_par * sin + c2_not_par * cos) * sin / (c3_not_par/2.)
        g[~parallel, 2] = _mod_offset(c3_not_par, 2 * torch.pi, 0.)
        return g

def _mod_offset(x, period, offset):
    """Compute `x` modulo `period` with offset `offset`."""
    return x - (x - offset)//period * period