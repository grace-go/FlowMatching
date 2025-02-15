"""
    SE2byRn
    =======

    Various tools related to SE(2) x R^n
"""

from SE2 import SE2
from Rn import Rn
import torch

class SE2byRn():
    """"""
    def __init__(self, se2: SE2, rn: Rn):
        self.dim = se2.dim + rn.dim
        self.se2 = se2
        self.rn = rn
    
    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2 = (x_2, p_2)` by `g_1 = (x_1, p_1)`, i.e.
        `(x_1 + x_2, p_1 p_2)`.
        """
        g = torch.zeros_like(g_2)
        g[..., :3] = self.se2.L(g_1[..., :3], g_2[..., :3])
        g[..., 3:] = self.rn.L(g_1[..., 3:], g_2[..., 3:])
        return g_1 + g_2
    
    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2 = (x_2, p_2)` by `g_1^-1 = (-x_1, p_1^-1)`,
        i.e. `(x_2 - x_1, p_1^-1 p_2)`.
        """
        g = torch.zeros_like(g_2)
        g[..., :3] = self.se2.L_inv(g_1[..., :3], g_2[..., :3])
        g[..., 3:] = self.rn.L_inv(g_1[..., 3:], g_2[..., 3:])
        return g_2 - g_1
    
    def log(self, g):
        """
        Lie group logarithm of `g = (x, p)`, i.e. `(x, P)` with `P` in Lie
        algebra such that `exp(P) = p`.
        """
        A = torch.zeros_like(g)
        A[..., :3] = self.se2.log(g[..., :3])
        A[..., 3:] = self.rn.log(g[..., 3:])
        return A
    
    def exp(self, A):
        """
        Lie group exponential of `A = (x, P)`, i.e. `(x, p)` with `p` in Lie
        group such that `exp(P) = p`.
        """
        g = torch.zeros_like(A)
        g[..., :3] = self.se2.exp(A[..., :3])
        g[..., 3:] = self.rn.exp(A[..., 3:])
        return g
    
    def __repr__(self):
        return f"SE(2) x R^{self.rn.dim}"