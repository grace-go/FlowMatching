"""
    SE2
    ===

    Various tools related to SO(3)
"""

import torch

class SO3():
    """"""
    def __init__(self):
        self.dim = 3
    
    def L(self, R_1, R_2):
        """
        Left multiplication of `R_2` by `R_1`.
        """
        return torch.matmul(R_1, R_2)
    
    def L_inv(self, R_1, R_2):
        """
        Left multiplication of `R_2` by `R_1^-1`.
        """
        return torch.linalg.solve(R_1, R_2)

    def log(self, R):
        """
        Lie group logarithm of `R`, i.e. `A` in Lie algebra such that
        `exp(A) = R`.

        Pytorch does not actually have a matrix log built in, but for SO(3) it
        is not too complicated.
        """
        q = torch.arccos((R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2)
        return (R - R.transpose(-2, -1)) / (2 * torch.sinc(q[..., None, None] / torch.pi))

    def exp(self, A):
        """
        Lie group exponential of `A`, i.e. `R` in Lie group such that
        `exp(A) = R`.
        """
        return torch.matrix_exp(A)
    
    def __repr__(self):
        return "SO(3)"