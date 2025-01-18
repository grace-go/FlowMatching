"""
    Rn
    ==

    Various tools related to R^n
"""

class Rn():
    """"""
    def __init__(self, n):
        self.dim = n
    
    def L(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1`, i.e. `g_1 + g_2`.
        """
        return g_1 + g_2
    
    def L_inv(self, g_1, g_2):
        """
        Left multiplication of `g_2` by `g_1^-1`, i.e. `g_2 - g_1`.
        """
        return g_2 - g_1
    
    def log(self, g):
        """
        Lie group logarithm of `g`, i.e. `A` in Lie algebra such that `exp(A) = g`.
        """
        return g.clone()
    
    def exp(self, A):
        """
        Lie group exponential of `A`, i.e. `g` in Lie group such that `exp(A) = g`.
        """
        return A.clone()