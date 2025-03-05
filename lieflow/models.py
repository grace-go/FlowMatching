"""
    models
    ======

    Implementations of flow matching[1] and shortcut modeling[2] on Lie groups
    over exponential curves.[3]
    The implementation is slightly different for groups of type `Group` than
    for those of type `MatrixGroup`; the methods `get_model_FM` and
    `get_model_SCFM` will return an instance of a flow matching and shortcut
    model, respectively, corresponding to the type of the group.

    References:
      [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
      "Flow Matching for Generative Modeling." arXiv preprint (2022).
      DOI:10.48550/arXiv.2210.02747.
      [2]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
      "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
      DOI:10.48550/arXiv.2410.12557
      [3]:
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from lieflow.groups import Group, MatrixGroup


def get_model_FM(G: Group | MatrixGroup, H=64, L=2):
    """
    Return an instance of a flow matching model corresponding to the type of
    `G`.

    Args:
        G: group of type `Group` or `MatrixGroup`.
      Optional:
        H: width of the network: number of channels. Defaults to 64.
        L: depth of the network: number of layers - 2. Defaults to 2.
    """
    if isinstance(G, Group):
        return FlowFieldGroup(G, H=H, L=L)
    elif isinstance(G, MatrixGroup):
        return FlowFieldMatrixGroup(G, H=H, L=L)
    else:
        raise ValueError(f"{G} is neither a `Group` nor a `Matrix Group`!")
    
def get_model_SCFM(G: Group | MatrixGroup, H=64, L=2):
    """
    Return an instance of a shortcut model corresponding to the type of `G`.
    
    Args:
        G: group of type `Group` or `MatrixGroup`.
      Optional:
        H: width of the network: number of channels. Defaults to 64.
        L: depth of the network: number of layers - 2. Defaults to 2.
    """
    if isinstance(G, Group):
        return ShortCutFieldGroup(G, H=H, L=L)
    elif isinstance(G, MatrixGroup):
        return ShortCutFieldMatrixGroup(G, H=H, L=L)
    else:
        raise ValueError(f"{G} is neither a `Group` nor a `Matrix Group`!")
    

class FlowFieldGroup(nn.Module):
    """
    Model for flow matching[1] on Lie groups of type `Group` over exponential
    curves.[2]
    
    Args:
        G: group of type `Group` or `MatrixGroup`.
      Optional:
        H: width of the network: number of channels. Defaults to 64.
        L: depth of the network: number of layers - 2. Defaults to 2.

    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]:
    """
    
    def __init__(self, G: Group, H=64, L=2):
        super().__init__()
        self.G = G
        self.network = nn.Sequential(
            nn.Linear(G.dim+1, H), nn.ReLU(),
            *(L*(nn.Linear(H, H), nn.ReLU(),)),
            nn.Linear(H, G.dim)
        )

    def forward(self, g_t, t):
        return self.network(torch.cat((g_t, t), dim=-1))
    
    def step(self, g_t, t, Δt):
        t = t.view(1, 1).expand(g_t.shape[0], 1)
        return self.G.L(g_t, self.G.exp(Δt * self(g_t, t)))
    
    def train_network(self, device, train_loader, optimizer, loss):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        for i, (g_0, g_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch"
        ):
            t = torch.rand(len(g_1), 1).to(device)
            g_0, g_1 = g_0.to(device), g_1.to(device)
            A_t = self.G.log(self.G.L_inv(g_0, g_1))
            g_t = self.G.L(g_0, self.G.exp(t * A_t))
            optimizer.zero_grad()
            batch_loss = loss(self(g_t, t), A_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()
    
class ShortCutFieldGroup(nn.Module):
    """
    Model for shortcut modeling[1] on Lie groups of type `Group` over
    exponential curves.[2]
    
    Args:
        G: group of type `Group` or `MatrixGroup`.
      Optional:
        H: width of the network: number of channels. Defaults to 64.
        L: depth of the network: number of layers - 2. Defaults to 2.
    
    References:
        [1]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
          "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
          DOI:10.48550/arXiv.2410.12557
        [2]:
    """

    def __init__(self, G: Group, H=64, L=2):
        super().__init__()
        self.G = G
        self.network = nn.Sequential(
            nn.Linear(G.dim+2, H), nn.ReLU(),
            *(L*(nn.Linear(H, H), nn.ReLU(),)),
            nn.Linear(H, G.dim)
        )

    def forward(self, g_t, t, Δt):
        return self.network(torch.cat((g_t, t, Δt), dim=-1))
    
    def step(self, g_t, t, Δt):
        t = t.view(1, 1).expand(g_t.shape[0], 1)
        Δt = Δt.view(1, 1).expand(g_t.shape[0], 1)
        return self.G.L(g_t, self.G.exp(Δt * self(g_t, t, Δt)))
    
    def train_network(self, device, train_loader, optimizer, loss, k=1/4):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        for i, (g_0, g_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch"
        ):
            N_total = len(g_1) # Total number of samples in batch.
            t = torch.rand(len(g_1), 1).to(device)
            g_0, g_1 = g_0.to(device), g_1.to(device)
            g_t = self.G.L(g_0, self.G.exp(t * self.G.log(self.G.L_inv(g_0, g_1))))
            
            Δt = torch.rand(N_total, 1).to(device) * (1 - t) # t + Δt <= 1.
            A_t = torch.zeros_like(g_t)
            N_SG = int(k * N_total) # Number of samples used for self-consistency loss.

            # Flow-Matching
            g_1_FM, g_0_FM = g_1[N_SG:], g_0[N_SG:]
            A_t[N_SG:] = self.G.log(self.G.L_inv(g_0_FM, g_1_FM))
            Δt[N_SG:] = 0.

            # Self-Consistency
            g_t_SG, t_SG, Δt_SG = g_t[:N_SG], t[:N_SG], Δt[:N_SG]
            B_t = self(g_t_SG, t_SG, Δt_SG)
            g_t_Δt = self.G.L(g_t_SG, self.G.exp(Δt_SG * B_t))
            B_t_Δt = self(g_t_Δt, t_SG + Δt_SG, Δt_SG)
            with torch.no_grad():
                A_t[:N_SG] = (B_t + B_t_Δt) / 2

            optimizer.zero_grad()
            batch_loss = loss(self(g_t, t, 2 * Δt), A_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()
    

class FlowFieldMatrixGroup(nn.Module):
    """
    Model for flow matching[1] on Lie groups of type `MatrixGroup` over
    exponential curves.[2]
    
    Args:
        G: group of type `Group` or `MatrixGroup`.
      Optional:
        H: width of the network: number of channels. Defaults to 64.
        L: depth of the network: number of layers - 2. Defaults to 2.
    
    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
          "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
          DOI:10.48550/arXiv.2410.12557
        [3]:
    """
    
    def __init__(self, G: MatrixGroup, H=64, L=2):
        super().__init__()
        self.G = G
        self.network = nn.Sequential(
            nn.Linear(G.mat_dim+1, H), nn.ReLU(),
            *(L*(nn.Linear(H, H), nn.ReLU(),)),
            nn.Linear(H, G.dim)
        )

    def forward(self, R_t, t):
        return self.network(torch.cat((R_t.flatten(start_dim=-2, end_dim=-1), t.flatten(start_dim=-2, end_dim=-1)), dim=-1))
    
    def step(self, R_t, t, Δt):
        t = t.view(1, 1).expand(R_t.shape[0], 1, 1)
        # Components w.r.t. Lie algebra basis.
        a_t = self(R_t, t)
        basis = self.G.lie_algebra_basis
        A_t = (a_t[..., None, None] * basis).sum(-3)
        return self.G.L(R_t, self.G.exp(Δt * A_t))
    
    def train_network(self, device, train_loader, optimizer, loss):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        for i, (R_0, R_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch"
        ):
            t = torch.rand(len(R_1), 1, 1).to(device)
            R_0, R_1 = R_0.to(device), R_1.to(device)
            A_t = self.G.log(self.G.L_inv(R_0, R_1))
            R_t = self.G.L(R_0, self.G.exp(t * A_t))
            # Components w.r.t. Lie algebra basis.
            a_t = self.G.lie_algebra_components(A_t)
            optimizer.zero_grad()
            batch_loss = loss(self(R_t, t), a_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()
    
class ShortCutFieldMatrixGroup(nn.Module):
    """
    Model for shortcut modeling[1] on Lie groups of type `MatrixGroup` over
    exponential curves.[2]
    
    Args:
        G: group of type `Group` or `MatrixGroup`.
      Optional:
        H: width of the network: number of channels. Defaults to 64.
        L: depth of the network: number of layers - 2. Defaults to 2.
    
    References:
        [1]: Y. Lipman, R.T.Q. Chen, H. Ben-Hami, M. Nickel, and M. Le.
          "Flow Matching for Generative Modeling." arXiv preprint (2022).
          DOI:10.48550/arXiv.2210.02747.
        [2]: K. Frans, D. Hafner, S. Levine, and P. Abbeel.
          "One Step Diffusion via Shortcut Models." arXiv preprint (2024).
          DOI:10.48550/arXiv.2410.12557
        [3]:
    """

    def __init__(self, G: MatrixGroup, H=64, L=2):
        super().__init__()
        self.G = G
        self.network = nn.Sequential(
            nn.Linear(G.mat_dim+2, H), nn.ReLU(),
            *(L*(nn.Linear(H, H), nn.ReLU(),)),
            nn.Linear(H, G.dim)
        )

    def forward(self, R_t, t, Δt):
        return self.network(torch.cat((R_t.flatten(start_dim=-2, end_dim=-1), t.flatten(start_dim=-2, end_dim=-1), Δt.flatten(start_dim=-2, end_dim=-1)), dim=-1))
    
    def step(self, R_t, t, Δt):
        t = t.view(1, 1).expand(R_t.shape[0], 1, 1)
        Δt = Δt.view(1, 1).expand(R_t.shape[0], 1, 1)
        # Components w.r.t. Lie algebra basis.
        a_t = self(R_t, t, Δt)
        basis = self.G.lie_algebra_basis
        A_t = a_t[..., None] * basis
        return self.G.L(R_t, self.G.exp(Δt * A_t))
    
    def train_network(self, device, train_loader, optimizer, loss, k=1/4):
        self.train()
        N_batches = len(train_loader)
        losses = np.zeros(N_batches)
        basis = self.G.lie_algebra_basis
        for i, (R_0, R_1) in tqdm(
            enumerate(train_loader),
            total=N_batches,
            desc="Training",
            dynamic_ncols=True,
            unit="batch"
        ):
            N_total = len(R_1) # Total number of samples in batch.
            t = torch.rand(len(R_1), 1).to(device)
            R_0, R_1 = R_0.to(device), R_1.to(device)
            R_t = self.G.L(R_0, self.G.exp(t * self.G.log(self.G.L_inv(R_0, R_1))))
            
            Δt = torch.rand(N_total, 1, 1).to(device) * (1 - t) # t + Δt <= 1.
            A_t = torch.zeros_like(R_t)
            N_SG = int(k * N_total) # Number of samples used for self-consistency loss.

            # Flow-Matching
            R_1_FM, R_0_FM = R_1[N_SG:], R_0[N_SG:]
            A_t[N_SG:] = self.G.log(self.G.L_inv(R_0_FM, R_1_FM))
            Δt[N_SG:] = 0.

            # Self-Consistency
            R_t_SG, t_SG, Δt_SG = R_t[:N_SG], t[:N_SG], Δt[:N_SG]
            b_t = self(R_t_SG, t_SG, Δt_SG)
            B_t = (b_t[..., None, None] * basis).sum(-1)
            R_t_Δt = self.G.L(R_t_SG, self.G.exp(Δt_SG * B_t))
            b_t_Δt = self(R_t_Δt, t_SG + Δt_SG, Δt_SG)
            B_t_Δt = (b_t_Δt[..., None, None] * basis).sum(-1)
            with torch.no_grad():
                A_t[:N_SG] = (B_t + B_t_Δt) / 2
                
            # Components w.r.t. Lie algebra basis.
            a_t = self.G.lie_algebra_components(A_t)

            optimizer.zero_grad()
            batch_loss = loss(self(R_t, t, 2 * Δt), a_t)
            losses[i] = float(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        return losses.mean()


class LogarithmicDistance(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
        self.ρ = lambda A_1, A_2: self.ρ_c_normalised(A_1, A_2, w)
    
    def ρ_c_normalised(self, A_1, A_2, w):
        return (w**2 * (A_2 - A_1)**2).mean(-1)
    
    def forward(self, input, target):
        return self.ρ(input, target).mean(-1)

    def __repr__(self):
        return f"LogarithmicDistance{tuple(self.w.numpy())}"