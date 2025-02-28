import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class FlowField(nn.Module):
    def __init__(self, G, H=64, L=2):
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
    
class ShortCutField(nn.Module):
    def __init__(self, G, H=64, L=2):
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