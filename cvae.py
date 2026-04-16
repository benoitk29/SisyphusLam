"""
cvae.py  –  Multi-material CVAE
Encoder : [x(4) + c(21)] → μ(16), log_σ²(16)
Decoder : [z(16) + c(21)] → x^(4)   (tanh, range ≈ [-1,1] like LPs)
Loss    : L_recon + β·L_KL + λ_reg·L_reg + λ_gen·L_gen + λ_phys·L_phys
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

def _block(in_d: int, out_d: int, drop: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_d, out_d), nn.LayerNorm(out_d),
                         nn.SiLU(), nn.Dropout(drop))

class Encoder(nn.Module):
    def __init__(self, feat_dim, cond_dim, latent_dim, hidden_dims, dropout):
        super().__init__()
        in_d = feat_dim + cond_dim
        layers = []
        for h in hidden_dims:
            layers.append(_block(in_d, h, dropout)); in_d = h
        self.net       = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(in_d, latent_dim)
        self.fc_logvar = nn.Linear(in_d, latent_dim)

    def forward(self, x, c):
        h = self.net(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, feat_dim, hidden_dims, dropout):
        super().__init__()
        in_d = latent_dim + cond_dim
        layers = []
        for h in hidden_dims:
            layers.append(_block(in_d, h, dropout)); in_d = h
        self.net    = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_d, feat_dim)

    def forward(self, z, c):
        # tanh maps outputs to the range [-1, 1], suitable for Lamination Parameters
        return torch.tanh(self.fc_out(self.net(torch.cat([z, c], dim=-1))))

class CVAE(nn.Module):
    def __init__(self, feat_dim=4, cond_dim=21, latent_dim=16,
                 hidden_dims=None, dropout=0.05):
        super().__init__()
        if hidden_dims is None: hidden_dims = [256, 512, 512, 256]
        self.feat_dim = feat_dim; self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(feat_dim, cond_dim, latent_dim, hidden_dims, dropout)
        self.decoder = Decoder(latent_dim, cond_dim, feat_dim, hidden_dims[::-1], dropout)

    @staticmethod
    def reparameterise(mu, log_var):
        if not torch.is_grad_enabled(): return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def encode(self, x, c): return self.encoder(x, c)
    def decode(self, z, c): return self.decoder(z, c)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        return self.decode(self.reparameterise(mu, log_var), c), mu, log_var

    @torch.no_grad()
    def generate(self, c, n_samples=1, temperature=1.0):
        c = c.view(1, self.cond_dim).expand(n_samples, -1)
        return self.decode(torch.randn(n_samples, self.latent_dim,
                                       device=c.device) * temperature, c)

def miki_penalty(x_hat: torch.Tensor) -> torch.Tensor:
    """
    Calculates Miki's geometric penalty for the Lamination Parameters.
    V2 >= 2*V1^2 - 1  --> max(0, 2*V1^2 - 1 - V2)
    V4 >= 2*V3^2 - 1  --> max(0, 2*V3^2 - 1 - V4)
    """
    v1, v2, v3, v4 = x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], x_hat[:, 3]
    g1 = F.relu(2 * v1**2 - 1 - v2)
    g2 = F.relu(2 * v3**2 - 1 - v4)
    return torch.mean(g1**2 + g2**2)

def cvae_loss(x_hat, x, mu, log_var, beta=1.0,
              lam_reg=0., lam_gen=0., lam_phys=0.,
              x_gen_target=None, mu_gen=None,
              phys_penalty=None, feat_dim=4):
    """L = L_recon + β·KL + λ_reg·L_reg + λ_gen·L_gen + λ_phys·L_phys"""
    recon = F.mse_loss(x_hat, x)
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    reg   = (F.mse_loss(mu[:, :feat_dim], x) if lam_reg > 0
             else x.new_zeros(1).squeeze())
    gen   = (F.mse_loss(mu_gen[:, :feat_dim], x_gen_target)
             if lam_gen > 0 and mu_gen is not None and x_gen_target is not None
             else x.new_zeros(1).squeeze())
             
    phys  = phys_penalty if (phys_penalty is not None and lam_phys > 0) else x.new_zeros(1).squeeze()
    total = recon + beta * kl + lam_reg * reg + lam_gen * gen + lam_phys * phys
    return total, recon, kl, reg, gen, phys

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)