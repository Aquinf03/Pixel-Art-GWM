"""Action-conditioned dynamics over the tokenizer's latent grid.

Formulated as FLOW MATCHING rather than direct regression, deliberately: two
of the three hallucination predictors only exist if the dynamics model is a
denoiser.

    u_f  flow instability -- how much the predicted clean latent moves between
         Euler substeps. Needs a solver with substeps.
    u_s  inter-seed variance -- spread across independent noise seeds. Needs
         stochastic sampling.

A plain regressor would predict z_{t+1} directly and both would be undefined,
which is why the discrete-token fallback in the original brief was rejected.

The backbone is convolutional because the latent is a 16x16 grid: the fox
occupies specific cells, and convolution keeps that locality. Actions condition
via FiLM, which lets one action modulate every spatial position without the
model having to route it there.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (9.21 / half))
    a = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([a.cos(), a.sin()], -1)


class FiLM(nn.Module):
    """Condition a feature map on (action, noise level) with a per-channel affine."""

    def __init__(self, cond_dim, ch):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, ch * 2)

    def forward(self, h, c):
        s, b = self.to_scale_shift(c).chunk(2, -1)
        return h * (1 + s[..., None, None]) + b[..., None, None]


class Block(nn.Module):
    def __init__(self, cin, cout, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.film = FiLM(cond_dim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, h, c):
        r = self.skip(h)
        h = F.silu(self.norm1(self.conv1(h)))
        h = self.film(h, c)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + r


class FlowDynamics(nn.Module):
    """Predicts the CLEAN next latent from a noised one, past context and action.

    x1-prediction (predict the target, not the velocity) as in shortcut flow
    matching: it makes the Euler update b = (x1_hat - z) / (1 - tau), which is
    what u_f watches for instability.
    """

    def __init__(self, z_ch=8, ctx=8, action_dim=16, width=128, depth=4):
        super().__init__()
        self.z_ch, self.ctx = z_ch, ctx
        cond = width
        self.act_in = nn.Sequential(nn.Linear(action_dim, cond), nn.SiLU(),
                                    nn.Linear(cond, cond))
        self.t_in = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(),
                                  nn.Linear(cond, cond))
        cin = z_ch * (ctx + 1)                    # past context + noisy target
        self.stem = nn.Conv2d(cin, width, 3, 1, 1)
        self.blocks = nn.ModuleList([Block(width, width, cond) for _ in range(depth)])
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv2d(width, z_ch, 3, 1, 1))

    def forward(self, past, z_noisy, action, tau):
        """past (B,ctx,C,H,W) | z_noisy (B,C,H,W) | action (B,A) | tau (B,)"""
        B = past.shape[0]
        h = torch.cat([past.flatten(1, 2), z_noisy], 1)
        c = self.act_in(action) + self.t_in(timestep_embedding(tau, self.act_in[0].out_features))
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h, c)
        return torch.tanh(self.out(h))            # latents are tanh-bounded


def flow_loss(model, past, z_next, action):
    """Flow matching: interpolate noise->target, ask the model for the target."""
    B = z_next.shape[0]
    tau = torch.rand(B, device=z_next.device)
    noise = torch.randn_like(z_next)
    t = tau.view(-1, 1, 1, 1)
    z_tau = (1 - t) * noise + t * z_next
    pred = model(past, z_tau, action, tau)
    return F.mse_loss(pred, z_next)


@torch.no_grad()
def sample_next(model, past, action, steps=8, generator=None, return_trace=False):
    """Euler-solve noise -> next latent. `return_trace` feeds u_f."""
    B, _, C, H, W = past.shape
    z = torch.randn(B, C, H, W, device=past.device, generator=generator)
    trace = []
    for i in range(steps):
        tau = torch.full((B,), i / steps, device=past.device)
        x1 = model(past, z, action, tau)
        if return_trace:
            trace.append(x1)
        z = z + (x1 - z) / max(1e-4, 1 - i / steps) * (1 / steps)
    return (z, trace) if return_trace else z
