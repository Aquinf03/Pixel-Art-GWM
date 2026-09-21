"""A tokenizer built for this domain rather than adapted to it.

Two departures from dreamer4's, both aimed at the failure we measured
(char_acc peaked at step 1000 then fell 15x while whole-frame accuracy rose):

1. SPATIAL latents, not 16 global ones. dreamer4 compresses a frame to 16
   latent vectors with no spatial structure, so the character's position must
   be encoded globally -- exactly the "compact representation" DIAMOND
   (arXiv 2405.12399) blames for losing small but important objects. A
   convolutional grid gives the fox its own cells, which cannot be averaged
   into the background.

2. The decoder CLASSIFIES over the palette instead of regressing RGB. The
   renderer emits exactly 21 colours, so reconstruction is a 21-way choice per
   pixel. Cross-entropy cannot blur -- it either picks the right colour or it
   does not -- and class weights let us say directly that the character's
   colours matter more than sky, which is the thing MSE could never express.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PALETTE = [
    (26, 38, 52), (34, 52, 66), (36, 72, 54), (44, 68, 78), (58, 106, 74),
    (96, 158, 96), (246, 206, 84), (255, 244, 190), (36, 24, 30),
    (232, 132, 62), (168, 84, 40), (248, 214, 170), (24, 20, 26),
    (92, 46, 34), (255, 255, 255), (208, 110, 50), (120, 96, 148),
    (98, 66, 38), (146, 100, 58), (255, 248, 214), (206, 190, 236),
]
CHAR_IDX = [8, 9, 10, 11, 12, 13, 14, 15, 19]   # the fox's own colours
N_COLORS = len(PALETTE)


def palette_tensor(device) -> torch.Tensor:
    return torch.tensor(PALETTE, dtype=torch.float32, device=device)


def to_indices(rgb_u8: torch.Tensor, pal: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) uint8 -> (B,H,W) long palette indices. Exact for our frames."""
    x = rgb_u8.permute(0, 2, 3, 1).float()                  # (B,H,W,3)
    d = (x.unsqueeze(-2) - pal.view(1, 1, 1, -1, 3)).abs().sum(-1)
    return d.argmin(-1)


def to_rgb(idx: torch.Tensor, pal: torch.Tensor) -> torch.Tensor:
    """(B,H,W) indices -> (B,3,H,W) float in [0,1]."""
    return pal[idx].permute(0, 3, 1, 2) / 255.0


def _block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, width=64, z_ch=8, downs=3):
        super().__init__()
        layers, c = [_block(3, width)], width
        for _ in range(downs):                       # 128 -> 64 -> 32 -> 16
            layers += [_block(c, min(c * 2, 256), stride=2)]
            c = min(c * 2, 256)
            layers += [_block(c, c)]
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv2d(c, z_ch, 1)

    def forward(self, x):
        return torch.tanh(self.out(self.net(x)))     # bounded, like dreamer4's


class Decoder(nn.Module):
    def __init__(self, width=64, z_ch=8, downs=3, n_colors=N_COLORS):
        super().__init__()
        c = min(width * (2 ** downs), 256)
        layers = [_block(z_ch, c)]
        for _ in range(downs):
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), _block(c, max(c // 2, width))]
            c = max(c // 2, width)
            layers += [_block(c, c)]
        self.net = nn.Sequential(*layers)
        self.logits = nn.Conv2d(c, n_colors, 1)      # per-pixel palette choice

    def forward(self, z):
        return self.logits(self.net(z))              # (B,n_colors,H,W)


class PaletteTokenizer(nn.Module):
    def __init__(self, width=64, z_ch=8, downs=3):
        super().__init__()
        self.encoder = Encoder(width, z_ch, downs)
        self.decoder = Decoder(width, z_ch, downs)
        self.z_ch, self.downs = z_ch, downs

    def forward(self, x_rgb01):
        z = self.encoder(x_rgb01)
        return self.decoder(z), z

    @torch.no_grad()
    def reconstruct(self, x_rgb01, pal):
        logits, z = self.forward(x_rgb01)
        return to_rgb(logits.argmax(1), pal), z

    @torch.no_grad()
    def u_r(self, x_rgb01, pal):
        """Latent round trip: ||z - Enc(Dec(z))||, the hallucination signal.

        Unchanged in spirit from the paper -- training never optimises it,
        which is why it stays informative.
        """
        z = self.encoder(x_rgb01)
        rec = to_rgb(self.decoder(z).argmax(1), pal)
        z2 = self.encoder(rec)
        d = (z - z2).flatten(1).norm(dim=1)
        return d, d / z.flatten(1).norm(dim=1).clamp_min(1e-8)


def class_weights(char_weight: float, device) -> torch.Tensor:
    """Upweight the character's colours. MSE had no way to say this."""
    w = torch.ones(N_COLORS, device=device)
    if char_weight > 0:
        w[CHAR_IDX] = char_weight
    return w


def loss_fn(logits, target_idx, weights=None):
    return F.cross_entropy(logits, target_idx, weight=weights)
