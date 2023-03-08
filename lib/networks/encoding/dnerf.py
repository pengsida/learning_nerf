import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import isprime
from lib.networks.nerf.encoding.freq import Encoder as FreqEncoder
from lib.config import cfg

eps = 1e-6


class DNeRF(nn.Module):
    def __init__(self, skips=[4,], **kwargs):
        super(DNeRF, self).__init__()
        freq, W, D = kwargs['freq'], kwargs['W'], kwargs['D']
        encoder_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : freq-1,
                'num_freqs' : freq,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
        }
        self.xyz_embedder = FreqEncoder(**encoder_kwargs)
        xyz_ch = self.xyz_embedder.out_dim
        encoder_kwargs['input_dims'] = 1
        self.time_embedder = FreqEncoder(**encoder_kwargs)
        time_ch = self.time_embedder.out_dim

        self.skips = skips
        input_ch = xyz_ch + time_ch
        self.time_mlp = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.time_out = nn.Linear(W, 3)
        self.out_dim = xyz_ch

    def forward(self, x, wbounds=None):
        xyz, t = x[..., :3], x[..., 3:]
        if t[0] == -1:
            return self.xyz_embedder.embed(xyz)
        else:
            t = t / (cfg.num_frames - 1)
            xyz_encoding, t_encoding = self.xyz_embedder.embed(xyz), self.time_embedder.embed(t)
            encoding = torch.cat([xyz_encoding, t_encoding], dim=-1)
            h = encoding
            for i, l in enumerate(self.time_mlp):
                h = self.time_mlp[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([encoding, h], -1)
            delta_xyz = self.time_out(h)
            return self.xyz_embedder.embed(xyz+delta_xyz)


    def compute_delta(self, xyz, t):
        B, N_samples = xyz.shape[:2]
        xyz, t = xyz.reshape(-1, xyz.shape[-1]), t.reshape(-1, t.shape[-1])
        t = t / (cfg.num_frames - 1)
        xyz_encoding, t_encoding = self.xyz_embedder.embed(xyz), self.time_embedder.embed(t)
        encoding = torch.cat([xyz_encoding, t_encoding], dim=-1)
        h = encoding
        for i, l in enumerate(self.time_mlp):
            h = self.time_mlp[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([encoding, h], -1)
        delta_xyz = self.time_out(h)
        return delta_xyz.reshape(B, N_samples, -1)


    def compute_tv_loss(self, xyz, t, wbounds=None):
        time = t[0, 0].item()
        if time == 0.:
            delta_xyz_prev = self.compute_delta(xyz, t)
            delta_xyz_next = self.compute_delta(xyz, t + 1.)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        else:
            delta_xyz_prev = self.compute_delta(xyz, t - 1.)
            delta_xyz_next = self.compute_delta(xyz, t)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        return delta


