import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sympy import isprime
from lib.config import cfg
from lib.networks.nerf.encoding.freq import Encoder as FreqEncoder

eps = 1e-6
class HashGrid(nn.Module):
    def __init__(self, **kwargs):
        """
        """
        super(HashGrid, self).__init__()
        n_levels, log2_hashmap_size, base_resolution, desired_resolution = kwargs['num_levels'], kwargs['log2_hashmap_size'], kwargs['base_resolution'], kwargs['desired_resolution']
        b, n_features_per_level = kwargs['per_level_scale'], kwargs['level_dim']

        self.n_levels = n_levels
        self.n_entrys_per_level = 2**log2_hashmap_size
        while True:
            if isprime(self.n_entrys_per_level):
                break
            else:
                self.n_entrys_per_level += 1
        if desired_resolution != -1:
            self.b = (desired_resolution / base_resolution) ** (1 / (n_levels - 1))
        else:
            self.b = b
        self.base_resolution = base_resolution
        self.f = n_features_per_level
        self.out_dim = self.f * self.n_levels
        self.output_dim = self.out_dim

        self.offsets = [0]
        self.scales = []
        self.start_hash = -1
        for i in range(self.n_levels):
            res = int((self.base_resolution) * (self.b**i))
            self.scales.append(res)
            n_entrys = int((res + 1) ** 3)
            if n_entrys > self.n_entrys_per_level:
                if self.start_hash < 0:
                    self.start_hash = i
                n_entrys = self.n_entrys_per_level
            self.offsets.append(self.offsets[-1] + n_entrys)

        self.data = nn.Parameter(torch.zeros((self.offsets[-1], self.f)))
        std = 1e-4
        self.data.data.uniform_(-std, std)
        self.ps = [1, 19349663, 83492791]

        self.offsets_pos = torch.tensor([[0., 0., 0.],
                                     [0., 0., 1.],
                                     [0., 1., 0.],
                                     [0., 1., 1.],
                                     [1., 0., 0.],
                                     [1., 0., 1.],
                                     [1., 1., 0.],
                                     [1., 1., 1.]]).float().cuda()

        self.scales = torch.tensor(self.scales).cuda().float()
        self.offsets = torch.tensor(np.array(self.offsets)).cuda().long()

    def forward(self, xyz, wbounds=None, normalize=True):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]
        if normalize:
            inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
            inputs = inputs - wbounds[None][:, :3]
            inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        else:
            inputs = xyz

        inputs = inputs[None].repeat(self.n_levels, 1, 1)
        float_xyz = inputs * self.scales[:, None, None]
        int_xyz = (float_xyz[:, :, None] + self.offsets_pos[None, None]).long()
        offset_xyz = float_xyz - int_xyz[:, :, 0]

        ind = torch.zeros_like(int_xyz[..., 0])
        if self.start_hash == -1:
            sh = self.n_levels
        else:
            sh = self.start_hash

        ind[:sh] = int_xyz[:sh, ..., 0] * ((self.scales[:sh] + 1)**2)[:, None, None] + \
                   int_xyz[:sh, ..., 1] * ((self.scales[:sh] + 1))[:, None, None] + \
                   int_xyz[:sh, ..., 2]
        nl = self.n_levels
        if self.start_hash != -1:
            ind[sh:nl] = torch.bitwise_xor(torch.bitwise_xor(int_xyz[sh:nl, ..., 0]*self.ps[0], int_xyz[sh:nl, ..., 1]*self.ps[1]), int_xyz[sh:nl, ..., 2]*self.ps[2]) % self.n_entrys_per_level

        ind = ind.reshape(nl, -1)
        ind += self.offsets[:-1, None]
        ind = ind.reshape(-1)
        val = torch.gather(self.data, 0, ind[:, None].repeat(1, self.f))
        val = val.reshape(nl, -1, 8, self.f)

        weights_xyz = torch.clamp((1 - self.offsets_pos[None, None]) + (2 * self.offsets_pos[None, None] - 1.) * offset_xyz[:, :, None], min=0., max=1.)
        weights_xyz = weights_xyz[..., 0] * weights_xyz[..., 1] * weights_xyz[..., 2]
        val = (weights_xyz[..., None] * val).sum(dim=-2)
        val = val.permute(1, 0, 2).reshape(-1, nl*self.f)
        return val


class DNeRFNGP(nn.Module):
    def __init__(self, skips=[4,], **kwargs):
        super(DNeRFNGP, self).__init__()
        self.encoder = HashGrid(**kwargs)
        F, reso = 64, 256
        self.feat = nn.ParameterList([nn.Parameter(0.1 * torch.randn((3, F, cfg.num_frames, reso))) for i in range(3)])
        self.out_dim = self.encoder.out_dim

    def forward(self, x, wbounds=None):
        xyz, t = x[..., :3], x[..., 3:]
        inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item())
        if t[0] == -1.:
            return self.encoder(torch.clamp(inputs, min=0., max=1. - eps), normalize=False)
        else:
            delta_xyz = self.compute_delta(inputs, t)
            return self.encoder(torch.clamp(inputs+delta_xyz, min=0., max=1. - eps), normalize=False)

    def compute_delta(self, xyz, t):
        if len(xyz.shape) == 3:
            B, N_samples = xyz.shape[:2]
            xyz, t = xyz.reshape(-1, xyz.shape[-1]), t.reshape(-1, t.shape[-1])
        else:
            B = None
        t = t / (cfg.num_frames - 1)

        coord = torch.stack([torch.cat([xyz[..., i:i+1], t], dim=-1) for i in range(3)])
        feats = [F.grid_sample(self.feat[i], 2*coord[:, None]-1., align_corners=True)[:, :, 0] for i in range(3)]
        ret = [feat.prod(dim=0).sum(dim=0) for feat in feats ]
        delta_xyz = torch.stack(ret, dim=1)
        if B is None:
            return delta_xyz
        else:
            return delta_xyz.reshape(B, N_samples, -1)

    def compute_tv_loss(self, xyz, t, wbounds):
        inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        time = t[0, 0].item()
        if time == 0.:
            delta_xyz_prev = self.compute_delta(inputs, t)
            delta_xyz_next = self.compute_delta(inputs, t + 1.)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        else:
            delta_xyz_prev = self.compute_delta(inputs, t - 1.)
            delta_xyz_next = self.compute_delta(inputs, t)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        return delta

class DNeRFNGP_MLP(nn.Module):
    def __init__(self, skips=[4,], **kwargs):
        super(DNeRFNGP_MLP, self).__init__()
        self.encoder = HashGrid(**kwargs)
        self.out_dim = self.encoder.out_dim

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

        skips = [4,]
        self.skips = skips
        input_ch = xyz_ch + time_ch
        self.time_mlp = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.time_out = nn.Linear(W, 3)

    def forward(self, x, wbounds=None):
        xyz, t = x[..., :3], x[..., 3:]
        inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item())
        if t[0] == -1.:
            return self.encoder(torch.clamp(inputs, min=0., max=1. - eps), normalize=False)
        else:
            delta_xyz = self.compute_delta(inputs, t)
            return self.encoder(torch.clamp(inputs+delta_xyz, min=0., max=1. - eps), normalize=False)

    def compute_delta(self, xyz, t):
        if len(xyz.shape) == 3:
            B, N_samples = xyz.shape[:2]
            xyz, t = xyz.reshape(-1, xyz.shape[-1]), t.reshape(-1, t.shape[-1])
        else:
            B = None
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

        if B is None:
            return delta_xyz
        else:
            return delta_xyz.reshape(B, N_samples, -1)

    def compute_tv_loss(self, xyz, t, wbounds):
        inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        time = t[0, 0].item()
        if time == 0.:
            delta_xyz_prev = self.compute_delta(inputs, t)
            delta_xyz_next = self.compute_delta(inputs, t + 1.)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        else:
            delta_xyz_prev = self.compute_delta(inputs, t - 1.)
            delta_xyz_next = self.compute_delta(inputs, t)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        return delta

class DNeRFTensoRF(nn.Module):
    def __init__(self, skips=[4,], **kwargs):
        super(DNeRFTensoRF, self).__init__()
        freq = kwargs['freq']
        encoder_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : freq-1,
                'num_freqs' : freq,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
        }
        self.encoder = FreqEncoder(**encoder_kwargs)
        F, reso = 64, 256
        self.feat = nn.ParameterList([nn.Parameter(0.1 * torch.randn((3, F, cfg.num_frames, reso))) for i in range(3)])
        self.out_dim = self.encoder.out_dim

    def forward(self, x, wbounds=None):
        inputs, t = x[..., :3], x[..., 3:]
        if t[0] == -1.:
            __import__('ipdb').set_trace()
        else:
            delta_xyz = self.compute_delta(inputs, t)
            return self.encoder.embed(inputs+delta_xyz)

    def compute_delta(self, xyz, t):
        if len(xyz.shape) == 3:
            B, N_samples = xyz.shape[:2]
            xyz, t = xyz.reshape(-1, xyz.shape[-1]), t.reshape(-1, t.shape[-1])
        else:
            B = None
        t = t / (cfg.num_frames - 1)

        coord = torch.stack([torch.cat([xyz[..., i:i+1], t], dim=-1) for i in range(3)])
        feats = [F.grid_sample(self.feat[i], 2*coord[:, None]-1., align_corners=True)[:, :, 0] for i in range(3)]
        ret = [feat.prod(dim=0).sum(dim=0) for feat in feats ]
        delta_xyz = torch.stack(ret, dim=1)
        if B is None:
            return delta_xyz
        else:
            return delta_xyz.reshape(B, N_samples, -1)

    def compute_tv_loss(self, xyz, t, wbounds):
        inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        time = t[0, 0].item()
        if time == 0.:
            delta_xyz_prev = self.compute_delta(inputs, t)
            delta_xyz_next = self.compute_delta(inputs, t + 1.)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        else:
            delta_xyz_prev = self.compute_delta(inputs, t - 1.)
            delta_xyz_next = self.compute_delta(inputs, t)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        return delta


