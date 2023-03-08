import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
eps = 1e-6


class Plane(nn.Module):
    def __init__(self, input_dim=2, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=-1, **kwargs):
        super(Plane, self).__init__()
        self.n_levels = num_levels
        self.n_entrys_per_level = 2**log2_hashmap_size
        while True:
            if isprime(self.n_entrys_per_level):
                break
            else:
                self.n_entrys_per_level += 1
        if desired_resolution != -1:
            self.b = (desired_resolution / base_resolution) ** (1 / (num_levels - 1))
        else:
            self.b = per_level_scale
        self.base_resolution = base_resolution
        self.f = level_dim
        self.output_dim = self.f * self.n_levels
        self.out_dim = self.output_dim
        self.offsets = [0]
        self.scales = []
        self.start_hash = -1
        for i in range(self.n_levels):
            res = int((self.base_resolution) * (self.b**i))
            self.scales.append(res)
            n_entrys = int((res + 1) ** 2)
            if n_entrys > self.n_entrys_per_level:
                if self.start_hash < 0:
                    self.start_hash = i
                n_entrys = self.n_entrys_per_level
            self.offsets.append(self.offsets[-1] + n_entrys)

        self.data = nn.Parameter(torch.zeros((self.offsets[-1], self.f)))
        std = 1e-4
        self.data.data.uniform_(-std, std)
        self.ps = [1, 19349663, 83492791]

        self.offsets_pos = torch.tensor([[0., 0.],
                                         [0., 1.],
                                         [1., 0.],
                                         [1., 1.]]).float().cuda()

        self.scales = torch.tensor(self.scales).cuda().float()
        self.offsets = torch.tensor(np.array(self.offsets)).cuda().long()


    def forward(self, x):
        x = x[None].repeat(self.n_levels, 1, 1)
        float_x = x * self.scales[:, None, None]
        int_x = (float_x[:, :, None] + self.offsets_pos[None, None]).long()
        offset_x = float_x - int_x[:, :, 0]

        ind = torch.zeros_like(int_x[..., 0])
        if self.start_hash == -1:
            sh = self.n_levels
        else:
            sh = self.start_hash

        ind[:sh] = int_x[:sh, ..., 0] * ((self.scales[:sh] + 1))[:, None, None] + \
                   int_x[:sh, ..., 1]
        nl = self.n_levels
        if self.start_hash != -1:
            ind[sh:nl] = torch.bitwise_xor(int_x[sh:nl, ..., 0]*self.ps[0], int_x[sh:nl, ..., 1]*self.ps[1]) % self.n_entrys_per_level

        ind = ind.reshape(nl, -1)
        ind += self.offsets[:-1, None]
        ind = ind.reshape(-1)
        val = torch.gather(self.data, 0, ind[:, None].repeat(1, self.f))
        val = val.reshape(nl, -1, 4, self.f)

        weights_x = torch.clamp((1 - self.offsets_pos[None, None]) + (2 * self.offsets_pos[None, None] - 1.) * offset_x[:, :, None], min=0., max=1.)
        weights_x = weights_x[..., 0] * weights_x[..., 1]
        val = (weights_x[..., None] * val).sum(dim=-2)
        val = val.permute(1, 0, 2).reshape(-1, nl*self.f)
        return val



class TriPlane(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.xy_plane = Plane(**kwargs)
        self.yz_plane = Plane(**kwargs)
        self.xz_plane = Plane(**kwargs)
        self.out_dim = self.xy_plane.out_dim * 3

    def forward(self, xyz, wbounds):
        inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)

        xy_feat = self.xy_plane(inputs[..., [0, 1]])
        yz_feat = self.yz_plane(inputs[..., [1, 2]])
        xz_feat = self.xz_plane(inputs[..., [0, 2]])
        return torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)

