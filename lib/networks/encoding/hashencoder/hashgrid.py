import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from lib.config import cfg
import torch.nn.functional as F

from .backend import _backend

eps = 1e-6

class _hash_encode(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    #@custom_fwd
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous().to(inputs.device)

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.zeros(L, B, C, device=inputs.device, dtype=inputs.dtype)

        if calc_grad_inputs:
            dy_dx = torch.zeros(B, L * D * C, device=inputs.device, dtype=inputs.dtype)
        else:
            dy_dx = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, calc_grad_inputs, dy_dx)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, L * C]

        grad = grad.contiguous()

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs)

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None, None
        else:
            return None, grad_embeddings, None, None, None, None


hash_encode = _hash_encode.apply


class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=-1, **kwargs):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        __import__('ipdb').set_trace()
        if desired_resolution != -1:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.out_dim = self.output_dim

        # self.bounds = torch.tensor(np.array(bbox).reshape((2, input_dim))).float().cuda()
        # self.size = (self.bounds[1] - self.bounds[0]).max().item()
        # self.bounds[1] = self.bounds[1] - eps # [0, 1)

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # allocate parameters
        self.offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution + 1) ** input_dim) # limit max number
            params_in_level = int(params_in_level / 8) * 8 # make divisible
            self.offsets.append(offset)
            offset += params_in_level
        self.offsets.append(offset)
        self.offsets = torch.from_numpy(np.array(self.offsets, dtype=np.int32))

        self.n_params = self.offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.zeros(offset, level_dim))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} H={self.base_resolution} params={self.embeddings.shape}"

    def forward(self, xyz, wbounds=None, normalize=True):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]
        if normalize:
            inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
            inputs = inputs - wbounds[None][:, :3]
            inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        else:
            inputs = xyz

        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = hash_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        # return torch.cat([xyz, outputs], dim=-1)
        return outputs

class TriPlane(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.xy_plane = HashEncoder(**kwargs)
        self.yz_plane = HashEncoder(**kwargs)
        self.xz_plane = HashEncoder(**kwargs)
        self.out_dim = self.xy_plane.out_dim * 3

    def forward(self, xyz, wbounds):
        inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)

        xy_feat = self.xy_plane(inputs[..., [0, 1]], normalize=False)
        yz_feat = self.yz_plane(inputs[..., [1, 2]], normalize=False)
        xz_feat = self.xz_plane(inputs[..., [0, 2]], normalize=False)
        return torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)

class Motion2d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.xy_plane = HashEncoder(**kwargs)
        self.yz_plane = HashEncoder(**kwargs)
        self.xz_plane = HashEncoder(**kwargs)
        self.mlp = nn.Sequential(*[
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Sigmoid()
            ])
        self.out_dim = self.xy_plane.out_dim * 3

    def forward(self, xyzt, wbounds):
        inputs = torch.clamp(xyzt[..., :3], min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        inputs = torch.cat([inputs, xyzt[:, 3:] / (cfg.num_frames - 1)], dim=-1)

        if xyzt[0, 3] != 0:
            delta_xyz =  self.mlp(inputs)
            xyz = torch.clamp(inputs[..., :3] + 2*delta_xyz - 1., min=0., max=1.)
            xy_feat = self.xy_plane(xyz[..., [0, 1]], normalize=False)
            yz_feat = self.yz_plane(xyz[..., [1, 2]], normalize=False)
            xz_feat = self.xz_plane(xyz[..., [0, 2]], normalize=False)
        else:
            xy_feat = self.xy_plane(inputs[..., [0, 1]], normalize=False)
            yz_feat = self.yz_plane(inputs[..., [1, 2]], normalize=False)
            xz_feat = self.xz_plane(inputs[..., [0, 2]], normalize=False)

        return torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)

class HashLatent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        latent_dim = 32
        self.latent_t = nn.Parameter(torch.zeros((cfg.num_frames, latent_dim)))
        std = 1e-4
        self.latent_t.data.uniform_(-std, std)
        self.hashencoder = HashEncoder(**kwargs)
        self.out_dim = self.hashencoder.out_dim + latent_dim

    def forward(self, inputs, wbounds):
        xyz = inputs[:, :3]
        xyz_feat = self.hashencoder(xyz, wbounds)
        t_feat = self.latent_t[inputs[:, 3].long()]
        return torch.cat([xyz_feat, t_feat], dim=-1)


class HashEncoder4d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hashencoder = HashEncoder(**kwargs)
        self.out_dim = self.hashencoder.out_dim

    def forward(self, inputs, wbounds):
        xyz = inputs[:, :3]
        xyz = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        xyz = xyz - wbounds[None][:, :3]
        xyz = xyz / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        inputs = torch.cat([xyz, inputs[:, 3:] / cfg.num_frames], dim=-1)
        return self.hashencoder(inputs, normalize=False)


class HashEncoderCoef(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.basis_num = 6
        self.basis = nn.ModuleList([HashEncoder(**kwargs) for i in range(self.basis_num)])
        kwargs['input_dim'], kwargs['log2_hashmap_size'] = 4, 20
        self.coefs = HashEncoder(**kwargs)
        self.coefs_mlp = nn.ModuleList([
            nn.Linear(self.coefs.out_dim, 64),
            nn.Linear(64, self.basis_num)
            ])
        self.out_dim = self.basis[0].out_dim

    def forward(self, inputs, wbounds):
        xyz = inputs[:, :3]
        xyz = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        xyz = xyz - wbounds[None][:, :3]
        xyz = xyz / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        inputs = torch.cat([xyz, inputs[:, 3:] / cfg.num_frames], dim=-1)

        coefs_emb = self.coefs(inputs, normalize=False)
        for i in range(len(self.coefs_mlp) - 1):
            coefs_emb = self.coefs_mlp[i](coefs_emb)
            coefs_emb = F.relu(coefs_emb)
        coefs = F.softmax(self.coefs_mlp[-1](coefs_emb), dim=-1)
        embs = []
        for i in range(self.basis_num):
            embs.append(self.basis[i](xyz, normalize=False))
        ret = (torch.stack(embs, dim=1) * coefs[..., None]).sum(dim=1)
        return ret


class DNeRFNGP(nn.Module):
    def __init__(self, skips=[4,], **kwargs):
        super(DNeRFNGP, self).__init__()
        self.encoder = HashEncoder(**kwargs)
        F, reso = 64, 256
        self.feat = nn.ParameterList([nn.Parameter(0.1 * torch.randn((3, F, cfg.num_frames, reso))) for i in range(3)])
        self.out_dim = self.encoder.out_dim

    def forward(self, x, wbounds=None):
        xyz, t = x[..., :3], x[..., 3:]
        inputs = torch.clamp(xyz, min=wbounds[:3], max=wbounds[3:6])
        inputs = inputs - wbounds[None][:, :3]
        inputs = inputs / ((wbounds[3:6] - wbounds[:3]).max().item() + eps)
        if t[0] == 0.:
            return self.encoder(inputs, normalize=False)
        else:
            delta_xyz = self.compute_delta(inputs, t)
            return self.encoder(torch.clamp(inputs+delta_xyz, min=0., max=1.), normalize=False)

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
            delta = self.compute_delta(inputs, t)
            delta = (delta).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        else:
            delta_xyz_prev = self.compute_delta(inputs, t - 1.)
            delta_xyz_next = self.compute_delta(inputs, t)
            delta = (delta_xyz_next - delta_xyz_prev).pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        return delta


