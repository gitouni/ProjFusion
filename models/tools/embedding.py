# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Tuple
from einops import repeat

class RoPE2D(nn.Module):
    def __init__(self, dim: int, q_coord_y: torch.Tensor, q_coord_x: torch.Tensor, base_freq: int = 1000):
        """
        dim: 每个 head 的维度
        num_heads: 注意力头数

        """
        super().__init__()
        assert dim % 4 == 0, f"head_dim {dim} must be divisible by 4 for 2D RoPE"
        self.dim = dim
        self.base_freq = base_freq

        # 分别处理 y/x 方向
        self.register_buffer('inv_hfreq', math.pi / (self.base_freq ** (torch.arange(0, dim // 2, 2).float() / (dim // 2))))  # D // 4
        self.register_buffer('inv_wfreq', math.pi / (self.base_freq ** (torch.arange(0, dim // 2, 2).float() / (dim // 2))))  # D // 4

        # 分别处理 y/x 方向的sin和cos
        q_sin_y, q_cos_y = self._compute_sin_cos_from_pos(q_coord_y, self.inv_hfreq)  # (H*W, D // 4)
        q_sin_x, q_cos_x = self._compute_sin_cos_from_pos(q_coord_x, self.inv_wfreq)  # (H*W, D // 4)
        self.register_buffer('q_sin_y', q_sin_y)
        self.register_buffer('q_cos_y', q_cos_y)
        self.register_buffer('q_sin_x', q_sin_x)
        self.register_buffer('q_cos_x', q_cos_x)

    @staticmethod
    def _compute_sin_cos_from_grid(length: int, inv_freq: int):
        """为某一维度 (H or W) 生成sin和cos位置编码"""
        pos = torch.arange(length).float()
        freqs = torch.einsum("i,j->ij", pos, inv_freq)  # (length, dim/4)
        return freqs.sin(), freqs.cos()  # shape: (length, dim/4)

    @staticmethod
    def _compute_sin_cos_from_pos(pos: torch.Tensor, inv_freq: torch.Tensor):
        """
        pos: Tensor of shape (...,) 表示任意位置坐标（如 [0.5, -0.8]）
        inv_freq: 预设频率张量
        返回: sin, cos 编码张量
        """
        freqs = torch.einsum("...i,j->...j", pos.unsqueeze(-1), inv_freq)  # [..., D//4], inv_freq作用在每个坐标上
        return freqs.sin(), freqs.cos()  # (..., D // 4), (..., D // 4)，最后一个维度代表不同频率的坐标正余弦值

    def forward(self, q: torch.Tensor, k: torch.Tensor, k_coord_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对任意坐标的 tokens 应用 2D RoPE 编码
        q, k: (B, num_heads, N, head_dim)
        k_coord_xy: (B, N, 2) 表示每个 token 的 kv对应的 (y, x) 坐标，范围建议 [-1, 1]
        返回: 经过 RoPE 编码后的 q, k
        """
        def rotate_half(x: torch.Tensor):
            ori_shape = x.shape
            x = x.view(*x.shape[:-1], -1, 2)
            x = torch.stack([-x[..., 1], x[..., 0]], dim=-1)  # cos * x1 - sin * x2, cos * x1 + sin * x2
            return x.view(*ori_shape)

        def rope(x1: torch.Tensor, x2: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
            x1_rot = x1 * cos + rotate_half(x2) * sin
            x2_rot = x2 * cos - rotate_half(x1) * sin
            return x1_rot, x2_rot
        
        B, _, N, d = k.shape
        assert k_coord_xy.shape == (B, N, 2), f"coords shape should be ({B}, {N}, 2), got {k_coord_xy.shape}"
        assert d == self.dim, f'head dim of q is {d} != {self.dim}'

        # 计算 sin/cos
        k_sin_y, k_cos_y = self._compute_sin_cos_from_pos(k_coord_xy[..., 1], self.inv_hfreq)  # (B, N, D // 4)
        k_sin_x, k_cos_x = self._compute_sin_cos_from_pos(k_coord_xy[..., 0], self.inv_wfreq)  # (B, N, D // 4)
        k_sin_y, k_cos_y, k_sin_x, k_cos_x = map(lambda x: x.unsqueeze(1), [k_sin_y, k_cos_y, k_sin_x, k_cos_x])  # (B, N, D // 4) -> (B, 1, N, D // 4)
        q_sin_y, q_cos_y, q_sin_x, q_cos_x = map(lambda x: x[None, None, ...], [self.q_sin_y, self.q_cos_y, self.q_sin_x, self.q_cos_x])  # (H*W, D // 4) -> (1, 1, H*W, D // 4)
        # Split q and k
        q1y, q1x, q2y, q2x = q.chunk(4, dim=-1)  # (B, H, N, D // 4)
        k1y, k1x, k2y, k2x = k.chunk(4, dim=-1)  # (B, H, N, D // 4)

        # Apply RoPE
        q1y_, q2y_ = rope(q1y, q2y, q_sin_y, q_cos_y)  # (B, H, N, D // 4) q使用网格化的位置编码
        k1y_, k2y_ = rope(k1y, k2y, k_sin_y, k_cos_y)  # (B, H, N, D // 4) k使用的是投影坐标位置编码

        q1x_, q2x_ = rope(q1x, q2x, q_sin_x, q_cos_x)  # (B, H, N, D // 4) q使用网格化的位置编码
        k1x_, k2x_ = rope(k1x, k2x, k_sin_x, k_cos_x)  # (B, H, N, D // 4) k使用的是投影坐标位置编码
 
        # Concatenate back
        q1 = torch.cat([q1y_, q1x_], dim=-1)  # (B, H, N, D // 2)
        q2 = torch.cat([q2y_, q2x_], dim=-1)  # (B, H, N, D // 2)
        k1 = torch.cat([k1y_, k1x_], dim=-1)  # (B, H, N, D // 2)
        k2 = torch.cat([k2y_, k2x_], dim=-1)  # (B, H, N, D // 2)

        q = torch.cat([q1, q2], dim=-1)  # (B, H, N, D)
        k = torch.cat([k1, k2], dim=-1)  # (B, H, N, D)

        return q, k  # (B, H, N, D), (B, H, N, D)
    
# copied from pytorch3d
class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        The harmonic embedding layer supports the classical
        Nerf positional encoding described in
        `NeRF <https://arxiv.org/abs/2003.08934>`_
        and the integrated position encoding in
        `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.

        During the inference you can provide the extra argument `diag_cov`.

        If `diag_cov is None`, it converts
        rays parametrized with a `ray_bundle` to 3D points by
        extending each ray according to the corresponding length.
        Then it converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]

        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.


        If `diag_cov is not None`, it approximates
        conical frustums following a ray bundle as gaussians,
        defined by x, the means of the gaussians and diag_cov,
        the diagonal covariances.
        Then it converts each gaussian
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                sin(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),
                ...
                sin(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                cos(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                cos(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),,
                ...
                cos(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                x[..., i],              # only present if append_input is True.
            ]

        where N equals `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.

        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (embed.sin(), embed.cos(), x)
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=True)
        self.register_buffer(
            "_zero_half_pi", torch.tensor([0.0, 0.5 * torch.pi]), persistent=True
        )
        self.append_input = append_input

    def forward(
        self, x: torch.Tensor, diag_cov: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
            diag_cov: An optional tensor of shape `(..., dim)`
                representing the diagonal covariance matrices of our Gaussians, joined with x
                as means of the Gaussians.

        Returns:
            embedding: a harmonic embedding of `x` of shape
            [..., (n_harmonic_functions * 2 + int(append_input)) * num_points_per_ray]
        """
        # [..., dim, n_harmonic_functions]
        embed = x[..., None] * self._frequencies
        # [..., 1, dim, n_harmonic_functions] + [2, 1, 1] => [..., 2, dim, n_harmonic_functions]
        embed = embed[..., None, :, :] + self._zero_half_pi[..., None, None]
        # Use the trig identity cos(x) = sin(x + pi/2)
        # and do one vectorized call to sin([x, x+pi/2]) instead of (sin(x), cos(x)).
        embed = embed.sin()
        if diag_cov is not None:
            x_var = diag_cov[..., None] * torch.pow(self._frequencies, 2)
            exp_var = torch.exp(-0.5 * x_var)
            # [..., 2, dim, n_harmonic_functions]
            embed = embed * exp_var[..., None, :, :]

        embed = embed.reshape(*x.shape[:-1], -1)

        if self.append_input:
            return torch.cat([embed, x], dim=-1)
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int,
        n_harmonic_functions: int,
        append_input: bool,
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.

        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )


class TimeStepEmbedding(nn.Module):
    # learned from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py
    def __init__(self, dim=256, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        self.linear = nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim // 2))

        self.out_dim = dim // 2

    def _compute_freqs(self, half):
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        return freqs

    def forward(self, timesteps:torch.Tensor):
        half = self.dim // 2
        freqs = self._compute_freqs(half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]  # (N, 1) * (N, 2) -> (N, 2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        output = self.linear(embedding)
        return output


class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(n_harmonic_functions=n_harmonic_functions, append_input=append_input)

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding

class PositionEmbeddingCoordsSine2D(nn.Module):
    def __init__(self, d_model:int, height:int, width:int, margin:float, temperature=10000):
        """2D Positional Embedding 

        Args:
            d_model (int): embedding dim
            height (int): image height
            width (int): image width
            margin (float): threshold margin (projected points may exceed the bounding box of the image)
            temperature (int, optional): hyperparmeter for embedding. Defaults to 10000.
        """
        super().__init__()
        n_dim = 2  # 2D positional embedding
        self.d_model = d_model
        self.margin = margin
        self.height, self.width = height, width
        self.height_offset = height * self.margin
        self.width_offset = width * self.margin
        self.temperature = temperature
        assert d_model % (2 * n_dim) == 0, 'd_model ({}) % (2 * n_dim ({})) != 0'.format(d_model, n_dim)
        # Each dimension use half of d_model
        self.d_model_dim = d_model // n_dim
        self.div_term = torch.exp(torch.arange(0., self.d_model_dim, 2) * (-math.log(temperature) / self.d_model_dim))  # 2 means sin and cos

    def get_patched_coordinate_embedding(self, flatten=False):
        pos_w = (torch.arange(0., self.width)[:, None].to(self.div_term) + self.width_offset) * self.div_term[None,:]  # (w, 1) * (1, d) -> (w, d)
        pos_h = (torch.arange(0., self.height)[:, None].to(self.div_term) + self.height_offset) * self.div_term[None,:]   # (h, 1) * (1, d) -> (h, d)
        pe = torch.zeros(self.height, self.width, self.d_model).to(self.div_term)  # (D, H, W) take n_dim=2 as an example
        pe[..., 0:self.d_model_dim:2] = repeat(pos_w.sin()[None, :, :], '1 w d -> h w d', h=self.height)
        pe[..., 1:self.d_model_dim:2] = repeat(pos_w.cos()[None, :, :], '1 w d -> h w d', h=self.height)
        pe[..., self.d_model_dim::2] = repeat(pos_h.sin()[:, None, :], 'h 1 d -> h w d', w=self.width)
        pe[..., self.d_model_dim+1::2] = repeat(pos_h.cos()[:, None, :], 'h 1 d -> h w d', w=self.width)
        if flatten:
            pe = torch.flatten(pe, start_dim=0, end_dim=1)  # end_dim: closed region
        return pe # h,w,d

    def interpolate_pe(self, pcd:torch.Tensor):
        N = pcd.shape[0]
        pe = torch.zeros(N, self.d_model).to(pcd)  # (N, 2d)
        div_term = self.div_term[None, :].to(pcd)
        pos_x = (pcd[:,[0]] + self.width_offset) * div_term  # (N, 1) * (1, d) -> (N, d)
        pos_y = (pcd[:,[1]] + self.height_offset) * div_term  # (N, 1) * (1, d) -> (N, d)
        pe[:,0:self.d_model_dim:2] = pos_x.sin() # (B, N/2, d): N/2 -> [0, N]
        pe[:,1:self.d_model_dim:2] = pos_x.cos() # (B, N/2, d): N/2 -> [0, N]
        pe[:,self.d_model_dim::2] = pos_y.sin() # (B, N/2, d): N/2 -> [N+1, 2N]
        pe[:,self.d_model_dim+1::2] = pos_y.cos()  # (B, N/2, d): N/2 -> [N+1, 2N]
        return pe  # (N, d)

    def batched_interpolate_pe(self, pcd:torch.Tensor, dim_first=False):
        if dim_first:  # pcd: (B, D, N)
            B, N = pcd.shape[0], pcd.shape[2]
        else:  # pcd: (B, N, D)
            B, N = pcd.shape[:2]
        pe = torch.zeros(B, N, self.d_model).to(pcd)  # (B, N, 2d)
        div_term = self.div_term[None, None, :].to(pcd)  # (1, 1, d)
        pos_x = (pcd[...,[0]] + self.width_offset) * div_term  # (B, N, 1) * (1, 1, d) -> (B, N, d)
        pos_y = (pcd[...,[1]] + self.height_offset) * div_term  # (B, N, 1) * (1, 1, d) -> (B, N, d)
        pe[...,0:self.d_model_dim:2] = pos_x.sin()  # (B, N/2, d): N/2 -> [0, N]
        pe[...,1:self.d_model_dim:2] = pos_x.cos()  # (B, N/2, d): N/2 -> [0, N]
        pe[...,self.d_model_dim::2] = pos_y.sin()  # (B, N/2, d): N/2 -> [N+1, 2N]
        pe[...,self.d_model_dim+1::2] = pos_y.cos()  # (B, N/2, d): N/2 -> [N+1, 2N]
        return pe  # (B, N, d)
    

def positionalencoding2d(d_model:int, height:int, width:int):
    """
    :param d_model:int dimension of the model
    :param height:int height of the positions
    :param width:int width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe