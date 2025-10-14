# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os.path
from typing import Literal, List, Callable
from contextlib import contextmanager

import torch
import torch.nn as nn
from typing import Iterable, Literal, Optional, Tuple, TypeVar

from einops import rearrange
import math
import torch.nn.functional as F
from .activation import SwiGLU
from .embedding import RoPE2D

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d



def normalize_grid(grid:torch.Tensor, N:int) -> torch.Tensor:
    """normalize grid to [-1, 1]"""
    return 2 * grid.float() / (N - 1) - 1

def coord_2d_mesh(h:int, w:int, normalize:bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """compute coordinates of pixels

    Args:
        h (int): height
        w (int): width
        normalize (bool, optional). whether to normalize grid to [-1,1] Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: xgrid, ygrid (H, W)
    """
    xi = torch.arange(w)
    yi = torch.arange(h)
    xgrid, ygrid = torch.meshgrid(xi, yi, indexing='xy')  # (H, W)
    if normalize:
        xgrid = normalize_grid(xgrid, w)
        ygrid = normalize_grid(ygrid, h)
    return xgrid, ygrid  # (H, W), (H, W)

def patchify_coords(xgrid:torch.Tensor, ygrid:torch.Tensor) -> torch.Tensor:
    """Patchify grids

    Args:
        xgrid (torch.Tensor): (H, W)
        ygrid (torch.Tensor): (H, W)

    Returns:
        torch.Tensor: (H*W, 2)
    """
    coords = torch.stack([xgrid, ygrid], dim=-1)  # (H, W), (H, W) -> (H, W, 2)
    coords = coords.view(-1, -2)  # (H*W, 2)
    return coords

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            assert img_size % patch_size == 0
            num_patches = (img_size // patch_size) * (img_size // patch_size)
        elif isinstance(img_size, Iterable):
            assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            raise NotImplementedError("unrecognized img_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """patchify a image into patches

        Args:
            x (torch.Tensor): B, C, H, W

        Returns:
            torch.Tensor: (B, N, D)
        """
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, D, H, W) -> (B, D, H*W) -> (B, H*W, D)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, modelname:str, image_hw:Tuple[int,int], cache_dir="/home/bit/.cache/torch/hub/",
                source:Literal['github','local']='local', reshape:bool=False):
        super().__init__()
        if "dinov2" in modelname:
            self._net = torch.hub.load(os.path.join(cache_dir,"facebookresearch_dinov2_main"), modelname, source=source)  # facebookresearch/dinov2
            self._output_dim = self._net.norm.weight.shape[0]
        elif "dino" in modelname:
            self._net = torch.hub.load(os.path.join("facebookresearch_dino_main"), modelname,  source=source)  # facebookresearch/dino:main
            self._output_dim = self._net.norm.weight.shape[0]
        else:
            raise ValueError(f"Unknown model name {modelname}")
        self.patch_size = self._net.patch_size
        self.num_patches = (image_hw[0] // self.patch_size) * (image_hw[1] // self.patch_size)
        self.reshape = reshape

    def custom_input_dim(self, input_dim:int):
        patch_embed_proj:nn.Conv2d = self._net.patch_embed
        proj_argv = dict(in_channels=input_dim,
                         out_channels=patch_embed_proj.out_channels,
                         kernel_size=patch_embed_proj.kernel_size,
                         stride=patch_embed_proj.stride,
                         padding=patch_embed_proj.padding)
        self._net.patch_embed = nn.Conv2d(**proj_argv)

    def get_output_dim(self) -> int:
        return self._output_dim

    # refer to https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L298C5-L322C30
    def forward(self, image_rgb: torch.Tensor) -> torch.Tensor:
        features = self._net.get_intermediate_layers(image_rgb, reshape=self.reshape)[-1]  # reshape True: B, C, H, W; reshape False: B, N, D
        return features  # (B, N, D)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma
    
class RMSNormSingleHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))  # pytorch乘法会自动往前扩展

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma


# feedforward

def FeedForward(dim, hidden_dim, out_dim=None, dropout = 0., activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
    if out_dim is None:
        out_dim = dim
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        activation_fn(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.Dropout(dropout)
    )

# Modified From https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/na_vit.py
class Attention(nn.Module):
    def __init__(self, q_input_dim:int, kv_input_dim:int, heads = 8, dim_head = 64, dropout = 0., **argv):
        super().__init__()
        inner_dim = dim_head * heads
        self.out_dim = inner_dim
        self.heads = heads
        self.norm = LayerNorm(q_input_dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.dropout_p = dropout

        self.to_q = nn.Linear(q_input_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(kv_input_dim, inner_dim * 2, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, inner_dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        **argv
    ):
        """attention rope

        Args:
            x (torch.Tensor): (B, N, D) or (B*G, N, D)
            context (torch.Tensor): (B, N, D) or (B*G, N, D).
            attn_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if exists(attn_mask):
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]
            elif attn_mask.ndim == 4:
                pass
            else:
                raise NotImplementedError("attn_mask.ndim must be 2,3,4, got {}".format(attn_mask.ndim))
        x = self.norm(x)  # (B, N, D)
        kv_input = default(context, x)
        kv: torch.Tensor = self.to_kv(kv_input)
        q = self.to_q(x)
        k, v = kv.chunk(2, dim=-1)  # (B, M, D), (B, M, D)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [q, k, v])  # (B, N, D) -> (B, H, N, d)
        
        q: torch.Tensor = self.q_norm(q)  # (B, H, N, d)
        k: torch.Tensor = self.k_norm(k)  # (B, H, N, d)
        out = F.scaled_dot_product_attention(q, k, v,
            attn_mask = attn_mask,
            dropout_p = (self.dropout_p if self.training else 0.0),
            scale=1.0)  # (B, N, H, D)
        out = rearrange(out, 'b h n d -> b n (h d)')  # head merging
        return self.to_out(out)  # (B*G, N, D)


class AttentionRoPE(Attention):
    def __init__(self, q_input_dim:int, kv_input_dim: int, height: int, width: int, base_freq: float, heads = 8, dim_head = 64, dropout = 0., **argv):
        super().__init__(q_input_dim, kv_input_dim, heads, dim_head, dropout)
        self.q_coord_x, self.q_coord_y = coord_2d_mesh(height, width, normalize=True)  # (H*W,), (H*W, )
        self.rope = RoPE2D(dim_head, self.q_coord_y.flatten(), self.q_coord_x.flatten(), base_freq=base_freq)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        k_coord_xy: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """attention rope

        Args:
            x (torch.Tensor): (B, N, D)
            context (Optional[torch.Tensor], optional): (B, N, D).
            k_coord_xy (torch.Tensor): (B, N, 2).
            attn_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if exists(attn_mask):
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]
            elif attn_mask.ndim == 4:
                pass
            else:
                raise NotImplementedError("attn_mask.ndim must be 2,3,4, got {}".format(attn_mask.ndim))
        x = self.norm(x)  # (B, N, D)
        kv_input = default(context, x)
        kv: torch.Tensor = self.to_kv(kv_input)
        q = self.to_q(x)
        k, v = kv.chunk(2, dim=-1)  # (B, M, D), (B, M, D)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [q, k, v])  # (B, N, D) -> (B, H, N, d)
        
        q: torch.Tensor = self.q_norm(q)  # (B, H, N, d)
        k: torch.Tensor = self.k_norm(k)  # (B, H, N, d)
        q, k = self.rope(q, k, k_coord_xy)
        out = F.scaled_dot_product_attention(q, k, v,
            attn_mask = attn_mask,
            dropout_p = (self.dropout_p if self.training else 0.0),
            scale=1.0)  # (B, N, H, D)
        out = rearrange(out, 'b h n d -> b n (h d)')  # head merging
        return self.to_out(out)  # (B*G, N, D)

class Transformer(nn.Module):
    def __init__(self, inplanes, depth, heads, dim_head, mlp_dim, dropout = 0., activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True), ffn_layer:Literal['swiglu,mlp']='mlp'):
        super().__init__()
        self.norm = LayerNorm(inplanes)
        self.layers = nn.ModuleList([])
        if ffn_layer == 'swiglu':
            ffn_module_fn = lambda: SwiGLU(inplanes, mlp_dim, inplanes)
        elif ffn_layer == 'mlp':
            ffn_module_fn = lambda: FeedForward(inplanes, mlp_dim, dropout = dropout, activation_fn=activation_fn)
        else:
            raise NotImplementedError("ffn_layer must be \"swiglu\" or \"mlp\"")
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(inplanes, heads = heads, dim_head = dim_head, dropout = dropout),
                ffn_module_fn(),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:  # pre-norm
            x = attn(x) + x  # norm-first in the block
            x = ff(x) + x  # norm-first in the block

        return self.norm(x)  # last layer normalization


class SelfCrossTransformer(nn.Module):
    def __init__(self, layer_names:List[Literal['cross','self']], inplanes, heads, dim_head, mlp_dim, dropout = 0., activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True), ffn_layer:Literal['swiglu,mlp']='mlp'):
        super().__init__()
        self.check_layer_names(layer_names)
        depth = len(layer_names)
        self.layer_names = layer_names
        self.layers = nn.ModuleList([])
        if ffn_layer == 'swiglu':
            ffn_module_fn = lambda: SwiGLU(inplanes, mlp_dim, inplanes)
        elif ffn_layer == 'mlp':
            ffn_module_fn = lambda: FeedForward(inplanes, mlp_dim, dropout = dropout, activation_fn=activation_fn)
        else:
            raise NotImplementedError("ffn_layer must be \"swiglu\" or \"mlp\"")
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(inplanes, heads = heads, dim_head = dim_head, dropout = dropout),
                ffn_module_fn(),
            ]))

    @staticmethod
    def check_layer_names(layer_names:List[str]):
        for name in layer_names:
            if name not in ['cross','self']:
                raise TypeError('layer_name must be \"cross\" or \"self\"')

    def forward(self, x:torch.Tensor, y:torch.Tensor,
            x_pe:Optional[torch.Tensor]=None, y_pe:Optional[torch.Tensor]=None,
            cross_mask:Optional[torch.Tensor]=None, self_mask:Optional[torch.Tensor]=None):
        if cross_mask is not None:
            cross_mask_T = cross_mask.transpose(-1,-2).contiguous()  # (B, N, M) -> (B, M, N)
        else:
            cross_mask_T = None
        if x_pe is not None:
            if x.shape[1] != x_pe.shape[1]:
                assert x.shape[1] > x_pe.shape[1], 'cannot do pe expansion'
                x_pe = torch.cat([torch.zeros([x_pe.shape[0],x.shape[1]-x_pe.shape[1],x_pe.shape[2]]).to(x_pe), x_pe], dim=1)
        if y_pe is not None:
            if y.shape[1] != y_pe.shape[1]:
                assert y.shape[1] > y_pe.shape[1], 'cannot do pe expansion'
                y_pe = torch.cat([torch.zeros([y_pe.shape[0],y.shape[1]-y_pe.shape[1],y_pe.shape[2]]).to(y_pe), y_pe], dim=1)
        for name, layer in zip(self.layer_names, self.layers):
            attn, ffn = layer
            if name == 'cross':
                dx, dy = x + attn(x, y, x_pe, y_pe, attn_mask=cross_mask), y + attn(y, x, y_pe, x_pe, attn_mask=cross_mask_T)  # cross attention
            elif name == 'self':
                dx, dy = x + attn(x, x, x_pe, x_pe), y + attn(y, y, y_pe, y_pe, attn_mask=self_mask)  # self attention, only apply self-mask on projection points
            x, y = x + ffn(dx), y + ffn(dy) 
        return x, y

class CoordAttention(nn.Module):
    def __init__(self, input_feat_dim:int, input_coord_dim:int, num_feat_heads=8, dim_head=64, dropout=0.):
        """Feature and Coordinate Attention

        Args:
            input_feat_dim (int): embedding dim of the feature part. teal_embed_dim = input_feat_dim + input_coord_dim
            input_coord_dim (int): embedding dim of the coordinate part, only used for v projection
            num_feat_heads (int, optional): number of feature heads. num_heads = num_feat_heads + 1. Defaults to 8.
            dim_head (int, optional): dim of each head. Defaults to 64.
            dropout (_type_, optional): dropout prob. Defaults to 0..
        """
        super().__init__()
        num_heads = num_feat_heads + 1
        inner_feat_dim = dim_head *  num_feat_heads
        input_v_dim = input_feat_dim + input_coord_dim
        inner_total_dim = dim_head * num_heads
        self.feat_heads = num_feat_heads
        self.total_heads = num_heads
        self.inner_feat_dim = inner_feat_dim
        self.inner_total_dim = inner_total_dim
        self.scale:float = dim_head ** -0.5
        self.coord_scale = nn.Parameter(torch.ones(1) * self.scale,requires_grad=True)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(input_feat_dim, inner_feat_dim, bias = False)  # hxD
        self.to_k = nn.Linear(input_feat_dim, inner_feat_dim, bias=False) # hxD
        self.to_v = nn.Linear(input_v_dim, inner_total_dim, bias=False) # (h+1)xD
        self.to_out = nn.Sequential(
            nn.Linear(inner_total_dim, inner_feat_dim, bias = False),  # (h+1)xD -> hxD
            nn.Dropout(dropout)
        )  # output as the input of the next attention block

    def forward(
        self,
        x:torch.Tensor, y:torch.Tensor,
        coord_x:torch.Tensor, coord_y:torch.Tensor,
        attn_mask:Optional[torch.Tensor]=None,
    ):
        # cancel the x normalization because x is from the normalized image feature patches and there is only one CoordAttention Block
        q:torch.Tensor = self.to_q(x)  # (B, N, D)
        k:torch.Tensor = self.to_k(y)  # (B, M, D)
        v:torch.Tensor = self.to_v(torch.cat([y, coord_y], dim=-1))  # (B, M, D+D_h)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.feat_heads), [q, k])  # (B, n, D) -> (B, n, h, d) -> (B, h, n, d)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.total_heads)
        feat_dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (B, h, N, D) * (B, h, D, M) -> (B, h, N, M)
        coord_dosts = torch.matmul(coord_x, coord_y.transpose(-1,-2)) * self.coord_scale # (B, N, M)
        dots = torch.cat([feat_dots, coord_dosts.unsqueeze(1)],dim=1) # (B, h+1, N, M)
        if exists(attn_mask):
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]
            elif attn_mask.ndim == 4:
                pass
            else:
                raise NotImplementedError("attn_mask.ndim must be 2,3,4, got {}".format(attn_mask.ndim))
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max) # assign value of mask to -inf (after softmax will be 0)

        attn = self.softmax(dots)  # (B, h+1, N, M)
        attn = self.dropout(attn)  

        out = torch.matmul(attn, v)  # (B, h+1, N, M), (B, h+1, M, D) -> (B, h, N, D)
        out = rearrange(out, 'b h n d -> b n (h d)')  # head merging
        return self.to_out(out)

    
class CoordAttentionV2(nn.Module):
    def __init__(self, input_feat_dim:int, input_coord_dim:int, num_feat_heads=8, dim_head=64, dropout=0.):
        """Feature and Coordinate Attention
        Compared with V1, it encodes both coordinate and feature into q & v

        Args:
            input_feat_dim (int): embedding dim of the feature part. teal_embed_dim = input_feat_dim + input_coord_dim
            input_coord_dim (int): embedding dim of the coordinate part, only used for v projection
            num_feat_heads (int, optional): number of feature heads. num_heads = num_feat_heads + 1. Defaults to 8.
            dim_head (int, optional): dim of each head. Defaults to 64.
            dropout (_type_, optional): dropout prob. Defaults to 0..
        """
        super().__init__()
        num_heads = num_feat_heads + 1
        inner_feat_dim = dim_head *  num_feat_heads
        input_dim = input_feat_dim + input_coord_dim
        inner_total_dim = dim_head * num_heads
        self.feat_heads = num_feat_heads
        self.total_heads = num_heads
        self.inner_feat_dim = inner_feat_dim
        self.inner_total_dim = inner_total_dim
        self.scale:float = dim_head ** -0.5
        self.coord_scale = nn.Parameter(torch.ones(1) * self.scale,requires_grad=True)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(input_dim, inner_feat_dim, bias=False)  # hxD
        self.to_k = nn.Linear(input_dim, inner_feat_dim, bias=False) # hxD
        self.to_v = nn.Linear(input_dim, inner_total_dim, bias=False) # (h+1)xD
        self.to_out = nn.Sequential(
            nn.Linear(inner_total_dim, inner_feat_dim, bias = False),  # (h+1)xD -> hxD
            nn.Dropout(dropout)
        )  # output as the input of the next attention block

    def forward(
        self,
        x:torch.Tensor, y:torch.Tensor,
        coord_x:torch.Tensor, coord_y:torch.Tensor,
        x_pos_embed:Optional[torch.Tensor]=None,
        y_pos_embed:Optional[torch.Tensor]=None,
        attn_mask:Optional[torch.Tensor]=None,
    ):
        # cancel the x normalization because x is from the normalized image feature patches and there is only one CoordAttention Block
        posed_x = torch.cat([x, coord_x], dim=-1)
        posed_y = torch.cat([y, coord_y], dim=-1)
        q:torch.Tensor = self.to_q(posed_x) # (B, N, D)
        k:torch.Tensor = self.to_k(posed_y) # (B, M, D)
        if x_pos_embed is not None:
            q = q + x_pos_embed
        if y_pos_embed is not None:
            k = k + y_pos_embed
        v:torch.Tensor = self.to_v(posed_y)  # (B, M, D+D_h)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.feat_heads), [q, k])  # (B, n, D) -> (B, n, h, d) -> (B, h, n, d)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.total_heads)
        feat_dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (B, h, N, D) * (B, h, D, M) -> (B, h, N, M)
        coord_dosts = torch.matmul(coord_x, coord_y.transpose(-1,-2)) * self.coord_scale # (B, N, M)
        dots = torch.cat([feat_dots, coord_dosts.unsqueeze(1)],dim=1) # (B, h+1, N, M)
        if exists(attn_mask):
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]
            elif attn_mask.ndim == 4:
                pass
            else:
                raise NotImplementedError("attn_mask.ndim must be 2,3,4, got {}".format(attn_mask.ndim))
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max) # assign value of mask to -inf (after softmax will be 0)

        attn = self.softmax(dots)  # (B, h+1, N, M)
        attn = self.dropout(attn)  

        out = torch.matmul(attn, v)  # (B, h+1, N, M), (B, h+1, M, D) -> (B, h, N, D)
        out = rearrange(out, 'b h n d -> b n (h d)')  # head merging
        return self.to_out(out)

__ATTENTION_TYPE__ = {
    "Attention": Attention,
    "AttentionRoPE": AttentionRoPE,
}

__ATTENTION__ = TypeVar("__ATTENTION__", Attention, AttentionRoPE)