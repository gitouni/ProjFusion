import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from .utils import grid_sample_wrapper, mesh_grid, k_nearest_neighbor, batch_indexing, softmax, timer
from .mlp import Conv1dNormRelu, Conv2dNormRelu
from .attention import coord_2d_mesh, RMSNorm, LayerNorm
from .embedding import HarmonicEmbedding
from typing import Literal, Tuple

class CLFM(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, fusion_fn:Literal['add','concat','gated','sk']='sk', norm=None, fusion_knn:int=1):
        super().__init__()

        self.interp = FusionAwareInterp(in_channels_3d, k=fusion_knn, norm=norm)
        self.mlps3d = Conv1dNormRelu(in_channels_2d, in_channels_2d, norm=norm)

        if fusion_fn == 'add':
            self.fuse2d = AddFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
            self.fuse3d = AddFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm)
        elif fusion_fn == 'concat':
            self.fuse2d = ConcatFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
            self.fuse3d = ConcatFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm)
        elif fusion_fn == 'gated':
            self.fuse2d = GatedFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
            self.fuse3d = GatedFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm)
        elif fusion_fn == 'sk':
            self.fuse2d = SKFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm, reduction=2)
            self.fuse3d = SKFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm, reduction=2)
        else:
            raise ValueError

    def forward(self, uv: torch.Tensor, feat_2d:torch.Tensor, feat_3d:torch.Tensor):
        feat_3d_interp = self.interp(uv, feat_2d.detach(), feat_3d.detach())
        out2d = self.fuse2d(feat_2d, feat_3d_interp)

        feat_2d_sampled = grid_sample_wrapper(feat_2d.detach(), uv)
        out3d = self.fuse3d(self.mlps3d(feat_2d_sampled.detach()), feat_3d)

        return out2d, out3d  # (B,C,H,W), (B,C,N)

class CLFM_2D(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, fusion_fn:Literal['add','concat','gated','sk']='sk', norm=None, fusion_knn=1):
        super().__init__()
        self.interp = FusionAwareInterp(in_channels_3d, k=fusion_knn, norm=norm)
        if fusion_fn == 'add':
            self.fuse2d = AddFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
        elif fusion_fn == 'concat':
            self.fuse2d = ConcatFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
        elif fusion_fn == 'gated':
            self.fuse2d = GatedFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
        elif fusion_fn == 'sk':
            self.fuse2d = SKFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm, reduction=2)
        else:
            raise ValueError

    def forward(self, uv:torch.Tensor, feat_2d:torch.Tensor, feat_3d:torch.Tensor):
        # feat_2d = feat_2d.float()
        # feat_3d = feat_3d.float()
        feat_3d_interp = self.interp(uv, feat_2d.detach(), feat_3d.detach())
        out2d = self.fuse2d(feat_2d, feat_3d_interp)
        return out2d  # (N,C,H,W)

class FusionAwareInterp(nn.Module):
    def __init__(self, n_channels_3d:int, k:int=1, norm=None):
        super().__init__()
        self.k = k
        self.out_conv = Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm)
        self.score_net = nn.Sequential(
            Conv2dNormRelu(3, k),  # [dx, dy, |dx, dy|_2]
            Conv2dNormRelu(k, n_channels_3d, act='sigmoid'),
        )

    def forward(self, uv:torch.Tensor, image_hw:Tuple[int,int], feat_3d:torch.Tensor):
        image_h, image_w = image_hw
        bs = uv.shape[0]
        n_channels_3d = feat_3d.shape[1]

        grid = mesh_grid(bs, image_h, image_w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        knn_indices = k_nearest_neighbor(uv, grid, self.k)  # [B, HW, k]

        knn_uv, knn_feat3d = torch.split(
            batch_indexing(
                torch.cat([uv, feat_3d], dim=1),
                knn_indices,
                layout='channel_first'
            ), [2, n_channels_3d], dim=1)

        knn_offset = knn_uv - grid[..., None]  # [B, 2, HW, k]
        knn_offset_norm = torch.linalg.norm(knn_offset, dim=1, keepdim=True)  # [B, 1, HW, k]

        score_input = torch.cat([knn_offset, knn_offset_norm], dim=1)  # [B, 3, HW, K]
        score = self.score_net(score_input)  # [B, n_channels_3d, HW, k]
        # score = softmax(score, dim=-1)  # [B, n_channels_3d, HW, k]

        final = score * knn_feat3d  # [B, n_channels_3d, HW, k]
        final = final.sum(dim=-1).reshape(bs, -1, image_h, image_w)  # [B, n_channels_3d, H, W]
        final = self.out_conv(final)

        return final


class FusionAwareGAT(nn.Module):
    def __init__(self, img_feat_dim: int, point_feat_dim: int, img_hw: Tuple[int, int], k: int, 
            n_harmonic_functions: int, append_input: bool, use_mask: bool, max_dist: float,
            num_heads = 6, dim_head = 64, dropout = 0.):
        super().__init__()
        img_coord_x, img_coord_y = coord_2d_mesh(*img_hw, normalize=True)  # (H, W), (H, W)
        self.img_hw = img_hw
        self.img_uv = torch.stack([img_coord_x.flatten(), img_coord_y.flatten()], dim=-1)  # (hw, 2)
        if n_harmonic_functions <= 0:
            self.pos_emb = nn.Identity()
            pos_out_dim = 2
        else:
            self.pos_emb = HarmonicEmbedding(n_harmonic_functions, append_input=append_input)
            pos_out_dim = self.pos_emb.get_output_dim(input_dims=2)
        self.use_mask = use_mask
        self.max_dist = max_dist
        self.k = k
        inner_dim = dim_head *  num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.norm = LayerNorm(img_feat_dim)

        self.q_norm = RMSNorm(num_heads, dim_head)
        self.k_norm = RMSNorm(num_heads, dim_head)

        self.dropout_p = dropout

        self.to_q = nn.Linear(img_feat_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(point_feat_dim + pos_out_dim, inner_dim * 2, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, inner_dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, feat_2d: torch.Tensor, feat_3d: torch.Tensor, proj_uv: torch.Tensor, output_type: Literal['2d', '1d']):
        """Graph Attention

        Args:
            feat_2d (torch.Tensor): [B, N, D]
            feat_3d (torch.Tensor): [B, N, D]
            proj_uv (torch.Tensor): [B, N, 2], normalzied by [h, w]
            output_type (Literal['2d','1d'])

        Returns:
            _type_: _description_
        """
        B = feat_2d.shape[0]
        HW = self.img_uv.shape[0]
        img_uv = self.img_uv.unsqueeze(0).expand(B, -1, -1).to(feat_2d)  # (HW, 2) -> (B, HW, 2), float
        knn_indices = k_nearest_neighbor(proj_uv, img_uv, self.k)  # [B, HW, k]
        knn_indices = knn_indices.flatten(start_dim=1).unsqueeze(-1).expand(-1, -1, 2)  # (B, HW, K) -> (B, HW*K) -> (B, HW*K, 2)
        knn_uv = torch.gather(img_uv, 1, knn_indices).view(B, HW, self.k, 2)  # (B, HW, 2) -> (B, HW*K, 2) -> (B, HW, K, 2)
        knn_feat3d = torch.gather(feat_3d, 1, knn_indices).view(B, HW, self.k, -1)  # (B, HW, D) -> (B, HW*K, D) -> (B, HW, K, D)
        knn_offset = knn_uv - img_uv.unsqueeze(-2)  # (B, HW, K, 2), (B, HW, 1, 2) -> (B, HW, K, 2)
        if self.use_mask:
            with torch.no_grad():
                knn_norm = torch.linalg.norm(knn_offset, dim=-1)  # (B, HW, K, 2) -> (B, HW, K)
                mask = knn_norm <= self.max_dist  # (B, HW, K)
                mask = mask.view(B*HW, 1, self.k)  # (B*HW, 1, K)
        else:
            mask = None
        knn_offset_emb = self.pos_emb(knn_offset)  # (B, HW, K, 2) -> (B, HW, K, DC)
        x = self.norm(feat_2d)  # (B, HW, D1) -> (B, HW, D1)
        q = self.to_q(x).flatten(end_dim=1).unsqueeze(-2)  # (B, HW, D1) -> (B, HW, D) -> (B*HW, 1, D)
        k, v = torch.chunk(self.to_kv(torch.cat([knn_feat3d, knn_offset_emb], dim=-1)).flatten(end_dim=1), 2, dim=-1)  # (B, HW, K, DC) -> (B*HW, K, 2D) -> (B*HW, K, D), (B*HW, K, D)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> h h n d', h = self.num_heads, n = self.dim_head), [q, k, v])  # (B, N, D) -> (B, H, N, d), D = H * d
        q = self.q_norm(q)  # (B*HW, H, 1, d)
        k = self.k_norm(k)  # (B*HW, H, K, d)
        out = F.scaled_dot_product_attention(q, k, v,
                attn_mask=mask, dropout_p=(self.dropout_p if self.training else 0), scale=1.0).squeeze(-2)  # (B*HW, H, 1, d) -> (B*HW, H, d)
        out = self.to_out(out.flatten(start_dim=-2))  # (B*HW, H, d) -> (B*HW, H*d) -> (B*HW, D)
        if output_type == '2d':
            return rearrange(out, '(b h w) d -> b d h w', h = self.img_hw[0], w = self.img_hw[1], b = B)
        elif output_type == '1d':
            return rearrange(out, '(b h w) d -> b (h w) d', h = self.img_hw[0], w = self.img_hw[1], b = B)
        else:
            raise NotImplementedError(f'Unrecognized output_type: {output_type}')

class FusionAwareInterpCVPR(nn.Module):
    def __init__(self, n_channels_2d, n_channels_3d, k=3, norm=None) -> None:
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(n_channels_3d + 3, n_channels_3d, norm=norm),
            Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm),
            Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm),
        )

    def forward(self, uv, feat_2d, feat_3d):
        bs, _, h, w = feat_2d.shape

        grid = mesh_grid(bs, h, w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        with torch.no_grad():
            nn_indices = k_nearest_neighbor(uv, grid, k=1)[..., 0]  # [B, HW]
            nn_feat2d = batch_indexing(grid_sample_wrapper(feat_2d, uv), nn_indices)  # [B, n_channels_2d, HW]
            nn_feat3d = batch_indexing(feat_3d, nn_indices)  # [B, n_channels_3d, HW]
            nn_offset = batch_indexing(uv, nn_indices) - grid  # [B, 2, HW]
            nn_corr = torch.mean(nn_feat2d * feat_2d.reshape(bs, -1, h * w), dim=1, keepdim=True)  # [B, 1, HW]

        feat = torch.cat([nn_offset, nn_corr, nn_feat3d], dim=1)  # [B, n_channels_3d + 3, HW]
        feat = feat.reshape([bs, -1, h, w])  # [B, n_channels_3d + 3, H, W]
        final = self.mlps(feat)  # [B, n_channels_3d, H, W]

        return final


class AddFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None):
        super().__init__()

        if feat_format == 'nchw':
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
        elif feat_format == 'ncm':
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
        else:
            raise ValueError

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feat_2d, feat_3d):
        return self.relu(self.align1(feat_2d) + self.align2(feat_3d))


class ConcatFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None):
        super().__init__()

        if feat_format == 'nchw':
            self.mlp = Conv2dNormRelu(in_channels_2d + in_channels_3d, out_channels, norm=norm)
        elif feat_format == 'ncm':
            self.mlp = Conv1dNormRelu(in_channels_2d + in_channels_3d, out_channels, norm=norm)
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        return self.mlp(torch.cat([feat_2d, feat_3d], dim=1))


class GatedFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None):
        super().__init__()

        if feat_format == 'nchw':
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv2dNormRelu(out_channels, 2, norm=None, act='sigmoid')
            self.mlp2 = Conv2dNormRelu(out_channels, 2, norm=None, act='sigmoid')
        elif feat_format == 'ncm':
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv1dNormRelu(out_channels, 2, norm=None, act='sigmoid')
            self.mlp2 = Conv1dNormRelu(out_channels, 2, norm=None, act='sigmoid')
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        feat_2d = self.align1(feat_2d)
        feat_3d = self.align2(feat_3d)
        weight = self.mlp1(feat_2d) + self.mlp2(feat_3d)  # [N, 2, H, W]
        weight = softmax(weight, dim=1)  # [N, 2, H, W]
        return feat_2d * weight[:, 0:1] + feat_3d * weight[:, 1:2]


class SKFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None, reduction=1):
        super().__init__()

        if feat_format == 'nchw':
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif feat_format == 'ncm':
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError

        self.fc_mid = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(out_channels // reduction, out_channels * 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_2d, feat_3d):
        bs = feat_2d.shape[0]

        feat_2d = self.align1(feat_2d)
        feat_3d = self.align2(feat_3d)

        weight = self.avg_pool(feat_2d + feat_3d).reshape([bs, -1])  # [bs, C]
        weight = self.fc_mid(weight)  # [bs, C / r]
        weight = self.fc_out(weight).reshape([bs, -1, 2])  # [bs, C, 2]
        weight = softmax(weight, dim=-1)
        w1, w2 = weight[..., 0], weight[..., 1]  # [bs, C]

        if len(feat_2d.shape) == 4:
            w1 = w1.reshape([bs, -1, 1, 1])
            w2 = w2.reshape([bs, -1, 1, 1])
        else:
            w1 = w1.reshape([bs, -1, 1])
            w2 = w2.reshape([bs, -1, 1])

        return feat_2d * w1 + feat_3d * w2
