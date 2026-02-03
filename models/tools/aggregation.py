import torch
import torch.nn as nn
# from einops import repeat
from .core import BasicBlock, SEBlock
from .attention import Transformer, SelfCrossTransformer
from typing import Literal, Tuple, List, Union, TypeVar, Callable


class MiniSEAggregation(nn.Module):
    def __init__(self, inplanes:int, planes:int=96, final_feat: Tuple[int,int]=(2,4), se_reduction_ratio: int = 4, activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        self.out_dim = planes * final_feat[0] * final_feat[1]
        self.head_conv = nn.Sequential(
            BasicBlock(inplanes, planes*4, activation_fn=activation_fn),
            BasicBlock(planes*4, planes, activation_fn=activation_fn),
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=final_feat),
            nn.Flatten(start_dim=-3),  # (B, C, H, W) -> (B, C*H*W)
            SEBlock(self.out_dim, se_reduction_ratio, activation_fn)
        )

    def forward(self,x:torch.Tensor):
        x = self.head_conv(x)  # (B, C, H, W)
        x = self.pooling(x)  # (B, C, h, w) -> (B, C*h*w)
        return x  # (B, D)

class ResAggregation(nn.Module):
    def __init__(self, inplanes:int, planes=96, final_feat=(2,4), activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
        super(ResAggregation, self).__init__()
        self.head_conv = nn.Sequential(
            BasicBlock(inplanes, planes*8, activation_fn=activation_fn),
            BasicBlock(planes*8, planes*4, activation_fn=activation_fn),
            BasicBlock(planes*4, planes*2, activation_fn=activation_fn),
            BasicBlock(planes*2, planes, activation_fn=activation_fn),
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=final_feat),
            nn.Flatten(start_dim=1)
        )
        self.out_dim = planes * final_feat[0] * final_feat[1]

    def forward(self,x:torch.Tensor):
        x = self.head_conv(x)
        x_flatten = self.pooling(x)
        return x_flatten

class MiniResAggregation(nn.Module):
    def __init__(self, inplanes:int, planes:int=96, final_feat:Tuple[int,int]=(2,4), out_dim:int=-1, activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
        super(MiniResAggregation, self).__init__()
        self.head_conv = nn.Sequential(
            BasicBlock(inplanes, planes*4, activation_fn=activation_fn),
            BasicBlock(planes*4, planes, activation_fn=activation_fn),
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=final_feat),
            nn.Flatten(start_dim=1)
        )
        if out_dim > 0:
            self.align = nn.Sequential(
                nn.Linear(planes * final_feat[0] * final_feat[1], out_dim),
                activation_fn()
            )
            self.out_dim = out_dim
        else:
            self.align = nn.Identity()
            self.out_dim = planes * final_feat[0] * final_feat[1]
        self.act = activation_fn()

    def forward(self,x:torch.Tensor):
        x = self.head_conv(x)  # (B, C, H, W)
        x = self.pooling(x)  # (B, C, h, w) -> (B, C*h*w)
        x = self.align(x)  # (B, C*h*w) -> (B, D)
        x = self.act(x)  # (B, D)
        return x  # (B, D)

class AttentionAggregation(nn.Module):
    def __init__(self, embed_dim: int,
            num_tokens: int,
            depth: int,
            n_head: int,
            dim_head: int,
            feedforward_ratio: float,
            dropout: float = 0.,
            activation_fn: Callable[..., nn.Module] = lambda: nn.ReLU(inplace=True),
            ffn_layer: Literal['swiglu','mlp']='mlp'):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty(1, num_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, 0, 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        feedforward_dim = int(feedforward_ratio * embed_dim)
        self.transformer = Transformer(embed_dim, depth, n_head, dim_head, feedforward_dim, dropout, activation_fn, ffn_layer)
        self.mlp_head = nn.Linear(embed_dim, embed_dim)  # half for rotation, half for translation
        self.act = activation_fn()
        self.out_dim = embed_dim

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, D) -> (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1, D), (B, N, D) -> (B, N+1, D)
        x += self.pos_emb  # (B, N+1, D), (1, N+1, D) -> (B, N+1, D)
        x = self.transformer(x)  # (B, N+1, D)  -> (B, N+1, D)
        x = self.mlp_head(x[:, 0])  # (B, N+1, D) -> (B, D), only retrieve cls_token
        x = self.act(x)
        return x  # (B, D)

class CrossAttentionAggregation(nn.Module):
    def __init__(self, embed_dim:int,
                num_tokens:int,
                layer_names:List[str],
                n_head:int,
                dim_head:int,
                feedforward_ratio:float,
                dropout:float=0.,
                activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True),
                ffn_layer:Literal['swiglu','mlp']='mlp'):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens+1, embed_dim))  # +1 for cls_token
        self.cls_token_x = nn.Parameter(torch.randn(1, 1, embed_dim))
        feedforward_dim = int(feedforward_ratio * embed_dim)
        self.transformer = SelfCrossTransformer(layer_names, embed_dim, n_head, dim_head, feedforward_dim, dropout, activation_fn, ffn_layer)
        self.mlp_head = nn.Linear(embed_dim, embed_dim)  # half for rotation, half for translation
        self.act = activation_fn()
        self.out_dim = embed_dim

    def forward(self, x_y_mask:Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor], Union[None, torch.Tensor], Union[None, torch.Tensor], Union[None, torch.Tensor]]) -> torch.Tensor:
        assert isinstance(x_y_mask, (Tuple, List)), "xy must be 'tuple' or 'list', got {}".format(xy.__class__.__name__)
        x, y, x_pe, y_pe, cross_mask, self_mask = x_y_mask
        B = x.shape[0]
        cls_tokens_x = self.cls_token_x.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens_x, x), dim=1) + self.pos_embedding  # (B, N, d) -> (B, N+1, d)
        x, y = self.transformer(x, y, x_pe, y_pe, cross_mask, self_mask)  # (B, N+1, D)  -> (B, N+1, D)
        agg = self.mlp_head(x[:,0])  # (B, 2d) -> (B, 2d), only retrieve cls_token
        agg = self.act(agg)
        return agg # (B, D)

__AGGREGATION_DICT__ = {
    "MiniResAggregation": MiniResAggregation,
    "ResAggregation": ResAggregation,
    "AttentionAggregation": AttentionAggregation,
    "CrossAttentionAggregation": CrossAttentionAggregation,
    "MiniSEAggregation": MiniSEAggregation}

__AGGREGATION__ = TypeVar("__AGGREGATION__", MiniResAggregation, ResAggregation, AttentionAggregation, CrossAttentionAggregation, MiniSEAggregation)