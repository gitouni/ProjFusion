from typing import Literal, List, Dict, Tuple, Union, TypeVar, TypedDict, Callable, Any
from functools import partial
from abc import abstractmethod

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torchvision.models import (ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,
                ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights)
# from torch.nn import functional as F
# third-party libs
from .point_conv import PointConv
from .mlp import MLP1d
from .utils import project_pc2image, build_pc_pyramid_single, se3_transform
# from .utils import furthest_point_sampling, batch_indexing, knn_interpolation
from .csrc import correlation2d
from .clfm import FusionAwareInterp
# from ..Modules import resnet18 as custom_resnet
# from .convgru import BottleneckBlock, ResidualBlock, FeatureEncoder
from ..pointgpt import load_pointgpt
from .attention import ViTEncoder, normalize_grid, coord_2d_mesh, __ATTENTION__, __ATTENTION_TYPE__
from .embedding import HarmonicEmbedding
# from .attention import coord_2d_mesh, normalize_grid
from ..util.constant import BatchedCameraInfoDict, PredictMode
# def grid_sample(img: torch.Tensor, absolute_grid: torch.Tensor, mode: str = "bilinear", align_corners: Optional[bool] = None):
#     """Same as torch's grid_sample, with absolute pixel coordinates instead of normalized coordinates."""
#     h, w = img.shape[-2:]

#     xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
#     xgrid = 2 * xgrid / (w - 1) - 1
#     # Adding condition if h > 1 to enable this function be reused in raft-stereo
#     if h > 1:
#         ygrid = 2 * ygrid / (h - 1) - 1
#     normalized_grid = torch.cat([xgrid, ygrid], dim=-1)

#     return F.grid_sample(img, normalized_grid, mode=mode, align_corners=align_corners)

class AttentionArgv(TypedDict):
    input_dim: int
    height: int
    width: int
    base_freq: float
    heads: int
    dim_head: int
    dropout: float


class FrozenModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.module.eval()  # 强制 eval
        for p in self.module.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # 忽略外部传入的 mode，始终 eval
        super().train(False)
        self.module.eval()
        return self

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)

    def extraction(self, *args, **kwargs):
        if hasattr(self.module, "extraction"):
            with torch.no_grad():
                return self.module.extraction(*args, **kwargs)
        raise AttributeError("Underlying module has no 'extraction' method")

    def __getattr__(self, name: str):
        # 关键：这些名字交给父类处理，避免递归
        if name in {"module", "_wrap_methods"}:
            return super().__getattr__(name)
        # 同样关键：取 module 也走父类拿到真实属性，避免再次触发 __getattr__
        mod = super().__getattr__("module")
        try:
            return getattr(mod, name)  # 透传给底层
        except AttributeError:
            # 底层没有，再让父类继续常规查找（可能在本 wrapper 自己的属性/子模块里）
            return super().__getattr__(name)


def get_activation_func(activation:Literal['leakyrelu','relu','elu','gelu','silu'], inplace:bool) -> Callable[..., nn.Module]:
    if activation == 'leakyrelu':
        activation_fn = lambda: nn.LeakyReLU(0.1, inplace=inplace)
    elif activation == 'relu':
        activation_fn = lambda: nn.ReLU(inplace=inplace)
    elif activation == 'elu':
        activation_fn = lambda: nn.ELU(inplace=inplace)
    elif activation == 'gelu':
        activation_fn = lambda: nn.GELU()
    elif activation == 'silu':
        activation_fn = lambda: nn.SiLU(inplace=inplace)
    return activation_fn

class StateCache(TypedDict):
    feat_2d: torch.Tensor
    feat_3d: torch.Tensor
    xyz: torch.Tensor

class DepthImgGenerator:
    def __init__(self, pooling_size=1, max_depth=50.0):
        assert (pooling_size-1) % 2 == 0, 'pooling size must be odd to keep image size constant'
        if pooling_size == 1:
            self.pooling = lambda x:x
        else:
            self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size,stride=1,padding=(pooling_size-1)//2)
        self.max_depth = max_depth
        # InTran (3,4) or (4,4)

    @torch.no_grad()
    def project(self, pcd: torch.Tensor, camera_info: BatchedCameraInfoDict, return_uv: bool = False, margin_ratio: float = 0.0) ->\
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """transform point cloud to image

        Args:
            pcd (torch.Tensor): (B, 3, N)
            camera_info (BatchedCameraInfoDict): project information
            margin_ratio (float): ratio of expansion on each side (e.g. 0.1 means 10% padding of width/height)

        Returns:
            torch.Tensor: depth image (B, 1, H_new, W_new) where H_new = H + 2*dH
        """
        B = pcd.shape[0]
        uv = project_pc2image(pcd, camera_info)  # (B, 2, N)
        proj_x = uv[:, 0, :].type(torch.long)
        proj_y = uv[:, 1, :].type(torch.long)
        H, W = camera_info['sensor_h'], camera_info['sensor_w']
        margin = margin_ratio / 2  # half on left/right/top/bottom/padding
        # 1. 计算两侧的扩展像素量
        dH, dW = int(margin * H), int(margin * W)
        
        # 2. 修改画布大小：需要容纳 [-dW, W+dW) 的范围，总宽度为 W + 2*dW
        #    Height 同理: H + 2*dH
        final_H = H + 2 * dH
        final_W = W + 2 * dW
        
        # 3. 筛选在扩展范围内点的 Mask
        rev = (
            (proj_x >= -dW) & \
            (proj_x < (W + dW)) & \
            (proj_y >= -dH) & \
            (proj_y < (H + dH)) & \
            (pcd[:, 2, :] > 0)
        ).type(torch.bool)  # [B,N]

        batch_depth_img = torch.zeros(B, final_H, final_W, dtype=torch.float32).to(pcd.device)  # [B, H+2dH, W+2dW]

        for bi in range(B):
            rev_i = rev[bi, :]  # (N,)
            if not rev_i.any():
                continue
            
            # 4. 坐标偏移：将 [-dW, -dH] 平移到 [0, 0]
            proj_xrev = proj_x[bi, rev_i] + dW
            proj_yrev = proj_y[bi, rev_i] + dH
            
            # 这里的赋值操作现在是安全的
            batch_depth_img[bi, proj_yrev, proj_xrev] = pcd[bi, 2, rev_i] / self.max_depth # z

        if return_uv:
            return batch_depth_img.unsqueeze(1), uv
        return batch_depth_img.unsqueeze(1)   # (B, 1, H+2dH, W+2dW)
    
    @staticmethod
    @torch.no_grad()
    def binary_project(pcd: torch.Tensor, camera_info: Dict, margin_ratio: float = 0.0) -> torch.Tensor:
        """transform point cloud to image\n
        must be torch.no_grad() not torch.inference_mode(), otherwise the gradient backpropagation will fail
        Args:
            pcd (torch.Tensor): (B, 3, N)
            camera_info (Dict): project information
            margin_ratio (float): ratio of expansion on each side (e.g. 0.1 means 10% padding of width/height)

        Returns:
            torch.Tensor: binary mask image (B, 1, H_new, W_new) where H_new = H + 2*dH
        """
        B = pcd.shape[0]
        uv = project_pc2image(pcd, camera_info)
        proj_x = uv[:, 0, :].type(torch.long)
        proj_y = uv[:, 1, :].type(torch.long)
        H, W = camera_info['sensor_h'], camera_info['sensor_w']
        margin = margin_ratio / 2  # half on left/right/top/bottom/padding
        # 1. 计算两侧扩展像素
        dH, dW = int(margin * H), int(margin * W)
        
        # 2. 修正画布尺寸：需要容纳 [-dW, W+dW) 区间，总宽度为 W + 2*dW
        final_H = H + 2 * dH
        final_W = W + 2 * dW

        # 3. 掩码逻辑保持不变：筛选扩展范围内的点
        rev = (
            (proj_x >= -dW) & \
            (proj_x < (W + dW)) & \
            (proj_y >= -dH) & \
            (proj_y < (H + dH)) & \
            (pcd[:, 2, :] > 0)
        ).type(torch.bool)  # [B,N]
        
        # 4. 初始化修正后尺寸的画布
        batch_mask_img = torch.zeros(B, final_H, final_W, dtype=torch.float32).to(pcd.device)   # [B, H+2dH, W+2dW]
        
        for bi in range(B):
            rev_i = rev[bi, :]  # (N,)
            if not rev_i.any():
                continue
            
            # 5. 坐标偏移：[-dW, -dH] -> [0, 0]
            proj_xrev = proj_x[bi, rev_i] + dW
            proj_yrev = proj_y[bi, rev_i] + dH
            
            # 赋值 (现在索引安全了)
            batch_mask_img[bi * torch.ones_like(proj_xrev), proj_yrev, proj_xrev] = 1 
            
        return batch_mask_img.unsqueeze(1)   # (B, 1, H+2dH, W+2dW)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,
                 dilation=1, activation_fn: Callable = lambda: nn.ReLU(inplace=True), has_downsample:bool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, stride=stride,
                             padding=padding, dilation=dilation,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activate_func = activation_fn()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = nn.Identity()
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate_func(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.activate_func(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, input_dim, r=4, activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        self.weight_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // r),
            activation_fn(),
            nn.Linear(input_dim // r, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # x: (B, D)  
        w = self.weight_head(x)  # (B, D) -> (B, D // 4) -> (B, D), weight for each channel
        x = x * w  # (B, D) * (B, D) -> (B, D)
        return x

class SEConvBlock(nn.Module):
    def __init__(self, input_dim, r=4, activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [1,1]
        self.weight_head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // r, 1, 1, 0),
            activation_fn(),
            nn.Conv2d(input_dim // r, input_dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)  
        y = self.avg_pool(x)  # (B, C, H, W) -> (B, C, 1, 1)
        w = self.weight_head(y)  # (B, C, 1, 1) -> (B, C // 4, 1, 1) -> (B, C, 1, 1), weight for each channel
        x = x * w  # (B, C, H, W) * (B, C, 1, 1) -> (B, C, H, W)
        return x


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers:Literal[18, 34, 50, 101, 152], pretrained: bool, second_last: bool = True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: (resnet18, ResNet18_Weights.DEFAULT),
                   34: (resnet34, ResNet34_Weights.DEFAULT),
                   50: (resnet50, ResNet50_Weights.DEFAULT),
                   101: (resnet101, ResNet101_Weights.DEFAULT),
                   152: (resnet152, ResNet152_Weights.DEFAULT)}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        
        func, weights = resnets[num_layers]
        if not pretrained:
            weights = None
        self.encoder:ResNet = func(weights=weights)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        self.second_last = second_last
        if not second_last:
            self.downsample_ratio = 32
        else:
            self.downsample_ratio = 16

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        if not self.second_last:
            x = self.encoder.layer4(x)
        return x

    def get_output_dim(self) -> int:
        if not self.second_last:
            return self.num_ch_enc[-1]
        else:
            return self.num_ch_enc[-2]
    
    
    
class Encoder2D(nn.Module):
    def __init__(self, depth:Literal[18, 34, 50, 101, 152]=18, pretrained:bool=True):
        super().__init__()
        self.resnet = ResnetEncoder(depth, pretrained)
        self.out_chans = [64] + [self.resnet.encoder.base_width * 2 ** i for i in range(4)]
    
    def forward(self, x) -> List[torch.Tensor]:
        xs:List = self.resnet(x)  # List of features
        xs.pop(1)  # xs[1] obtained from xs[0] by max pooling
        return xs  # List of [B, C, H, W]

class Encoder3D(nn.Module):
    def __init__(self, n_channels:List[int], pcd_pyramid:List[int], embed_norm:bool=False, norm=None, k=16):
        super().__init__()
        assert len(pcd_pyramid)  == len(n_channels), "length of n_channels ({}) != length of pcd_pyramid ({})".format(len(n_channels), len(pcd_pyramid))
        self.pyramid_func = partial(build_pc_pyramid_single, n_samples_list=pcd_pyramid)
        in_chan = 4 if embed_norm else 3
        self.embed_norm = embed_norm
        self.level0_mlp = MLP1d(in_chan, [n_channels[0], n_channels[0]])
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.n_chans = n_channels
        for i in range(len(n_channels) - 1):
            self.mlps.append(MLP1d(n_channels[i], [n_channels[i], n_channels[i + 1]]))
            self.convs.append(PointConv(n_channels[i + 1], n_channels[i + 1], norm=norm, k=k))

    def forward(self, pcd:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """pcd hierchical encoding

        Args:
            pcd (torch.Tensor): B, 3, N

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: feats, xyzs
        """
        xyzs, _ = self.pyramid_func(pcd)
        inputs = xyzs[1]  # [bs, 3, n_points]
        if self.embed_norm:
            norm = torch.linalg.norm(inputs, dim=1, keepdim=True)
            inputs = torch.cat([inputs, norm], dim=1)
        feats = [self.level0_mlp(inputs)]

        for i in range(1,len(xyzs) - 1):
            feat = self.mlps[i-1](feats[-1])
            feat = self.convs[i-1](xyzs[i], feat, xyzs[i + 1])
            feats.append(feat)
        return feats, xyzs  

class CorrelationNet(nn.Module):
    def __init__(self, corr_dist:int, planes:int, activation:str, inplace:bool):
        super().__init__()
        activation_fn = get_activation_func(activation, inplace)
        corr_dim = (2 * corr_dist + 1) ** 2
        self.corr_block = partial(correlation2d, max_displacement=corr_dist, cpp_impl=True)
        self.corr_conv = BasicBlock(corr_dim, planes, activation_fn=activation_fn)
    def forward(self, img1:torch.Tensor, img2:torch.Tensor):
        corr = self.corr_block(img1, img2)  # (B, D, corr_dist, corr_dist)
        corr = self.corr_conv(corr)  # (B, C, )
        return corr

class MLPDualHead(nn.Module):
    def __init__(self, head_dims:List[int], sub_dims:List[int], activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        assert head_dims[-1] == sub_dims[0]
        head_mlps = []
        rot_mlps = []
        tsl_mlps = []
        for i in range(len(head_dims) -1):
            head_mlps.append(nn.Linear(head_dims[i], head_dims[i+1]))
            head_mlps.append(activation_fn())
        for i in range(len(sub_dims) - 2):
            rot_mlps.append(nn.Linear(sub_dims[i], sub_dims[i+1]))
            rot_mlps.append(activation_fn())
            tsl_mlps.append(nn.Linear(sub_dims[i], sub_dims[i+1]))
            tsl_mlps.append(activation_fn())
        rot_mlps.append(nn.Linear(sub_dims[-2], sub_dims[-1], bias=False))
        tsl_mlps.append(nn.Linear(sub_dims[-2], sub_dims[-1], bias=False))
        self.head_mlps = nn.Sequential(*head_mlps)
        self.rot_mlps = nn.Sequential(*rot_mlps)
        self.tsl_mlps = nn.Sequential(*tsl_mlps)

    def forward(self, x:torch.Tensor):
        x = self.head_mlps(x)
        rot_x = self.rot_mlps(x)
        tsl_x = self.tsl_mlps(x)
        return torch.cat([rot_x, tsl_x],dim=-1)
    
class MLPHead(nn.Module):
    def __init__(self, input_dim: int, head_dims:List[int], output_dim: int = 3,
            batchnorm: bool = False,
            activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        head_mlps = []
        head_dims = [input_dim] + head_dims + [output_dim]
        self.output_dim = output_dim
        for i in range(len(head_dims)-2):
            head_mlps.append(nn.Linear(head_dims[i], head_dims[i+1]))
            head_mlps.append(nn.BatchNorm1d(head_dims[i+1]) if batchnorm else nn.Identity())
            head_mlps.append(activation_fn())
        head_mlps.append(nn.Linear(head_dims[-2], head_dims[-1], bias=False))
        self.head_mlps = nn.Sequential(*head_mlps)

    def forward(self, x: torch.Tensor):
        x = self.head_mlps(x)  # 使得BatchNorm能真正其效果
        return x  # (*, self.output_dim)

class SEHead(nn.Module):
    def __init__(self, input_dim: int, se_reduction_ratio: int = 4, tanh: bool = False, activation_fn: Callable[..., nn.Module]=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        self.se_block = SEBlock(input_dim, se_reduction_ratio, activation_fn=activation_fn)
        if tanh:
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor):
        x = self.se_block(x)
        return self.act(torch.sum(x, dim=-1, keepdim=True))  # (*, D) -> (*, 1)
    

class Encoder(nn.Module):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.feat_buffer:StateCache = dict()

    @abstractmethod
    def store_buffer(self, image:torch.Tensor, pcd:torch.Tensor):
        pass

    @abstractmethod
    def store_buffer_direct(self, cache:StateCache):
        self.feat_buffer.update(cache)
    
    @abstractmethod
    def clear_buffer(self):
        self.feat_buffer.clear()
    
    @abstractmethod
    def get_buffers(self) -> StateCache:
        return self.feat_buffer

    
    @abstractmethod
    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:Dict[str, torch.Tensor]):
        return
    
    @abstractmethod
    def cache_forward(self, cache:StateCache, Tcl:torch.Tensor, camera_info:Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def encoder_cache(self, img:torch.Tensor, pcd:torch.Tensor) -> StateCache:
        pass

    @abstractmethod
    def kargv_for_aggregation(self) -> Dict:
        return



class AttenFusionNet(Encoder):
    def __init__(self,
            # 2D image Encoder
            image_hw: Tuple[int,int],
            img_encoder_type: Literal['vit','resnet'],
            img_encoder_args: Dict[str, Any],
            # 3D Encoder block
            pointgpt_config: str,
            pointgpt_checkpoint: str,
            # Harmonic Function
            use_coord: bool,
            use_harmonic: bool,
            harmonic_args: Dict,
            # boundary mask
            margin: float,
            use_mask: bool,
            # Cross Attention
            attention_type: str,
            attention_argv: Dict,
            # Global Config
            freeze_encoders: bool,
            # type:
            output_type: Literal['1d','2d']
            ):
        super().__init__()
        if img_encoder_type == 'vit':
            self.fnet_2d = ViTEncoder(**img_encoder_args, image_hw=image_hw, reshape=True)  # freeze parameters of fnet_2d
        elif img_encoder_type == 'resnet':
            self.fnet_2d = ResnetEncoder(**img_encoder_args)
        else:
            raise NotImplementedError("encoder type must be 'vit' or 'resnet'")
        self.fnet_3d, self.fnet_3d_max_depth = load_pointgpt(pointgpt_config, pointgpt_checkpoint)
        if freeze_encoders:
            self.fnet_2d = FrozenModule(self.fnet_2d)
            self.fnet_3d = FrozenModule(self.fnet_3d)
        embed_2d = self.fnet_2d.get_output_dim()
        embed_3d = self.fnet_3d.trans_dim
        num_heads = attention_argv.get('heads')
        head_dims = attention_argv.get('dim_head')
        self.embed_dim = num_heads * head_dims
        if img_encoder_type == 'vit':
            self.feat_w = image_hw[1] // self.fnet_2d.patch_size
            self.feat_h = image_hw[0] // self.fnet_2d.patch_size
        else:
            self.feat_w = image_hw[1] // self.fnet_2d.downsample_ratio
            self.feat_h = image_hw[0] // self.fnet_2d.downsample_ratio
        if output_type == '1d':  # attention aggregation
            self.aggregation_kargv = {"embed_dim": self.embed_dim, "num_tokens": self.feat_h * self.feat_w}
        elif output_type == '2d':
            self.aggregation_kargv = {"inplanes": self.embed_dim}
        else:
            raise NotImplementedError("output_type must be '1d' or '2d', got {}".format(output_type))
        if use_coord:
            if use_harmonic:
                self.harmonic_embedding = HarmonicEmbedding(**harmonic_args)
                q_input_dim = embed_2d + self.harmonic_embedding.get_output_dim(input_dims=2)  # projected points are 2D vectors
                kv_input_dim = embed_3d + self.harmonic_embedding.get_output_dim(input_dims=2)  # projected points are 2D vectors
            else:
                self.harmonic_embedding = nn.Identity()
                q_input_dim = embed_2d + 2
                kv_input_dim = embed_3d + 2
            img_coord_x, img_coord_y = coord_2d_mesh(self.feat_h, self.feat_w, normalize=True)  # (H*W, 2)
            img_coord_xy = torch.stack([img_coord_x.flatten(), img_coord_y.flatten()], dim=-1)  # (h*w, 2)
            self.img_uv_emb: torch.Tensor = self.harmonic_embedding(img_coord_xy)  # (H*W, D)
        else:
            self.harmonic_embedding = None
            q_input_dim = embed_2d
            kv_input_dim = embed_3d
        self.cross_attnetion_type = attention_type
        self.cross_attention_argv: AttentionArgv = dict(q_input_dim = q_input_dim,
                                                        kv_input_dim = kv_input_dim,
                                                        height=self.feat_h, width=self.feat_w, **attention_argv)  # height和width是为了兼容RoPE
        self.use_mask = use_mask
        # self.use_pos_embed = use_pos_embed
        self.margin = margin
        # if use_pos_embed:
        #     self.pos_embed = PositionEmbeddingCoordsSine2D(self.embed_dim, self.feat_h, self.feat_w, margin, pos_embed_temp)
        #     self.pos_embed_patches = nn.Parameter(self.pos_embed.get_patched_coordinate_embedding(flatten=True).unsqueeze(0), requires_grad=False)  # (hw, N) -> (1, hw, N)
        self.output_type = output_type
        self.feat_buffer = dict()

    def _lazy_init(self):
        self.cross_attention: __ATTENTION__ = __ATTENTION_TYPE__[self.cross_attnetion_type](**self.cross_attention_argv)  # additional head for coordinate attention (query: feat_2d, kv: feat_3d)
        self.out_dim = self.cross_attention.out_dim
        
    def kargv_for_aggregation(self) -> Dict:
        return self.aggregation_kargv
    
    def clear_buffer(self):
        self.feat_buffer.clear()

    def store_buffer(self, img: torch.Tensor, pcd: torch.Tensor):
        cache = self.encoder_cache(img, pcd)
        self.store_buffer_direct(cache)

    def store_buffer_direct(self, cache: StateCache):
        self.feat_buffer.update(cache)

    @torch.no_grad()
    def encoder_cache(self, img: torch.Tensor, pcd: torch.Tensor) -> StateCache:
        feat_2d = self.fnet_2d(img)  # (B, C, H, W)
        xyz, feat_3d = self.fnet_3d.extraction(pcd / self.fnet_3d_max_depth)  # (B, M, 3), (B, M, D)
        feat_2d = rearrange(feat_2d, 'b c n h -> b (n h) c')  # (B, H*W, C)
        return dict(feat_2d=feat_2d,  # (B, N, C)
                    feat_3d=feat_3d,  # (B, M, D)
                    xyz=xyz)  # (B, M, 3)
    
    def get_buffers(self) -> StateCache:
        return self.feat_buffer
    
    def forward(self, img: torch.Tensor, pcd: torch.Tensor, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        """forward through data

        Args:
            img (torch.Tensor): B, C, H, W
            pcd (torch.Tensor): B, M, 3
            Tcl (torch.Tensor): B, 4, 4
            camera_info (BatchedCameraInfoDict): intran information for projection
        Returns:
            torch.Tensor: crossed features of point cloud (B, M, D), M is the group number of pcd
        """
        if len(self.feat_buffer) == 0:
            cache = self.encoder_cache(img, pcd)
        else:
            cache = self.get_buffers()
        return self.cache_forward(cache, Tcl, camera_info)

    def cache_forward(self, cache: StateCache, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        """forward through cache/observation

        Args:
            cache (Dict[str, torch.Tensor]): cache output by the encoder
            Tcl (torch.Tensor): init_extran (B, 4, 4)
            camera_info (Dict[str, torch.Tensor]): intran information for projection

        Returns:
            torch.Tensor: crossed features of point cloud (B*G, M, D) or (B*G, D, h, w)
        """
        feat_2d, feat_3d, xyz = cache['feat_2d'], cache['feat_3d'], cache['xyz']  # (B, H*W, D), (B, M, D), (B, M, 3)
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        # Apply SE(3) transform for each group
        xyz_tf = se3_transform(Tcl, xyz.transpose(-1, -2))  # (B, 4, 4), (B, N, 3) -> (B, 3, N)

        # Prepare camera info for feature resolution
        feat_camera_info = camera_info.copy()
        feat_w, feat_h = self.feat_w, self.feat_h
        kx = feat_w / sensor_w
        ky = feat_h / sensor_h
        feat_camera_info.update({
            'sensor_w': feat_w,
            'sensor_h': feat_h,
            'fx': kx * camera_info['fx'],
            'fy': ky * camera_info['fy'],
            'cx': kx * camera_info['cx'],
            'cy': ky * camera_info['cy']
        })
        # Project point cloud to image plane
        
        proj_uv = project_pc2image(xyz_tf, feat_camera_info) # (B, 3, N) -> (B, 2, N)
        proj_uv.transpose_(-1, -2).contiguous()  # (B, 2, N) -> (B, N, 2)
        # Normalize and clamp projection coordinates
        margin_ratio = [-1 - self.margin, 1 + self.margin]
        proj_uv[..., 0] = normalize_grid(proj_uv[..., 0], feat_w)  # (B, N)  normalized to [-1, 1]
        proj_uv[..., 1] = normalize_grid(proj_uv[..., 1], feat_h)  # (B, N)
        # Build attention mask if needed
        if not self.use_mask:
            attn_mask = None
        else:
            valid_proj = (
                (proj_uv[..., 0] >= margin_ratio[0]) & (proj_uv[..., 0] <= margin_ratio[1]) &
                (proj_uv[..., 1] >= margin_ratio[0]) & (proj_uv[..., 1] <= margin_ratio[1]) & (xyz_tf[..., 2, :] > 0)  # only use depth > 0 points
            )  # (B, N)
            attn_mask = valid_proj.unsqueeze(-2).expand(-1, self.feat_h * self.feat_w, -1).detach()  # (B, H*W, N)
        proj_uv.clamp_(*margin_ratio)  # (B, N, 2)
        B = proj_uv.shape[0]
        if self.harmonic_embedding is not None:  # force 3d attention
            feat_3d_pos_emb = self.harmonic_embedding(proj_uv).detach()  # (B, N, D2)
            feat_3d = torch.cat([feat_3d, feat_3d_pos_emb], dim=-1)  # (B, N, D1+D2)
            feat_2d_pos_emb = self.img_uv_emb[None, ...].expand(B, -1, -1).to(feat_3d_pos_emb).detach()  # (B, H*W, D)
            feat_2d = torch.cat([feat_2d, feat_2d_pos_emb], dim=-1)  # (B, H*W, D1+D2)
        cross_feat_2d = self.cross_attention(
            feat_2d, feat_3d,
            k_coord_xy=proj_uv,
            attn_mask=attn_mask
        )  # (B, H*W, D)

        # Output based on type
        if self.output_type == '1d':
            return cross_feat_2d  # (B, M, D)
        elif self.output_type == '2d':
            return rearrange(cross_feat_2d, 'b (h w) d -> b d h w', h=feat_h, w=feat_w)  # (B, D, h, w)
        else:
            raise NotImplementedError('Unknown type: {}, must be 1d or 2d'.format(self.output_type))


class AttenDualFusionNet(AttenFusionNet):
    def __init__(self, **argv):
        super().__init__(**argv)

    def _lazy_init(self):
        self.rot_cross_attention: __ATTENTION__ = __ATTENTION_TYPE__[self.cross_attnetion_type](**self.cross_attention_argv)  # additional head for coordinate attention (query: feat_2d, kv: feat_3d)
        self.tsl_cross_attention: __ATTENTION__ = __ATTENTION_TYPE__[self.cross_attnetion_type](**self.cross_attention_argv)
        self.out_dim = self.rot_cross_attention.out_dim
        
    def kargv_for_aggregation(self) -> Dict:
        return self.aggregation_kargv
    
    def clear_buffer(self):
        self.feat_buffer.clear()

    def store_buffer(self, img: torch.Tensor, pcd: torch.Tensor):
        cache = self.encoder_cache(img, pcd)
        self.store_buffer_direct(cache)

    def store_buffer_direct(self, cache: StateCache):
        self.feat_buffer.update(cache)

    @torch.no_grad()
    def encoder_cache(self, img: torch.Tensor, pcd: torch.Tensor) -> StateCache:
        feat_2d = self.fnet_2d(img)  # (B, C, H, W)
        xyz, feat_3d = self.fnet_3d.extraction(pcd / self.fnet_3d_max_depth)  # (B, M, 3), (B, M, D)
        feat_2d = rearrange(feat_2d, 'b c n h -> b (n h) c')  # (B, H*W, C)
        return dict(feat_2d=feat_2d.detach(),  # (B, N, C)
                    feat_3d=feat_3d.detach(),  # (B, M, D)
                    xyz=xyz.detach())  # (B, M, 3)
    
    def get_buffers(self) -> StateCache:
        return self.feat_buffer
    
    def forward(self, img: torch.Tensor, pcd: torch.Tensor, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        """forward through data

        Args:
            img (torch.Tensor): B, C, H, W
            pcd (torch.Tensor): B, M, 3
            Tcl (torch.Tensor): B, 4, 4
            camera_info (BatchedCameraInfoDict): intran information for projection
        Returns:
            torch.Tensor: crossed features of point cloud (B, M, D), M is the group number of pcd
        """
        if len(self.feat_buffer) == 0:
            cache = self.encoder_cache(img, pcd)
        else:
            cache = self.get_buffers()
        return self.cache_forward(cache, Tcl, camera_info)

    def cache_forward(self, cache: StateCache, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        """forward through cache/observation

        Args:
            cache (Dict[str, torch.Tensor]): cache output by the encoder
            Tcl (torch.Tensor): init_extran (B, 4, 4)
            camera_info (Dict[str, torch.Tensor]): intran information for projection

        Returns:
            torch.Tensor: crossed features of point cloud (B*G, M, D) or (B*G, D, h, w)
        """
        feat_2d, feat_3d, xyz = cache['feat_2d'], cache['feat_3d'], cache['xyz']  # (B, H*W, D), (B, M, D), (B, M, 3)
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        # Apply SE(3) transform for each group
        xyz_tf = se3_transform(Tcl, xyz.transpose(-1, -2))  # (B, 4, 4), (B, N, 3) -> (B, 3, N)

        # Prepare camera info for feature resolution
        feat_camera_info = camera_info.copy()
        feat_w, feat_h = self.feat_w, self.feat_h
        kx = feat_w / sensor_w
        ky = feat_h / sensor_h
        feat_camera_info.update({
            'sensor_w': feat_w,
            'sensor_h': feat_h,
            'fx': kx * camera_info['fx'],
            'fy': ky * camera_info['fy'],
            'cx': kx * camera_info['cx'],
            'cy': ky * camera_info['cy']
        })
        # Project point cloud to image plane
        
        proj_uv = project_pc2image(xyz_tf, feat_camera_info) # (B, 3, N) -> (B, 2, N)
        proj_uv.transpose_(-1, -2).contiguous()  # (B, 2, N) -> (B, N, 2)
        # Normalize and clamp projection coordinates
        margin_ratio = [-1 - self.margin, 1 + self.margin]
        proj_uv[..., 0] = normalize_grid(proj_uv[..., 0], feat_w)  # (B, N)
        proj_uv[..., 1] = normalize_grid(proj_uv[..., 1], feat_h)  # (B, N)
        # Build attention mask if needed
        if not self.use_mask:
            attn_mask = None
        else:
            valid_proj = (
                (proj_uv[..., 0] >= margin_ratio[0]) & (proj_uv[..., 0] <= margin_ratio[1]) &
                (proj_uv[..., 1] >= margin_ratio[0]) & (proj_uv[..., 1] <= margin_ratio[1]) & (xyz_tf[..., 2, :] > 0)  # only use depth > 0 points
            )  # (B, N)
            attn_mask = valid_proj.unsqueeze(-2).expand(-1, self.feat_h * self.feat_w, -1).detach()  # (B, H*W, N)
        proj_uv.clamp_(*margin_ratio)  # (B, N, 2)
        B = proj_uv.shape[0]
        if self.harmonic_embedding is not None:  # force 3d attention
            feat_3d_pos_emb = self.harmonic_embedding(proj_uv).detach()  # (B, N, D2)
            feat_3d = torch.cat([feat_3d, feat_3d_pos_emb], dim=-1)  # (B, N, D1+D2)
            feat_2d_pos_emb = self.img_uv_emb[None, ...].expand(B, -1, -1).to(feat_3d_pos_emb).detach()  # (B, H*W, D)
            feat_2d = torch.cat([feat_2d, feat_2d_pos_emb], dim=-1)  # (B, H*W, D1+D2)
        rot_cross_feat_2d = self.rot_cross_attention(
            feat_2d, feat_3d,
            k_coord_xy=proj_uv,
            attn_mask=attn_mask
        )  # (B, H*W, D)
        tsl_cross_feat_2d = self.tsl_cross_attention(
            feat_2d, feat_3d,
            k_coord_xy=proj_uv,
            attn_mask=attn_mask
        )  # (B, H*W, D)
        # Output based on type
        if self.output_type == '1d':
            return rot_cross_feat_2d, tsl_cross_feat_2d  # (B, M, D)
        elif self.output_type == '2d':
            return map(lambda x: rearrange(x, 'b (h w) d -> b d h w', h=feat_h, w=feat_w), [rot_cross_feat_2d, tsl_cross_feat_2d])  # (B, D, h, w)
        else:
            raise NotImplementedError('Unknown type: {}, must be 1d or 2d'.format(self.output_type))

class DepthFusionNet(Encoder):
    def __init__(self,
            # 2D image Encoder
            image_hw: Tuple[int, int],
            img_encoder_type: Literal['vit', 'resnet'],
            img_encoder_args: Dict[str, Any],
            # 3D Depth Encoder block
            depth_encoder_type: Literal['resnet'],
            depth_encoder_args: Dict[str, Any],
            # Harmonic Function
            use_coord: bool,
            use_harmonic: bool,
            harmonic_args: Dict[str, Any],
            # Boundary expansion
            margin: float,
            # Cross Attention
            attention_type: str,
            attention_argv: Dict[str, Any],
            # Depth Generation Args
            pooling_size: int = 1,
            max_depth: float = 50.0,
            # Global Config
            freeze_encoders: bool = False,
            output_type: Literal['1d', '2d'] = '1d',
            use_mask: bool = False
            ):
        super().__init__()
        
        # --- 1. Initialize Encoders (Copied logic, no PointGPT) ---
        if img_encoder_type == 'vit':
            self.fnet_2d = ViTEncoder(**img_encoder_args, image_hw=image_hw, reshape=True)
        elif img_encoder_type == 'resnet':
            self.fnet_2d = ResnetEncoder(**img_encoder_args)
        else:
            raise NotImplementedError("img_encoder_type must be 'vit' or 'resnet'")

        if depth_encoder_type == 'vit':
            vit_image_hw = (int((1 + margin) * image_hw[0]), int((1 + margin) * image_hw[1]))
            self.fnet_3d = ViTEncoder(**depth_encoder_args, image_hw=vit_image_hw, reshape=True)
        elif depth_encoder_type == 'resnet':
            self.fnet_3d = ResnetEncoder(**depth_encoder_args)
        else:
            raise NotImplementedError("Currently only 'resnet' is supported for depth encoding")

        if freeze_encoders:
            self.fnet_2d = FrozenModule(self.fnet_2d)
            self.fnet_3d = FrozenModule(self.fnet_3d)

        self.depth_generator = DepthImgGenerator(pooling_size=pooling_size, max_depth=max_depth)
        
        # --- 2. Calculate Feature Dimensions ---
        embed_2d = self.fnet_2d.get_output_dim()
        embed_3d = self.fnet_3d.get_output_dim()

        # Calculate standard feature map size
        if img_encoder_type == 'vit':
            self.img_feat_w = image_hw[1] // self.fnet_2d.patch_size
            self.img_feat_h = image_hw[0] // self.fnet_2d.patch_size
        else:
            self.img_feat_w = image_hw[1] // self.fnet_2d.downsample_ratio
            self.img_feat_h = image_hw[0] // self.fnet_2d.downsample_ratio

        if depth_encoder_type == 'vit':
            self.depth_feat_w = ((1.0 + margin) * image_hw[1]) // self.fnet_3d.patch_size
            self.depth_feat_h = ((1.0 + margin) * image_hw[0]) // self.fnet_3d.patch_size
        else:
            self.depth_feat_w = ((1.0 + margin) * image_hw[1]) // self.fnet_3d.downsample_ratio
            self.depth_feat_h = ((1.0 + margin) * image_hw[0]) // self.fnet_3d.downsample_ratio
        
        # Calculate Expanded Feature Dimensions for Depth
        # margin implies expanding the original image by (1+margin)
        self.margin = margin
        
        # Determine output aggregation
        num_heads = attention_argv.get('heads')
        head_dims = attention_argv.get('dim_head')
        self.embed_dim = num_heads * head_dims
        
        if output_type == '1d':
            self.aggregation_kargv = {"embed_dim": self.embed_dim, "num_tokens": self.feat_h * self.feat_w}
        elif output_type == '2d':
            self.aggregation_kargv = {"inplanes": self.embed_dim}
        else:
            raise NotImplementedError("output_type must be '1d' or '2d'")
        self.output_type = output_type

        # --- 3. Coordinate Embeddings (Pre-calculated) ---
        if use_coord:
            if use_harmonic:
                self.harmonic_embedding = HarmonicEmbedding(**harmonic_args)
                coord_out_dim = self.harmonic_embedding.get_output_dim(input_dims=2)
            else:
                self.harmonic_embedding = nn.Identity()
                coord_out_dim = 2
            
            q_input_dim = embed_2d + coord_out_dim
            kv_input_dim = embed_3d + coord_out_dim

            # A. 2D Image Coordinates (Standard [-1, 1])
            img_coord_x, img_coord_y = coord_2d_mesh(self.img_feat_h, self.img_feat_w, normalize=True)
            img_coord_xy = torch.stack([img_coord_x.flatten(), img_coord_y.flatten()], dim=-1) # (H*W, 2)
            self.img_uv_emb: torch.Tensor = self.harmonic_embedding(img_coord_xy).detach() # (H*W, D)

            # B. 3D Depth Map Coordinates (Expanded range)
            # Normalize grid creates [-1, 1] for the expanded size. 
            # Multiply by (1+margin) to map it to the coordinate system of the original image.
            depth_coord_x, depth_coord_y = coord_2d_mesh(self.depth_feat_h, self.depth_feat_w, normalize=True)  # [-1, 1]
            depth_coord_xy = torch.stack([depth_coord_x.flatten(), depth_coord_y.flatten()], dim=-1) * (1 + margin) # (-1 - margin, 1 + margin)
            self.depth_uv_emb: torch.Tensor = self.harmonic_embedding(depth_coord_xy).detach() # (H_ex*W_ex, D)
        else:
            self.harmonic_embedding = None
            q_input_dim = embed_2d
            kv_input_dim = embed_3d

        # --- 4. Attention Module ---
        self.cross_attnetion_type = attention_type
        self.cross_attention_argv: Dict = dict(
            q_input_dim=q_input_dim,
            kv_input_dim=kv_input_dim,
            height=self.img_feat_h, 
            width=self.img_feat_w, 
            **attention_argv
        )
        self.use_mask = use_mask
        self.feat_buffer = dict()

    def _lazy_init(self):
        self.cross_attention: __ATTENTION__ = __ATTENTION_TYPE__[self.cross_attnetion_type](**self.cross_attention_argv)
        self.out_dim = self.cross_attention.out_dim

    def kargv_for_aggregation(self) -> Dict:
        return self.aggregation_kargv
    
    def clear_buffer(self):
        self.feat_buffer.clear()

    def store_buffer(self, img: torch.Tensor, pcd: torch.Tensor):
        cache = self.encoder_cache(img, pcd)
        self.store_buffer_direct(cache)

    def store_buffer_direct(self, cache: StateCache):
        self.feat_buffer.update(cache)

    def get_buffers(self) -> StateCache:
        return self.feat_buffer

    @torch.no_grad()
    def encoder_cache(self, img: torch.Tensor, pcd: torch.Tensor) -> StateCache:
        """
        Encode 2D image. Store raw PCD because 3D features are view-dependent (depend on Tcl).
        """
        feat_2d = self.fnet_2d(img)  # (B, C, H, W)
        feat_2d = rearrange(feat_2d, 'b c n h -> b (n h) c')  # (B, H*W, C)
        
        return dict(feat_2d=feat_2d,      # (B, N_img, C)
                    feat_3d=None,         # Placeholder
                    xyz=pcd)              # (B, M, 3) Raw points

    def forward(self, img: torch.Tensor, pcd: torch.Tensor, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        if len(self.feat_buffer) == 0:
            cache = self.encoder_cache(img, pcd)
        else:
            cache = self.get_buffers()
        return self.cache_forward(cache, Tcl, camera_info)

    def cache_forward(self, cache: StateCache, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        feat_2d, xyz = cache['feat_2d'], cache['xyz']
        B = feat_2d.shape[0]

        # --- 1. Point Cloud Transformation ---
        xyz_tf = se3_transform(Tcl, xyz.transpose(-1, -2))  # (B, 3, N)

        # --- 2. Generate and Encode Depth Map ---
        # (B, 1, H_ex, W_ex)
        depth_img = self.depth_generator.project(xyz_tf, camera_info, margin_ratio=self.margin)
        
        # ResNet expects 3 channels usually, replicate channel
        depth_img_3c = depth_img.repeat(1, 3, 1, 1)
        
        # Encode -> (B, D, h_ex, w_ex)
        feat_3d_map = self.fnet_3d(depth_img_3c)
        
        # Ensure dimensions match our pre-calculated logic
        # Ideally feat_3d_map.shape[-2:] == (self.expand_feat_h, self.expand_feat_w)
        # Flatten -> (B, N_ex, D)
        feat_3d = rearrange(feat_3d_map, 'b c h w -> b (h w) c')

        # --- 3. Add Pre-calculated Coordinate Embeddings ---
        if self.harmonic_embedding is not None:
            # Load pre-calculated embeddings (on correct device)
            # Use registered buffers to handle device placement automatically
            feat_3d_pos_emb = self.depth_uv_emb.unsqueeze(0).expand(B, -1, -1) # (B, N_ex, D_coord)
            feat_2d_pos_emb = self.img_uv_emb.unsqueeze(0).expand(B, -1, -1)   # (B, N_img, D_coord)
            
            feat_3d = torch.cat([feat_3d, feat_3d_pos_emb], dim=-1)
            feat_2d = torch.cat([feat_2d, feat_2d_pos_emb], dim=-1)

        # --- 5. Cross Attention ---
        cross_feat_2d = self.cross_attention(
            feat_2d, 
            feat_3d,
            attn_mask=None # Dense attention (image to depth map), usually no mask needed unless depth is sparse/invalid
        )

        # --- 6. Output ---
        if self.output_type == '1d':
            return cross_feat_2d
        elif self.output_type == '2d':
            return rearrange(cross_feat_2d, 'b (h w) d -> b d h w', h=self.img_feat_h, w=self.img_feat_w)
        
class DepthDualFusionNet(DepthFusionNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _lazy_init(self):
        # Initialize two separate attention heads
        self.rot_cross_attention: __ATTENTION__ = __ATTENTION_TYPE__[self.cross_attnetion_type](**self.cross_attention_argv)
        self.tsl_cross_attention: __ATTENTION__ = __ATTENTION_TYPE__[self.cross_attnetion_type](**self.cross_attention_argv)
        self.out_dim = self.rot_cross_attention.out_dim # Assuming both heads have same output dim

    def cache_forward(self, cache: StateCache, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        feat_2d, xyz = cache['feat_2d'], cache['xyz']
        B = feat_2d.shape[0]

        # --- 1. Point Cloud Transformation (Same as Base) ---
        xyz_tf = se3_transform(Tcl, xyz.transpose(-1, -2))  # (B, 3, N)

        # --- 2. Update Camera Info for Margin (Same as Base) ---
        # has been updated in the depth_generator

        # --- 3. Generate and Encode Depth Map (Same as Base) ---
        depth_img = self.depth_generator.project(xyz_tf, camera_info, margin_ratio=self.margin)
        depth_img_3c = depth_img.repeat(1, 3, 1, 1)
        feat_3d_map = self.fnet_3d(depth_img_3c)
        feat_3d = rearrange(feat_3d_map, 'b c h w -> b (h w) c')

        # --- 4. Add Coordinate Embeddings (Same as Base) ---
        if self.harmonic_embedding is not None:
            # Note: Ensure device placement matches
            feat_3d_pos_emb = self.depth_uv_emb.unsqueeze(0).expand(B, -1, -1).to(feat_3d.device)
            feat_2d_pos_emb = self.img_uv_emb.unsqueeze(0).expand(B, -1, -1).to(feat_2d.device)
            
            feat_3d = torch.cat([feat_3d, feat_3d_pos_emb], dim=-1)
            feat_2d = torch.cat([feat_2d, feat_2d_pos_emb], dim=-1)

        # --- 5. Dual Cross Attention (Split Paths) ---
        rot_cross_feat_2d = self.rot_cross_attention(
            feat_2d, 
            feat_3d,
            attn_mask=None
        )
        
        tsl_cross_feat_2d = self.tsl_cross_attention(
            feat_2d, 
            feat_3d,
            attn_mask=None
        )

        # --- 6. Output ---
        if self.output_type == '1d':
            return rot_cross_feat_2d, tsl_cross_feat_2d
        elif self.output_type == '2d':
            # Map rearrange over both outputs
            return map(lambda x: rearrange(x, 'b (h w) d -> b d h w', h=self.img_feat_h, w=self.img_feat_w), 
                       [rot_cross_feat_2d, tsl_cross_feat_2d])
        else:
            raise NotImplementedError

class ConcatFusionNet(Encoder):
    def __init__(self,
            # 2D image Encoder
            img_encoder_type: Literal['vit', 'resnet'],
            img_encoder_args: Dict[str, str],
            image_hw: Tuple[int, int],
            # 3D Encoder block
            pointgpt_config: str,
            pointgpt_checkpoint: str,
            # Global Config
            freeze_encoders: bool,
            output_type: Literal['1d', '2d']
            ):
        super().__init__()
        self.output_type = output_type
        
        # 1. Image Encoder
        if img_encoder_type == 'vit':
            self.fnet_2d = ViTEncoder(**img_encoder_args, image_hw=image_hw, reshape=True)
        elif img_encoder_type == 'resnet':
            self.fnet_2d = ResnetEncoder(**img_encoder_args)
        else:
            raise NotImplementedError("encoder type must be 'vit' or 'resnet'")
        
        # 2. Point Cloud Encoder (PointGPT)
        self.fnet_3d, self.fnet_3d_max_depth = load_pointgpt(pointgpt_config, pointgpt_checkpoint)
        
        if freeze_encoders:
            self.fnet_2d = FrozenModule(self.fnet_2d)
            self.fnet_3d = FrozenModule(self.fnet_3d)

        embed_2d = self.fnet_2d.get_output_dim()
        embed_3d = self.fnet_3d.trans_dim
        
        # 3. Dimensions
        self.embed_2d = embed_2d
        self.embed_3d = embed_3d
        self.out_dim = embed_2d + embed_3d  # Concatenation
        
        # Feature map size
        if img_encoder_type == 'vit':
            self.feat_w = image_hw[1] // self.fnet_2d.patch_size
            self.feat_h = image_hw[0] // self.fnet_2d.patch_size
        else:
            self.feat_w = image_hw[1] // self.fnet_2d.downsample_ratio
            self.feat_h = image_hw[0] // self.fnet_2d.downsample_ratio
            
        # 4. Aggregation Args
        if output_type == '1d':
            self.aggregation_kargv = {"embed_dim": self.out_dim, "num_tokens": self.feat_h * self.feat_w}
        elif output_type == '2d':
            self.aggregation_kargv = {"inplanes": self.out_dim}
        else:
            raise NotImplementedError("output_type must be '1d' or '2d'")
            
        self.feat_buffer = dict()

    def _lazy_init(self):
        self.head = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True))

    def kargv_for_aggregation(self) -> Dict:
        return self.aggregation_kargv
    
    def clear_buffer(self):
        self.feat_buffer.clear()

    def store_buffer(self, img: torch.Tensor, pcd: torch.Tensor):
        cache = self.encoder_cache(img, pcd)
        self.store_buffer_direct(cache)

    def store_buffer_direct(self, cache: StateCache):
        self.feat_buffer.update(cache)

    @torch.no_grad()
    def encoder_cache(self, img: torch.Tensor, pcd: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_2d = self.fnet_2d(img)  # (B, D, H, W)
        xyz, feat_3d = self.fnet_3d.extraction(pcd / self.fnet_3d_max_depth)  # (B, N, 3), (B, N, D)
        return dict(feat_2d=feat_2d,
                    feat_3d=feat_3d,
                    xyz=xyz)
    
    def get_buffers(self) -> Dict[str, torch.Tensor]:
        return self.feat_buffer

    def project_features_by_z_buffer(self, 
            feat_3d: torch.Tensor, 
            xyz_camera: torch.Tensor, 
            camera_info: Dict) -> torch.Tensor:
        """
        Project 3D features to 2D grid using Z-buffer with Clamping.
        Instead of discarding points outside FOV, coordinates are clamped to [0, W-1] and [0, H-1].
        
        Args:
            feat_3d: (B, N, C)
            xyz_camera: (B, 3, N) - Points in camera frame
            camera_info: dict with fx, fy, cx, cy, sensor_h, sensor_w
            
        Returns:
            grid: (B, C, H, W)
        """
        B, N, C = feat_3d.shape
        H, W = camera_info['sensor_h'], camera_info['sensor_w']
        
        # 1. Project to UV
        proj_uv = project_pc2image(xyz_camera, camera_info) # (B, 2, N)
        depth = xyz_camera[:, 2, :] # (B, N)
        
        # 2. Discretize and Clamp Coordinates
        # Clamp to valid image indices [0, W-1] and [0, H-1]
        # This forces outliers to the edges
        u = torch.round(proj_uv[:, 0, :]).long().clamp(0, W - 1)
        v = torch.round(proj_uv[:, 1, :]).long().clamp(0, H - 1)
        
        # 3. Create Flattened Indices
        batch_idx = torch.arange(B, device=feat_3d.device).unsqueeze(1).expand(B, N)
        
        # 4. Filter: Only discard points behind the camera (depth <= 0)
        # We no longer filter by u/v range since we clamped them.
        valid_mask = depth > 0 
        
        # Optimization: Early return if empty
        if not valid_mask.any():
            return torch.zeros((B, C, H, W), device=feat_3d.device, dtype=feat_3d.dtype)
            
        b_valid = batch_idx[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        d_valid = depth[valid_mask]
        f_valid = feat_3d[valid_mask] # (M, C)
        
        # 5. Sort and Unique to implement Z-Buffer
        
        # Construct flat spatial index: b * H * W + v * W + u
        flat_idx = b_valid * (H * W) + v_valid * W + u_valid
        
        # Step A: Sort by Depth Ascending
        # Puts closer points first.
        perm_depth = torch.argsort(d_valid)
        flat_idx_sorted = flat_idx[perm_depth]
        f_valid_sorted = f_valid[perm_depth]
        
        # Step B: Stable Sort by Spatial Index
        # Groups identical pixels together (preserving depth order within group).
        # Note: Points clamped to the same edge pixel will now be grouped here.
        perm_spatial = torch.argsort(flat_idx_sorted, stable=True)
        flat_idx_final = flat_idx_sorted[perm_spatial]
        f_valid_final = f_valid_sorted[perm_spatial]
        
        # Step C: Unique (Keep First)
        # For each pixel (including edge pixels), keep the first point (which is the closest one).
        mask_unique = torch.ones_like(flat_idx_final, dtype=torch.bool)
        mask_unique[1:] = flat_idx_final[1:] != flat_idx_final[:-1]
        
        unique_indices = flat_idx_final[mask_unique]
        unique_feats = f_valid_final[mask_unique]
        
        # 6. Scatter to Grid
        grid_flat = torch.zeros((B * H * W, C), device=feat_3d.device, dtype=feat_3d.dtype)
        grid_flat[unique_indices] = unique_feats
        
        # Reshape to (B, C, H, W)
        grid = grid_flat.view(B, H, W, C).permute(0, 3, 1, 2)
        return grid

    def forward(self, img: torch.Tensor, pcd: torch.Tensor, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        if len(self.feat_buffer) == 0:
            cache = self.encoder_cache(img, pcd)
        else:
            cache = self.get_buffers()
        return self.cache_forward(cache, Tcl, camera_info)

    def cache_forward(self, cache: StateCache, Tcl: torch.Tensor, 
            camera_info: BatchedCameraInfoDict) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # 1. Unpack Cache
        feat_2d = cache['feat_2d']  # (B, D_2d, H, W)
        feat_3d = cache['feat_3d']  # (B, N, D_3d)
        xyz = cache['xyz']          # (B, N, 3)
        
        # 2. Transform Point Cloud
        xyz_tf = se3_transform(Tcl, xyz.transpose(-1, -2))  # (B, 3, N)
        
        # 3. Prepare Feature Projection Camera Info
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        feat_camera_info = camera_info.copy()
        feat_h, feat_w = self.feat_h, self.feat_w
        
        # Verify feature map size matches
        assert feat_2d.shape[-2] == feat_h and feat_2d.shape[-1] == feat_w
        
        kx = feat_w / sensor_w
        ky = feat_h / sensor_h
        feat_camera_info.update({
            'sensor_w': feat_w,
            'sensor_h': feat_h,
            'fx': kx * camera_info['fx'],
            'fy': ky * camera_info['fy'],
            'cx': kx * camera_info['cx'],
            'cy': ky * camera_info['cy']
        })
        
        # 4. Project 3D Features to 2D Grid (Z-Buffer)
        # Returns (B, D_3d, H, W)
        feat_3d_proj = self.project_features_by_z_buffer(feat_3d, xyz_tf, feat_camera_info)
        
        # 5. Concatenate
        # feat_2d: (B, D2, H, W), feat_3d_proj: (B, D3, H, W)
        fusion_map = torch.cat([feat_2d, feat_3d_proj], dim=1)  # (B, D2+D3, H, W)
        fusion_map = self.head(fusion_map)  # (B, D2+D3, H, W)
        # 6. Format Output
        if self.output_type == '1d':
            # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
            return rearrange(fusion_map, 'b c h w -> b (h w) c')
        elif self.output_type == '2d':
            return fusion_map
        else:
            raise NotImplementedError


class ConcatDualFusionNet(ConcatFusionNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Since Concatenation is deterministic, we add lightweight heads 
        # to separate the features for Rotation and Translation tasks.
        # This ensures the network has the capacity to "extract separate features"
        # as requested.
    
    def _lazy_init(self):
        self.rot_head = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True)
        )
        
        self.tsl_head = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True)
        )

    def cache_forward(self, cache: StateCache, Tcl: torch.Tensor, 
            camera_info: BatchedCameraInfoDict) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # 1. Perform Standard Fusion (Get concatenated map)
        # We call the logic from parent, but since parent returns formatted output,
        # we might want the raw 2D map. Let's duplicate logic slightly or temporarily set output_type.
        
        # Reuse logic from parent to get fusion_map (B, C, H, W)
        # Note: We can't simply call super().cache_forward because it reshapes based on output_type.
        # So we replicate the core logic:
        
        feat_2d = cache['feat_2d']
        feat_3d = cache['feat_3d']
        xyz = cache['xyz']
        
        xyz_tf = se3_transform(Tcl, xyz.transpose(-1, -2))
        
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        feat_camera_info = camera_info.copy()
        feat_h, feat_w = self.feat_h, self.feat_w
        
        kx = feat_w / sensor_w
        ky = feat_h / sensor_h
        feat_camera_info.update({
            'sensor_w': feat_w, 'sensor_h': feat_h,
            'fx': kx * camera_info['fx'], 'fy': ky * camera_info['fy'],
            'cx': kx * camera_info['cx'], 'cy': ky * camera_info['cy']
        })
        
        feat_3d_proj = self.project_features_by_z_buffer(feat_3d, xyz_tf, feat_camera_info)
        fusion_map = torch.cat([feat_2d, feat_3d_proj], dim=1) # (B, C, H, W)
        
        # 2. Split Heads
        rot_feat = self.rot_head(fusion_map)
        tsl_feat = self.tsl_head(fusion_map)
        
        # 3. Format Output
        if self.output_type == '1d':
            return (rearrange(rot_feat, 'b c h w -> b (h w) c'), 
                    rearrange(tsl_feat, 'b c h w -> b (h w) c'))
        elif self.output_type == '2d':
            return rot_feat, tsl_feat
        else:
            raise NotImplementedError
        

class PoolFusionNet(Encoder):
    def __init__(self,
            # 2D image Encoder
            img_encoder_type: Literal['vit','resnet'],
            img_encoder_args: Dict[str, Any],
            image_hw: Tuple[int,int],
            # 3D Encoder block
            pointgpt_config:str,
            pointgpt_checkpoint:str,
            # proj args:
            margin: float,
            # Interp Function
            fusion_args: Dict[str, Any],
            # Global Config
            freeze_encoders: bool,
            output_type: Literal['1d', '2d']
            ):
        super().__init__()
        self.output_type = output_type
        self.margin = margin
        if img_encoder_type == 'vit':
            self.fnet_2d = ViTEncoder(**img_encoder_args, image_hw=image_hw, freeze=freeze_encoders, reshape=True)  # freeze parameters of fnet_2d
        elif img_encoder_type == 'resnet':
            self.fnet_2d = ResnetEncoder(**img_encoder_args, freeze=freeze_encoders)
        else:
            raise NotImplementedError("encoder type must be 'vit' or 'resnet'")
        self.fnet_3d, self.fnet_3d_max_depth = load_pointgpt(pointgpt_config, pointgpt_checkpoint)
        if freeze_encoders:
            self.fnet_2d = FrozenModule(self.fnet_2d)
            self.fnet_3d = FrozenModule(self.fnet_3d)
        # if freeze_encoders:
        #     self.fnet_3d.freeze()  # freeze parameters of fnet_3d
        embed_2d = self.fnet_2d.get_output_dim()
        embed_3d = self.fnet_3d.trans_dim
        # assert embed_2d == embed_3d, 'fnet2d output_dim ({}) != fnet3d output_dim ({})'.format(embed_2d, embed_3d)
        self.fusion = FusionAwareInterp(embed_3d, **fusion_args)
        planes = embed_2d + embed_3d
        self.out_dim = planes
        if img_encoder_type == 'vit':
            self.feat_w = image_hw[1] // self.fnet_2d.patch_size
            self.feat_h = image_hw[0] // self.fnet_2d.patch_size
        else:
            self.feat_w = image_hw[1] // self.fnet_2d.downsample_ratio
            self.feat_h = image_hw[0] // self.fnet_2d.downsample_ratio
        if output_type == '1d':  # attention aggregation
            self.aggregation_kargv = {"embed_dim": embed_2d, "num_tokens": self.feat_h * self.feat_w}
        elif output_type == '2d':
            self.aggregation_kargv = {"inplanes": planes}
        else:
            raise NotImplementedError("output_type must be '1d' or '2d', got {}".format(output_type))
        self.feat_buffer = dict()

    def _lazy_init(self):
        self.head = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True))

    def kargv_for_aggregation(self) -> Dict:
        return self.aggregation_kargv
    
    def clear_buffer(self):
        self.feat_buffer.clear()

    def store_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        cache = self.encoder_cache(img, pcd)
        self.store_buffer_direct(cache)

    def store_buffer_direct(self, cache: StateCache):
        self.feat_buffer.update(cache)

    @torch.no_grad()
    def encoder_cache(self, img: torch.Tensor, pcd: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode 2d and 3d features

        Args:
            img (torch.Tensor): [B, 3, H, W]
            pcd (torch.Tensor): [B, N, 3]

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        feat_2d = self.fnet_2d(img)  # (B, 3, H, W) -> (B, D, h, w)
        xyz, feat_3d = self.fnet_3d.extraction(pcd / self.fnet_3d_max_depth)  # (B, N1, 3), (B, N1, D)
        return dict(feat_2d=feat_2d,  # (B, D, H, W)
                    feat_3d=feat_3d,  # (B, N1, 3)
                    xyz=xyz)  # (B, N1, D)
    
    def get_buffers(self) -> Dict[str, torch.Tensor]:
        return dict(feat_2d=self.feat_buffer['feat_2d'],  # (B, D, H, W)
                    feat_3d=self.feat_buffer['feat_3d'],  # (B, N1, 3)
                    xyz=self.feat_buffer['xyz'])  # (B, N1, D)
    
    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:BatchedCameraInfoDict):
        """forward through data

        Args:
            img (torch.Tensor): [B, 3, H, W]
            pcd (torch.Tensor): [B, N, 3]
            Tcl (torch.Tensor): [B, G, 4, 4], [:,0,:,:] is the GT SE(3)
            camera_info (BatchedCameraInfoDict): parameters of the camera intrinsic matrix
        Returns:
            torch.Tensor: crossed features of point cloud (B, M, D), M is the group number of pcd
        """
        if len(self.feat_buffer) == 0:
            cache = self.encoder_cache(img, pcd)
        else:
            cache = self.get_buffers()
        return self.cache_forward(cache, Tcl, camera_info)

    def cache_forward(self, cache: StateCache, Tcl: torch.Tensor, 
            camera_info: BatchedCameraInfoDict) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """forward through cache/observation

        Args:
            cache (StateCache): cache of env
            Tcl (torch.Tensor): groups of extrans [B, 4, 4]
            camera_info (BatchedCameraInfoDict): intran information for projection

        Returns:
            fusion_map: (B, C, h, w) or (B, h*w, C)
        """
        feat_2d, feat_3d, xyz = cache['feat_2d'], cache['feat_3d'], cache['xyz']   # (B, D, H, W), (B, N1, D), (B, N1, 3)
        feat_2d = rearrange(feat_2d, 'b d h w -> b (h w) d')  # (B, D, h, w) -> (B, h*w, D)
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        xyz_tf = se3_transform(Tcl, xyz.transpose(-1, -2))  # (B, 4, 4) x (B, 3, N) -> (B, 3, N)
        feat_camera_info = camera_info.copy()
        feat_h, feat_w = feat_2d.shape[-2:] # (B, D, h, w)
        assert feat_h == self.feat_h and feat_w == self.feat_w
        kx = feat_w / sensor_w
        ky = feat_h / sensor_h
        feat_camera_info.update({
            'sensor_w': feat_w,
            'sensor_h': feat_h,
            'fx': kx * camera_info['fx'],
            'fy': ky * camera_info['fy'],
            'cx': kx * camera_info['cx'],
            'cy': kx * camera_info['cy']
        })
        proj_uv = project_pc2image(xyz_tf, feat_camera_info)  # (B, 2, N)
        bnd_ratio = [-1 - self.margin, 1 + self.margin]
        proj_uv[:,:,0] = normalize_grid(proj_uv[:,:,0], feat_w).clamp(*bnd_ratio)  # (B, N)  clamp to supress nan
        proj_uv[:,:,1] = normalize_grid(proj_uv[:,:,1], feat_h).clamp(*bnd_ratio)  # (B, N)
        interp_2d = self.fusion(proj_uv.transpose(-1, -2).contiguous(), feat_2d.shape, feat_3d.detach())  # (B, C, H, W)
        fusion_map = self.head(torch.cat([feat_2d, interp_2d], dim=1))  # (B, C1, H, W), (B, C2, H, W) -> (B, C1+C2, H, W)
        if self.output_type == '2d':
            return fusion_map  # (B, D, H, W)
        elif self.output_type == '1d':
            return rearrange(fusion_map, 'b c h w -> b (h w) c')  # (B, H*W, D)
        else:
            raise NotImplementedError("output_type must be '1d' or '2d', got {}".format(self.output_type))

class PoolDualFusionNet(PoolFusionNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _lazy_init(self):
        self.rot_head = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True)
        )
        
        self.tsl_head = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True)
        )

    def cache_forward(self, cache: StateCache, Tcl: torch.Tensor, 
            camera_info: BatchedCameraInfoDict) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """forward through cache/observation

        Args:
            cache (StateCache): cache of env
            Tcl (torch.Tensor): groups of extrans [B, 4, 4]
            camera_info (BatchedCameraInfoDict): intran information for projection

        Returns:
            fusion_map: (B, C, h, w) or (B, h*w, C)
        """
        feat_2d, feat_3d, xyz = cache['feat_2d'], cache['feat_3d'], cache['xyz']   # (B, D, H, W), (B, N1, D), (B, N1, 3)
        feat_2d = rearrange(feat_2d, 'b d h w -> b (h w) d')  # (B, D, h, w) -> (B, h*w, D)
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        xyz_tf = se3_transform(Tcl, xyz.transpose(-1, -2))  # (B, 4, 4) x (B, 3, N) -> (B, 3, N)
        feat_camera_info = camera_info.copy()
        feat_h, feat_w = feat_2d.shape[-2:] # (B, D, h, w)
        assert feat_h == self.feat_h and feat_w == self.feat_w
        kx = feat_w / sensor_w
        ky = feat_h / sensor_h
        feat_camera_info.update({
            'sensor_w': feat_w,
            'sensor_h': feat_h,
            'fx': kx * camera_info['fx'],
            'fy': ky * camera_info['fy'],
            'cx': kx * camera_info['cx'],
            'cy': kx * camera_info['cy']
        })
        proj_uv = project_pc2image(xyz_tf, feat_camera_info)  # (B, 2, N)
        bnd_ratio = [-1 - self.margin, 1 + self.margin]
        proj_uv[:,:,0] = normalize_grid(proj_uv[:,:,0], feat_w).clamp(*bnd_ratio)  # (B, N)  clamp to supress nan
        proj_uv[:,:,1] = normalize_grid(proj_uv[:,:,1], feat_h).clamp(*bnd_ratio)  # (B, N)
        interp_2d = self.fusion(proj_uv.transpose(-1, -2).contiguous(), feat_2d.shape, feat_3d.detach())  # (B, C, H, W)
        fusion_map = self.head(torch.cat([feat_2d, interp_2d], dim=1))  # (B, C1, H, W), (B, C2, H, W) -> (B, C1+C2, H, W)
        
        # 2. Split Heads
        rot_feat = self.rot_head(fusion_map)
        tsl_feat = self.tsl_head(fusion_map)
        
        # 3. Format Output
        if self.output_type == '2d':
            return rot_feat, tsl_feat
        elif self.output_type == '1d':
            return (rearrange(rot_feat, 'b c h w -> b (h w) c'),
                    rearrange(tsl_feat, 'b c h w -> b (h w) c'))
        else:
            raise NotImplementedError("output_type must be '1d' or '2d', got {}".format(self.output_type))

        
__FUSION_NET_DICT__ = {"AttenFusionNet": AttenFusionNet,
                       "DepthFusionNet": DepthFusionNet,
                       "ConcatFusionNet": ConcatFusionNet}

__DUAL_FUSION_NET_DICT__ = {"AttenDualFusionNet": AttenDualFusionNet,
                          "DepthDualFusionNet": DepthDualFusionNet,
                          "ConcatDualFusionNet": ConcatDualFusionNet}

__FUSION_NET__ = TypeVar("__FUSION_NET__", AttenFusionNet, DepthFusionNet, ConcatFusionNet)
__DUAL_FUSION_NET__ = TypeVar("__DUAL_FUSION_NET__", AttenDualFusionNet, DepthDualFusionNet, ConcatDualFusionNet)