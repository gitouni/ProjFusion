import torch
import torch.nn as nn
from typing import Dict
from contextlib import contextmanager
from .tools.core import get_activation_func
from .tools.core import AttenFusionNet, AttenDualFusionNet, MLPHead
from .tools.aggregation import __AGGREGATION_DICT__, __AGGREGATION__
from .util.constant import *

def xavier_init_mlp(m:nn.Module):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class ProjFusion(nn.Module):
    # class for Energy-based Embedding
    def __init__(self, encoder_argv: Dict,  aggregation_argv: Dict, mlp_argv: Dict, disentangle: bool = False) -> None:
        super().__init__()
        self.encoder = AttenFusionNet(**encoder_argv)
        getattr(self.encoder, '_lazy_init', lambda: None)()  # 调用函数的延迟初始化方法
        if "activation_fn" in aggregation_argv['args']:
            aggregation_argv['args']['activation_fn'] = get_activation_func(**aggregation_argv['args']['activation_fn'])
        aggregation_argv['args'].update(self.encoder.kargv_for_aggregation())
        
        mlp_argv['input_dim'] = self.aggregation.out_dim
        mlp_argv['output_dim'] = 3  # rot and mlp output dim
        if "activation_fn" in mlp_argv:
            mlp_argv['activation_fn'] = get_activation_func(**mlp_argv['activation_fn'])
        self.aggregation_argv = aggregation_argv
        self.mlp_argv = mlp_argv
        self.register_buffer('disentangle', torch.zeros((), dtype=torch.bool), persistent=True)
        

    def _lazy_init(self):
        self.aggregation: __AGGREGATION__ = __AGGREGATION_DICT__[self.aggregation_argv['type']](**self.aggregation_argv)
        self.rot_mlp = MLPHead(**self.mlp_argv)
        self.tsl_mlp = MLPHead(**self.mlp_argv)
        self.rot_mlp.apply(xavier_init_mlp)
        self.tsl_mlp.apply(xavier_init_mlp)
    
    def forward(self, img: torch.Tensor, pcd: torch.Tensor, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        """state embedding forward

        Args:
            img (torch.Tensor): [B, 3, H, W]
            pcd (torch.Tensor): [B, N, 3]
            Tcl (torch.Tensor): [B, 4, 4]
            camera_info (CameraInfoDict): parameters of the camera intrinsic matrix

        Returns:
            x0: (B, F) embeddings of fused features
        """
        x = self.encoder(img, pcd, Tcl, camera_info)  # (B, D, h, w)
        x = self.aggregation(x)  # (B, D)
        rot = self.rot_mlp(x)
        tsl = self.tsl_mlp(x)
        return rot, tsl  # (B, 3), (B, 3)
    

    def transform(self, Tcl: torch.Tensor, pred: torch.Tensor):
        disentangle = self.disentangle.item()
        if not disentangle:
            return torch.matmul(pred, Tcl)
        else:
            res = torch.eye(4).to(Tcl)
            res[..., :3, :3] = pred[..., :3, :3] @ Tcl[..., :3, :3]
            res[..., :3, 3] = pred[..., :3, 3] + Tcl[..., :3, 3]
            return res
    
    @contextmanager
    def cache_manager(self, img: torch.Tensor, pcd: torch.Tensor):
        """leverage buffer provided by itself

        Args:
            img (torch.Tensor): (B, 3, H, W)
            pcd (torch.Tensor): (B, N, 3)
        """
        try:
            self.encoder.store_buffer(img, pcd)
            yield
        finally:
            self.encoder.clear_buffer()


class ProjDualFusion(ProjFusion):
    # class for Energy-based Embedding
    def __init__(self, **argv) -> None:
        super().__init__(**argv)
    
    def _lazy_init(self):
        self.rot_aggregation: __AGGREGATION__ = __AGGREGATION_DICT__[self.aggregation_argv['type']](**self.aggregation_argv['args'])
        self.tsl_aggregation: __AGGREGATION__ = __AGGREGATION_DICT__[self.aggregation_argv['type']](**self.aggregation_argv['args'])
        self.rot_mlp = MLPHead(**self.mlp_argv)
        self.tsl_mlp = MLPHead(**self.mlp_argv)
        self.rot_mlp.apply(xavier_init_mlp)
        self.tsl_mlp.apply(xavier_init_mlp)

    def forward(self, img: torch.Tensor, pcd: torch.Tensor, Tcl: torch.Tensor, camera_info: BatchedCameraInfoDict):
        """state embedding forward

        Args:
            img (torch.Tensor): [B, 3, H, W]
            pcd (torch.Tensor): [B, N, 3]
            Tcl (torch.Tensor): [B, 4, 4]
            camera_info (CameraInfoDict): parameters of the camera intrinsic matrix

        Returns:
            x0: (B, F) embeddings of fused features
        """
        rot_x, tsl_x = self.encoder(img, pcd, Tcl, camera_info)  # (B, D, h, w)
        rot_x = self.rot_aggregation(rot_x)  # (B, D)
        tsl_x = self.tsl_aggregation(tsl_x)  # (B, D)
        rot = self.rot_mlp(rot_x)
        tsl = self.tsl_mlp(tsl_x)
        return rot, tsl  # (B, 3), (B, 3)