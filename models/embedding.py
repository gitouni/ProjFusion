import torch.nn as nn
import torch
from torch.nn import functional as F
import enum
from einops import rearrange
from typing import Dict, Literal, Union
from contextlib import contextmanager
from .tools.aggregation import __AGGREGATION_DICT__, __AGGREGATION__
from .tools.core import __ENCODER_DICT__, __ENCODER__, StateCache, get_activation_func
from .util.constant import CameraInfoDict, PredictMode
from typing import Tuple


class StateEmbedding(nn.Module):
    # class for Energy-based Embedding
    def __init__(self,
                encoder_type:Literal['PoolFusionNet','PoolDualFusionNet',"AttenFusionNet"], encoder_argv:Dict,
                aggregation_type:Literal['MiniResAggregation','ResAggregation', "AttentionAggregation"], aggregation_argv:Dict) -> None:
        super().__init__()
        encoder_class = __ENCODER_DICT__[encoder_type]
        aggregation_class = __AGGREGATION_DICT__[aggregation_type]
        self.encoder:__ENCODER__ = encoder_class(**encoder_argv)
        getattr(self.encoder, '_lazy_init', lambda: None)()  # 调用函数的延迟初始化方法
        if "activation_fn" in aggregation_argv:
            aggregation_argv['activation_fn'] = get_activation_func(**aggregation_argv['activation_fn'])
        self.aggregation: __AGGREGATION__ = aggregation_class(**self.encoder.kargv_for_aggregation(), **aggregation_argv)
        self.out_dim = self.aggregation.out_dim

    def mlp_forward(self, feat:torch.Tensor, G:int):
        x0 = self.aggregation(feat)
        x0 = F.normalize(x0, p=2, dim=-1)   # normalize embedding to enhance ce loss stability
        # return x0
        return rearrange(x0, '(b g) ... -> b g ...',g=G)  # (B, G, F)
    
    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:CameraInfoDict):
        """state embedding forward

        Args:
            img (torch.Tensor): [B, 3, H, W]
            pcd (torch.Tensor): [B, N, 3]
            Tcl (torch.Tensor): [B, G, 4, 4], [:, 0, :, :] is the GT SE(3)
            camera_info (CameraInfoDict): parameters of the camera intrinsic matrix

        Returns:
            x0: (B, F) embeddings of fused features
        """
        feat = self.encoder(img, pcd, Tcl, camera_info)  # (B, D)
        if Tcl.ndim == 4:
            G = Tcl.shape[1]
        else:
            G = 1
        return self.mlp_forward(feat, G)
    
    def cache_forward(self, cache:StateCache, Tcl:torch.Tensor, camera_info:CameraInfoDict):
        feat = self.encoder.cache_forward(cache, Tcl, camera_info)
        if Tcl.ndim == 4:
            G = Tcl.shape[1]
        else:
            G = 1
        return self.mlp_forward(feat, G)
    
    def encoder_cache(self, img:torch.Tensor, pcd:torch.Tensor) -> StateCache:
        return self.encoder.encoder_cache(img, pcd)  # usually in @torch.no_grad()
    
    def store_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        self.encoder.store_buffer(img, pcd)

    def store_buffer_direct(self, cache:Union[Dict[str, torch.Tensor], None]):
        if cache is not None:
            self.encoder.store_buffer_direct(cache)

    def get_buffers(self):
        return self.encoder.get_buffers()

    def clear_buffer(self):
        self.encoder.clear_buffer()


    @contextmanager
    def model_buffer_manager(self, img:torch.Tensor, pcd:torch.Tensor):
        """leverage buffer provided by itself

        Args:
            img (torch.Tensor): (B, 3, H, W)
            pcd (torch.Tensor): (B, N, 3)
        """
        try:
            self.store_buffer(img, pcd)
            yield
        finally:
            self.clear_buffer()
    
    @contextmanager
    def env_buffer_manger(self, cache:StateCache):
        """leverage buffer provided by the environment

        Args:
            cache:StateCache
        """
        try:
            self.store_buffer_direct(cache)
            yield
        finally:
            self.clear_buffer()


class DualStateEmbedding(nn.Module):
    # class for Rot-Tsl-interleaved Energy-based Embedding
    def __init__(self,
                encoder_type:Literal['PoolFusionNet','PoolDualFusionNet',"AttenFusionNet", "AttenDualFusionNet", "AttenDualSplitFusionNet"], encoder_argv:Dict,
                aggregation_type:Literal['MiniResAggregation','ResAggregation', "AttentionAggregation"], aggregation_argv:Dict) -> None:
        super().__init__()
        encoder_class = __ENCODER_DICT__[encoder_type]
        aggregation_class = __AGGREGATION_DICT__[aggregation_type]
        self.encoder:__ENCODER__ = encoder_class(**encoder_argv)
        getattr(self.encoder, '_lazy_init', lambda: None)()  # 调用函数的延迟初始化方法
        if "activation_fn" in aggregation_argv:
            aggregation_argv['activation_fn'] = get_activation_func(**aggregation_argv['activation_fn'])
        self.rot_aggregation: __AGGREGATION__ = aggregation_class(**self.encoder.kargv_for_aggregation(), **aggregation_argv)
        self.tsl_aggregation: __AGGREGATION__ = aggregation_class(**self.encoder.kargv_for_aggregation(), **aggregation_argv)
        self.out_dim = self.rot_aggregation.out_dim

    def mlp_forward(self, feat:Tuple[torch.Tensor, torch.Tensor], G:int, Mode:PredictMode):
        if Mode != PredictMode.Both:
            if Mode == PredictMode.RotOnly:
                x0 = self.rot_aggregation(feat)
            elif Mode == PredictMode.TslOnly:
                x0 = self.tsl_aggregation(feat)
            else:
                raise NotImplementedError("Mode must be in {}, got {}".format(list(PredictMode), Mode))
            x0 = F.normalize(x0, p=2, dim=-1)   # normalize embedding to enhance ce loss stability
            # return x0
            return rearrange(x0, '(b g) ... -> b g ...',g=G)  # (B, G, F)
        else:  # rotation and translation aggregation share the same fusion feature
            rot_x0 = self.rot_aggregation(feat)
            tsl_x0 = self.tsl_aggregation(feat)
            rot_x0 = F.normalize(rot_x0, p=2, dim=-1)   # normalize embedding to enhance ce loss stability
            tsl_x0 = F.normalize(tsl_x0, p=2, dim=-1)   # normalize embedding to enhance ce loss stability
            # return rot_x0, tsl_x0
            return rearrange(rot_x0, '(b g) ... -> b g ...',g=G), rearrange(tsl_x0, '(b g) ... -> b g ...',g=G)  # (B, G, F)
        
    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:CameraInfoDict, Mode:PredictMode):
        """state embedding forward

        Args:
            img (torch.Tensor): [B, 3, H, W]
            pcd (torch.Tensor): [B, N, 3]
            Tcl (torch.Tensor): [B, G, 4, 4], [:, 0, :, :] is the GT SE(3)
            camera_info (CameraInfoDict): parameters of the camera intrinsic matrix

        Returns:
            x0: (B, F) embeddings of fused features
        """
        feat = self.encoder(img, pcd, Tcl, camera_info, Mode)  # (B, D)
        if Tcl.ndim == 4:
            G = Tcl.shape[1]
        else:
            G = 1
        return self.mlp_forward(feat, G, Mode)

    def cache_forward(self, cache:StateCache, Tcl:torch.Tensor, camera_info:CameraInfoDict, Mode:PredictMode):
        feat = self.encoder.cache_forward(cache, Tcl, camera_info)
        if Tcl.ndim == 4:
            G = Tcl.shape[1]
        else:
            G = 1
        return self.mlp_forward(feat, G, Mode)
    
    def encoder_cache(self, img:torch.Tensor, pcd:torch.Tensor) -> StateCache:
        return self.encoder.encoder_cache(img, pcd)  # usually in @torch.no_grad()
    
    def store_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        self.encoder.store_buffer(img, pcd)

    def store_buffer_direct(self, cache:Union[Dict[str, torch.Tensor], None]):
        if cache is not None:
            self.encoder.store_buffer_direct(cache)

    def get_buffers(self):
        return self.encoder.get_buffers()

    def clear_buffer(self):
        self.encoder.clear_buffer()


    @contextmanager
    def model_buffer_manager(self, img:torch.Tensor, pcd:torch.Tensor):
        """leverage buffer provided by itself

        Args:
            img (torch.Tensor): (B, 3, H, W)
            pcd (torch.Tensor): (B, N, 3)
        """
        try:
            self.store_buffer(img, pcd)
            yield
        finally:
            self.clear_buffer()
    
    @contextmanager
    def env_buffer_manger(self, cache:StateCache):
        """leverage buffer provided by the environment

        Args:
            cache:StateCache
        """
        try:
            self.store_buffer_direct(cache)
            yield
        finally:
            self.clear_buffer()


class DualStateEmbeddingSplitAgg(nn.Module):
    # class for Rot-Tsl-interleaved Energy-based Embedding
    def __init__(self,
                encoder_type:Literal['PoolFusionNet','PoolDualFusionNet',"AttenFusionNet"], encoder_argv:Dict,
                aggregation_type:Literal['MiniResAggregation','ResAggregation', "AttentionAggregation"], aggregation_argv:Dict) -> None:
        super().__init__()
        encoder_class = __ENCODER_DICT__[encoder_type]
        aggregation_class = __AGGREGATION_DICT__[aggregation_type]
        self.encoder:__ENCODER__ = encoder_class(**encoder_argv)
        if "activation_fn" in aggregation_argv:
            aggregation_argv['activation_fn'] = get_activation_func(**aggregation_argv['activation_fn'])
        self.rot_aggregation: __AGGREGATION__ = aggregation_class(**self.encoder.kargv_for_aggregation(), **aggregation_argv)
        self.tsl_aggregation: __AGGREGATION__ = aggregation_class(**self.encoder.kargv_for_aggregation(), **aggregation_argv)
        self.out_dim = self.rot_aggregation.out_dim

    def mlp_forward(self, feat:Tuple[torch.Tensor, torch.Tensor], G:int, Mode:PredictMode):
        if Mode == PredictMode.Both:
            rot_feat, tsl_feat = feat
        elif Mode == PredictMode.RotOnly:
            rot_feat = feat
        elif Mode == PredictMode.TslOnly:
            tsl_feat = feat
        else:
            pass
        if Mode != PredictMode.Both:
            if Mode == PredictMode.RotOnly:
                x0 = self.rot_aggregation(rot_feat)
            elif Mode == PredictMode.TslOnly:
                x0 = self.tsl_aggregation(tsl_feat)
            else:
                raise NotImplementedError("Mode must be in {}, got {}".format(list(PredictMode), Mode))
            x0 = F.normalize(x0, p=2, dim=-1)   # normalize embedding to enhance ce loss stability
            return rearrange(x0, '(b g) ... -> b g ...',g=G)  # (B, G, F)
        else:  # rotation and translation aggregation share the same fusion feature
            rot_x0 = self.rot_aggregation(rot_feat)
            tsl_x0 = self.tsl_aggregation(tsl_feat)
            rot_x0 = F.normalize(rot_x0, p=2, dim=-1)   # normalize embedding to enhance ce loss stability
            tsl_x0 = F.normalize(tsl_x0, p=2, dim=-1)   # normalize embedding to enhance ce loss stability
            return rearrange(rot_x0, '(b g) ... -> b g ...',g=G), rearrange(tsl_x0, '(b g) ... -> b g ...',g=G)  # (B, G, F)
        
    def forward(self, img:torch.Tensor, pcd:torch.Tensor, Tcl:torch.Tensor, camera_info:CameraInfoDict, Mode:PredictMode):
        """state embedding forward

        Args:
            img (torch.Tensor): [B, 3, H, W]
            pcd (torch.Tensor): [B, N, 3]
            Tcl (torch.Tensor): [B, G, 4, 4], [:, 0, :, :] is the GT SE(3)
            camera_info (CameraInfoDict): parameters of the camera intrinsic matrix

        Returns:
            x0: (B, F) embeddings of fused features
        """
        feat = self.encoder(img, pcd, Tcl, camera_info, Mode)  # (B, D)
        if Tcl.ndim == 4:
            G = Tcl.shape[1]
        else:
            G = 1
        return self.mlp_forward(feat, G, Mode)

    def cache_forward(self, cache:StateCache, Tcl:torch.Tensor, camera_info:CameraInfoDict, Mode:PredictMode):
        feat = self.encoder.cache_forward(cache, Tcl, camera_info)
        if Tcl.ndim == 4:
            G = Tcl.shape[1]
        else:
            G = 1
        return self.mlp_forward(feat, G, Mode)
    
    def encoder_cache(self, img:torch.Tensor, pcd:torch.Tensor) -> StateCache:
        return self.encoder.encoder_cache(img, pcd)  # usually in @torch.no_grad()
    
    def store_buffer(self, img:torch.Tensor, pcd:torch.Tensor):
        self.encoder.store_buffer(img, pcd)

    def store_buffer_direct(self, cache:Union[Dict[str, torch.Tensor], None]):
        if cache is not None:
            self.encoder.store_buffer_direct(cache)

    def get_buffers(self):
        return self.encoder.get_buffers()

    def clear_buffer(self):
        self.encoder.clear_buffer()


    @contextmanager
    def model_buffer_manager(self, img:torch.Tensor, pcd:torch.Tensor):
        """leverage buffer provided by itself

        Args:
            img (torch.Tensor): (B, 3, H, W)
            pcd (torch.Tensor): (B, N, 3)
        """
        try:
            self.store_buffer(img, pcd)
            yield
        finally:
            self.clear_buffer()
    
    @contextmanager
    def env_buffer_manger(self, cache:StateCache):
        """leverage buffer provided by the environment

        Args:
            cache:StateCache
        """
        try:
            self.store_buffer_direct(cache)
            yield
        finally:
            self.clear_buffer()