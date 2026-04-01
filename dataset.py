import os
from pathlib import Path
import json
import torch
from PIL import Image
from torchvision.transforms import transforms as Tf
import numpy as np
# kitti dependences
import pykitti
# nuscenes dependences
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.map_mask import MapMask
# from nuscenes.utils.color_map import get_colormap
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
import time
import open3d as o3d
from PIL import Image
from torch import Generator, randperm
from torch.utils.data import Dataset, Subset, BatchSampler
from typing import Iterable, Generator, List, Dict, Union, Optional, Tuple, Sequence, Literal, TypeVar
from models.tools.csrc import furthest_point_sampling
from models.util import transform, se3
from models.util.transform import nptran, inv_pose_np
from models.util.constant import IMAGENET_DEFAULT_MEAN as IMAGENET_MEAN
from models.util.constant import IMAGENET_DEFAULT_STD as IMAGENET_STD
import re
from copy import deepcopy
from models.util.constant import *

def retrieval_dict(camera_info:BatchedCameraInfoDict, batch_idx:torch.LongTensor) -> BatchedCameraInfoDict:
    copied_camera_info = deepcopy(camera_info)
    for key, value in copied_camera_info.items():
        if isinstance(value, torch.Tensor):
            copied_camera_info[key] = value[batch_idx.to(value.device)]
    return copied_camera_info


def subset_split(dataset:Dataset, lengths:Sequence[int], seed:Optional[int]=None):
	"""
	split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
	"""
	if seed is not None:
		generator = Generator().manual_seed(seed)
	else:
		generator = None
	indices = randperm(sum(lengths), generator=generator).tolist()
	Subsets = []
	for offset, length in zip(np.add.accumulate(lengths), lengths):
		if length == 0:
			Subsets.append(None)
		else:
			Subsets.append(Subset(dataset, indices[offset - length : offset]))
	return Subsets

def check_length(root:str,save_name='data_len.json'):
    seq_dir = os.path.join(root,'sequences')
    seq_list = os.listdir(seq_dir)
    seq_list.sort()
    dict_len = dict()
    for seq in seq_list:
        len_velo = len(os.listdir(os.path.join(seq_dir,seq,'velodyne')))
        dict_len[seq]=len_velo
    with open(os.path.join(root,save_name),'w')as f:
        json.dump(dict_len,f)
        
class KITTIFilter:
    def __init__(self, voxel_size:Optional[float]=None, positive_x:bool=True, min_dist:float=0.15, skip_point:int=1):
        """KITTIFilter

        Args:
            voxel_size (float, optional): voxel size for downsampling. Defaults to 0.3.
            concat (str, optional): concat operation for normal estimation, 'none','xyz' or 'zero-mean'. Defaults to 'none'.
        """
        self.voxel_size = voxel_size
        self.min_dist = min_dist
        self.skip_point = skip_point
        self.positive_x = positive_x
        
    def __call__(self, x:np.ndarray):
        if self.skip_point > 1:
            x = x[::self.skip_point,:]
        rev_x = np.linalg.norm(x, axis=1) > self.min_dist
        if self.positive_x:
            rev_x = np.logical_and(rev_x, x[:,0] > 0)
        
        # _, ind = pcd.remove_radius_outlier(nb_points=self.n_neighbor, radius=self.voxel_size)
        # pcd.select_by_index(ind)
        if self.voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(x[rev_x,:])
            pcd = pcd.voxel_down_sample(self.voxel_size)
            pcd_xyz = np.array(pcd.points,dtype=np.float32)
        else:
            pcd_xyz = x[rev_x,:]
        return pcd_xyz

class Resampler:
    """ [N, D] -> [M, D]\n
    used for training
    """
    def __init__(self, num):
        self.num = num

    def __call__(self, x: np.ndarray):
        num_points = x.shape[0]
        idx = np.random.permutation(num_points)
        if self.num < 0:
            return x[idx]
        elif self.num <= num_points:
            idx = idx[:self.num] # (self.num,dim)
            return x[idx]
        else:
            idx = np.hstack([idx,np.random.choice(num_points,self.num-num_points,replace=True)]) # (self.num,dim)
            return x[idx]
        

class MaxResampler:
    """ [N, D] -> [M, D] (M<=max_num)\n
    used for testing
    """
    def __init__(self,num):
        self.num = num
    def __call__(self, x:np.ndarray):
        num_points = x.shape[0]
        x_ = np.random.permutation(x)
        if num_points <= self.num:
            return x_  # permutation
        else:
            return x_[:self.num]

class ToTensor:
    def __init__(self,type=torch.float):
        self.tensor_type = type
    
    def __call__(self, x: np.ndarray):
        return torch.from_numpy(x).type(self.tensor_type)


class SeqBatchSampler(BatchSampler):
    def __init__(self, num_sequences:int, len_of_sequences:Sequence[int], dataset_len:int, num_samples:int=4):
        # Batch sampler with a dynamic number of sequences
        # max_images >= number_of_sequences * images_per_sequence
        assert num_sequences == len(len_of_sequences)
        self.num_samples = num_samples
        self.num_sequences = num_sequences
        self.len_of_sequences = len_of_sequences
        self.dataset_len = dataset_len

    def __iter__(self):
        for _ in range(self.dataset_len):
            # number per sequence
            seq_indices = np.random.choice(self.num_sequences, self.num_samples, replace=True)
            batches = [(seq_idx, np.random.choice(self.len_of_sequences[seq_idx], 1).item()) for seq_idx in seq_indices]
            yield batches

    def __len__(self):
        return self.dataset_len

class BaseKITTIDataset(Dataset):
    def __init__(self,basedir:str,
                 seqs:List[str]=['09','10'], cam_id:int=2,
                 meta_json:str='data_len.json', skip_frame:int=1, skip_point:int=1,
                 voxel_size:Optional[float]=None, min_dist=0.1, pcd_sample_num=8192, fps=False,
                 resize_size:Optional[Tuple[int,int]]=None, extend_ratio=(2.5,2.5),
                 ):
        if not os.path.exists(os.path.join(basedir,meta_json)):
            check_length(basedir,meta_json)
        with open(os.path.join(basedir,meta_json),'r')as f:
            dict_len = json.load(f)
        frame_list = []
        for seq in seqs:
            frame = list(range(0,dict_len[seq],skip_frame))
            # cut_index = len(frame) % batch_size
            # if cut_index > 0:
            #     frame = frame[:-cut_index]  # to be flexible to pykitti
            frame_list.append(frame)
        self.kitti_datalist = [pykitti.odometry(basedir,seq,frames=frame) for seq,frame in zip(seqs,frame_list)]  
        # concat images from different seq into one batch will cause error
        self.cam_id = cam_id
        self.resize_size = resize_size
        for seq,obj in zip(seqs,self.kitti_datalist):
            self.check(obj,cam_id,seq)
        self.sep = [len(data) for data in self.kitti_datalist]
        self.sumsep = np.cumsum(self.sep)
        if fps:
            self.resample_tran = lambda pcd: pcd[furthest_point_sampling(pcd, n_samples=pcd_sample_num, cpp_impl=True), :]
        else:
            self.resample_tran = Resampler(pcd_sample_num)
        self.tensor_tran = lambda x:torch.from_numpy(x).to(torch.float32)
        self.img_tran = Tf.Compose([Tf.ToTensor(),
                                    Tf.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.pcd_tran = KITTIFilter(voxel_size, min_dist=min_dist, skip_point=skip_point)
        self.extend_ratio = extend_ratio
        
    def __len__(self):
        return self.sumsep[-1]
    
    def get_seq_params(self) -> Tuple[int, List[int]]:
        """Get num_seqs, num_data_per_seq

        Returns:
            Tuple[int, List[int]]: num_seqs, num_data_per_seq
        """
        return len(self.sep), self.sep
    
    @staticmethod
    def check(odom_obj:pykitti.odometry,cam_id:int,seq:str)->bool:
        calib = odom_obj.calib
        cam_files_length = len(getattr(odom_obj,'cam%d_files'%cam_id))
        velo_files_lenght = len(odom_obj.velo_files)
        head_msg = '[Seq %s]:'%seq
        assert cam_files_length>0, head_msg+'None of camera %d files'%cam_id
        assert cam_files_length==velo_files_lenght, head_msg+"number of cam %d (%d) and velo files (%d) doesn't equal!"%(cam_id,cam_files_length,velo_files_lenght)
        assert hasattr(calib,'T_cam0_velo'), head_msg+"Crucial calib attribute 'T_cam0_velo' doesn't exist!"
        
    def __getitem__(self, index:Union[int, Tuple[int,int]]) -> BaseDatasetOutput:
        if isinstance(index, Tuple):
            return self.group_sub_item(index)
        group_id = np.digitize(index,self.sumsep,right=False).item()
        if group_id > 0:
            sub_idx = index - self.sumsep[group_id-1]
        else:
            sub_idx = index
        return self.group_sub_item((group_id, sub_idx))
        
    def group_sub_item(self, tuple_index:Tuple[int,int]) -> BaseDatasetOutput:
        group_idx, sub_idx = tuple_index
        assert group_idx < len(self.sep) and sub_idx < self.sep[group_idx], 'Invalid group_idx ({}), sub_dix ({})'.format(group_idx, sub_idx)
        data = self.kitti_datalist[group_idx]
        T_cam2velo = getattr(data.calib,'T_cam%d_velo'%self.cam_id)  
        raw_img:Image.Image = getattr(data,'get_cam%d'%self.cam_id)(sub_idx)  # PIL Image
        H,W = raw_img.height, raw_img.width
        K_cam:np.ndarray = getattr(data.calib,'K_cam%d'%self.cam_id)  
        if self.resize_size is not None:
            RH, RW = self.resize_size
            K_cam = np.diag([RW / W, RH / H, 1.0]) @ K_cam
        else:
            RH = H
            RW = W
       
        raw_img = raw_img.resize([RW,RH],Image.Resampling.BILINEAR)
        _img = self.img_tran(raw_img)  # raw img input (3,H,W)
        pcd:np.ndarray = data.get_velo(sub_idx)[:,:3]
        pcd = self.pcd_tran(pcd)
        if self.extend_ratio is not None:
            calibed_pcd = nptran(pcd, T_cam2velo).T
            REVH,REVW = self.extend_ratio[0]*RH,self.extend_ratio[1] * RW
            K_cam_extend = K_cam.copy()  # K_cam_extend for dilated projection
            K_cam_extend[0,-1] *= self.extend_ratio[0]
            K_cam_extend[1,-1] *= self.extend_ratio[1]
            *_,rev = transform.binary_projection((REVH,REVW), K_cam_extend, calibed_pcd)
            pcd = pcd[rev,:]

        pcd = self.resample_tran(pcd) # (n, 3)
        _pcd = self.tensor_tran(pcd)  # (n, 3)  easier for transformation
        T_cam2velo = self.tensor_tran(T_cam2velo)
        camera_info = {
            "fx": K_cam[0,0].item(),
            "fy": K_cam[1,1].item(),
            "cx": K_cam[0,2].item(),
            "cy": K_cam[1,2].item(),
            "sensor_h": RH,
            "sensor_w": RW,
            "projection_mode": "perspective"
        }
        return dict(img=_img,pcd=_pcd, camera_info=camera_info, extran=T_cam2velo, group_idx=group_idx, sub_idx=sub_idx)
    
    @staticmethod
    def collate_fn(zipped_x:Iterable[BaseDatasetOutput]) -> BatchedBaseDatasetOutput:
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['extran'] = torch.stack([x['extran'] for x in zipped_x])
        batch['group_idx'] = [x['group_idx'] for x in zipped_x]
        batch['sub_idx'] = [x['sub_idx'] for x in zipped_x]
        batch['camera_info'] = zipped_x[0]['camera_info']
        batch['camera_info']['fx'] = torch.tensor([x['camera_info']['fx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['fy'] = torch.tensor([x['camera_info']['fy'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cx'] = torch.tensor([x['camera_info']['cx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cy'] = torch.tensor([x['camera_info']['cy'] for x in zipped_x], dtype=torch.float32)
        return batch
    
    @staticmethod
    def concat_fn(pretend_batch:BatchedBaseDatasetOutput, append_batch:BatchedBaseDatasetOutput):
        batch = dict()
        for key in pretend_batch.keys():
            pretend_value = pretend_batch[key]
            append_value = append_batch[key]
            if isinstance(pretend_value, torch.Tensor):
                batch[key] = torch.cat([pretend_value, append_value], dim=0)
            elif isinstance(pretend_value, List):
                batch[key] = pretend_value + append_value
            elif isinstance(pretend_value, Dict):
                batch[key] = dict()
                for sub_key in pretend_value.keys():
                    batch[key][sub_key] = torch.cat([pretend_value[sub_key], append_value[sub_key]], dim=0)
            else:
                raise TypeError("unkown type:{}".format(type(pretend_value)))


class BaseKITTIIMGDataset(Dataset):
    def __init__(self, basedir: str,
                 seqs: List[str] = ['09', '10'],
                 cam_id: int = 2,
                 meta_json: str = 'data_len.json',
                 skip_frame: int = 1,
                 resize_size: Optional[Tuple[int, int]] = None, **argv):
        if not os.path.exists(os.path.join(basedir, meta_json)):
            check_length(basedir, meta_json)
        with open(os.path.join(basedir, meta_json), 'r') as f:
            dict_len = json.load(f)
        frame_list = []
        for seq in seqs:
            frame = list(range(0, dict_len[seq], skip_frame))
            frame_list.append(frame)
        self.kitti_datalist = [
            pykitti.odometry(basedir, seq, frames=frame)
            for seq, frame in zip(seqs, frame_list)
        ]
        self.cam_id = cam_id
        self.resize_size = resize_size
        for seq, obj in zip(seqs, self.kitti_datalist):
            self.check(obj, cam_id, seq)
        self.sep = [len(data) for data in self.kitti_datalist]
        self.sumsep = np.cumsum(self.sep)
        self.img_tran = lambda img: torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2,0,1)

    def __len__(self):
        return self.sumsep[-1]

    def get_seq_params(self) -> Tuple[int, List[int]]:
        return len(self.sep), self.sep

    @staticmethod
    def check(odom_obj: pykitti.odometry, cam_id: int, seq: str) -> bool:
        cam_files_length = len(getattr(odom_obj, 'cam%d_files' % cam_id))
        head_msg = '[Seq %s]:' % seq
        assert cam_files_length > 0, head_msg + 'None of camera %d files' % cam_id

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        if isinstance(index, Tuple):
            return self.group_sub_item(index)
        group_id = np.digitize(index, self.sumsep, right=False).item()
        sub_idx = index - self.sumsep[group_id-1] if group_id > 0 else index
        return self.group_sub_item((group_id, sub_idx))

    def group_sub_item(self, tuple_index: Tuple[int, int]):
        group_idx, sub_idx = tuple_index
        assert group_idx < len(self.sep) and sub_idx < self.sep[group_idx], \
            'Invalid group_idx (%d), sub_idx (%d)' % (group_idx, sub_idx)
        data = self.kitti_datalist[group_idx]
        # 获取指定摄像头的图像（PIL Image）
        raw_img: Image.Image = getattr(data, 'get_cam%d' % self.cam_id)(sub_idx)
        H, W = raw_img.height, raw_img.width
        if self.resize_size is not None:
            RH, RW = self.resize_size
        else:
            RH, RW = H, W
        raw_img = raw_img.resize([RW, RH], Image.Resampling.BILINEAR)
        # 最终只返回Image信息
        _img = self.img_tran(raw_img)
        return _img



class LightNuscenes(NuScenes):
    def __init__(self, version = 'v1.0-test', dataroot = 'data/nuscenes', verbose = True, map_resolution = 0.1):
        """
        Nuscenes Class Lint
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'log', 'scene', 'sample', 'sample_data','map']

        assert os.path.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        for name in self.table_names:
            setattr(self, name, self.__load_table__(name))  # remove unnecessary keys: ego_pose, sample_annotation, map
        for map_record in self.map:
            map_record['mask'] = MapMask(os.path.join(self.dataroot, map_record['filename']), resolution=map_resolution)

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        # for record in self.sample_annotation:
        #     inst = self.get('instance', record['instance_token'])
        #     record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        # for ann_record in self.sample_annotation:
        #     sample_record = self.get('sample', ann_record['sample_token'])
        #     sample_record['anns'].append(ann_record['token'])

        # Add reverse indices from log records to map records.
        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

class NuSceneDataset(Dataset):
    def __init__(self, version='v1.0-trainval', dataroot='data/nuscenes/',
            scene_names:Optional[Union[str,List[str]]]=None, daylight:bool=True,
            cam_sensor_name:Literal['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']='CAM_FRONT',
            point_sensor_name:str='LIDAR_TOP', skip_point:int=1,
            voxel_size:Optional[float]=None, min_dist=0.15, pcd_sample_num=8192,
            resize_size:Optional[Tuple[int,int]]=None, extend_ratio:Optional[Tuple[float,float]]=None) -> None:
        self.nusc = LightNuscenes(version=version, dataroot=dataroot, verbose=True)
        self.cam_sensor_name = cam_sensor_name
        self.point_sensor_name = point_sensor_name
        if isinstance(scene_names, str):
            if os.path.exists(scene_names):
                scene_names = np.loadtxt(scene_names, dtype=str).tolist()
            else:
                scene_names = [scene_names]
        if scene_names is not None:
            scene_names.sort()
            self.select_scene = [scene for scene in self.nusc.scene if scene['name'] in scene_names]
        else:
            self.select_scene = [scene for scene in self.nusc.scene if (not daylight) or (re.search('night',scene['description'].lower()) is None)]
            self.select_scene.sort(key=lambda t: t['name'])
        self.scene_num_list = []
        self.scene_name_list = []
        for scene in self.select_scene:
            self.scene_name_list.append(scene['name'])
            self.scene_num_list.append(scene['nbr_samples'])
        self.sumsep = np.cumsum(self.scene_num_list)
        self.sample_tokens_by_scene = []
        for scene, nbr_sample in zip(self.select_scene, self.scene_num_list):
            sample_tokens = []
            first_sample_token = scene['first_sample_token']
            sample_tokens.append(first_sample_token)
            sample_token  = first_sample_token
            for _ in range(nbr_sample - 1):
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']
                sample_tokens.append(sample_token)
            self.sample_tokens_by_scene.append(sample_tokens)
        self.img_tran = Tf.Compose([
             Tf.ToTensor(),
             Tf.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.pcd_filter = KITTIFilter(voxel_size, min_dist, skip_point)
        self.tensor_tran = lambda x:torch.from_numpy(x).to(torch.float32)
        self.resample_tran = Resampler(pcd_sample_num)
        self.resize_size = resize_size
        self.extend_ratio = extend_ratio
        
    def __len__(self):
        return self.sumsep[-1]
    
    def get_seq_params(self) -> Tuple[int, List[int]]:
        """Get num_seqs, num_data_per_seq

        Returns:
            Tuple[int, List[int]]: num_seqs, num_data_per_seq
        """
        return len(self.scene_num_list), self.scene_num_list
    
    def __getitem__(self, index:Union[int, Tuple[int,int]]) -> BaseDatasetOutput:
        if isinstance(index, Tuple):
            return self.group_sub_item(*index)
        group_id = np.digitize(index,self.sumsep,right=False).item()
        if group_id > 0:
            sub_idx = index - self.sumsep[group_id-1]
        else:
            sub_idx = index
        return self.group_sub_item(group_id, sub_idx)

    def group_sub_item(self, group_idx:int, sub_idx:int) -> BaseDatasetOutput:
        token = self.sample_tokens_by_scene[group_idx][sub_idx]
        sample = self.nusc.get('sample', token)
        img, pcd, extran, intran = self.get_data(sample, self.cam_sensor_name, self.point_sensor_name)
        camera_info = {
            "fx": intran[0,0].item(),
            "fy": intran[1,1].item(),
            "cx": intran[0,2].item(),
            "cy": intran[1,2].item(),
            "sensor_h": img.height,
            "sensor_w": img.width,
            "projection_mode": "perspective"
        }
        _img = self.img_tran(img)
        _pcd = self.tensor_tran(pcd)  # (N, 3)
        extran = self.tensor_tran(extran)
        return dict(img=_img,pcd=_pcd, camera_info=camera_info, extran=extran, group_idx=group_idx, sub_idx=sub_idx)
    
    def split_dataset(self) -> Generator[Tuple[Dataset, str], None, None]:
        for group_idx, (scene_num, scene_name) in enumerate(zip(self.scene_num_list, self.scene_name_list)):
            yield NusceneDatasetSeqWrapper(self, group_idx, scene_num), scene_name
    
    def get_data(self, sample_record:Dict,
            camera_channel:Literal['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT'],
            pointsensor_channel:Literal['LIDAR_TOP']) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        pointsensor_token = sample_record['data'][pointsensor_channel]
        camera_token = sample_record['data'][camera_channel]
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        img_path = os.path.join(self.nusc.dataroot, cam['filename'])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor['filename'])
        img:Image.Image = Image.open(img_path)
        pc = LidarPointCloud.from_file(pcl_path)
        pcd = np.copy(pc.points).transpose(1,0)[:,:3]  # (N, 3)
        lidar_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        cam_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pose_lidar = np.eye(4)
        pose_lidar[:3,:3] = Quaternion(lidar_record['rotation']).rotation_matrix
        pose_lidar[:3,3] = np.array(lidar_record['translation'])
        pose_rgb = np.eye(4)
        pose_rgb[:3,:3] = Quaternion(cam_record['rotation']).rotation_matrix
        pose_rgb[:3,3] = np.array(cam_record['translation'])
        extran = inv_pose_np(pose_rgb) @ pose_lidar
        intran =  np.array(cam_record['camera_intrinsic'])
        H, W = img.height, img.width
        if self.resize_size is not None:
            RH, RW = self.resize_size[0], self.resize_size[1]
        else:
            RH, RW = H, W
        kx, ky = RW / W, RH / H
        intran[0,:] *= kx
        intran[1,:] *= ky
        if self.extend_ratio is not None:
            REVH, REVW = self.extend_ratio[0] * RH, self.extend_ratio[1] * RW
            K_cam_extend = intran.copy()  # K_cam_extend for dilated projection
            K_cam_extend[0,-1] *= self.extend_ratio[0]
            K_cam_extend[1,-1] *= self.extend_ratio[1]
            calibed_pcd = nptran(pcd, extran)
            *_,rev = transform.binary_projection((REVH,REVW), K_cam_extend, calibed_pcd.T)  # input pcd is (3, N)
            pcd = pcd[rev,:]
        pcd = self.resample_tran(pcd) # (n,3)
        img = img.resize([RW,RH],Image.Resampling.BILINEAR)
        # _img = self.img_tran(img)  # raw img input (3,H,W)
        return img, pcd, extran, intran

    @staticmethod
    def collate_fn(zipped_x:Iterable[BaseDatasetOutput]) -> BatchedBaseDatasetOutput:
        batch = dict()
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['extran'] = torch.stack([x['extran'] for x in zipped_x])
        batch['group_idx'] = [x['group_idx'] for x in zipped_x]
        batch['sub_idx'] = [x['sub_idx'] for x in zipped_x]
        batch['camera_info'] = zipped_x[0]['camera_info']
        batch['camera_info']['fx'] = torch.tensor([x['camera_info']['fx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['fy'] = torch.tensor([x['camera_info']['fy'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cx'] = torch.tensor([x['camera_info']['cx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cy'] = torch.tensor([x['camera_info']['cy'] for x in zipped_x], dtype=torch.float32)
        return batch

class NuSceneIMGDataset(Dataset):
    def __init__(self, version='v1.0-trainval', dataroot='data/nuscenes/',
                 scene_names: Optional[Union[str, List[str]]] = None, daylight: bool = True,
                 cam_sensor_name: Literal['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'] = 'CAM_FRONT',
                 resize_size: Optional[Tuple[int, int]] = None, **argv) -> None:
        self.nusc = LightNuscenes(version=version, dataroot=dataroot, verbose=True)
        self.cam_sensor_name = cam_sensor_name
        self.resize_size = resize_size

        if isinstance(scene_names, str):
            if os.path.exists(scene_names):
                scene_names = np.loadtxt(scene_names, dtype=str).tolist()
            else:
                scene_names = [scene_names]

        if scene_names is not None:
            scene_names.sort()
            self.select_scene = [scene for scene in self.nusc.scene if scene['name'] in scene_names]
        else:
            self.select_scene = [scene for scene in self.nusc.scene if (not daylight) or 
                                 (re.search('night', scene['description'].lower()) is None)]
            self.select_scene.sort(key=lambda t: t['name'])

        self.scene_num_list = []
        self.scene_name_list = []
        for scene in self.select_scene:
            self.scene_name_list.append(scene['name'])
            self.scene_num_list.append(scene['nbr_samples'])
        self.sumsep = np.cumsum(self.scene_num_list)

        self.sample_tokens_by_scene = []
        for scene, nbr_sample in zip(self.select_scene, self.scene_num_list):
            sample_tokens = []
            first_sample_token = scene['first_sample_token']
            sample_tokens.append(first_sample_token)
            sample_token = first_sample_token
            for _ in range(nbr_sample - 1):
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']
                sample_tokens.append(sample_token)
            self.sample_tokens_by_scene.append(sample_tokens)
        self.img_tran = lambda img: torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2,0,1)

    def __len__(self):
        return self.sumsep[-1]

    def get_seq_params(self) -> Tuple[int, List[int]]:
        return len(self.scene_num_list), self.scene_num_list

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        if isinstance(index, Tuple):
            return self.group_sub_item(*index)
        group_id = np.digitize(index, self.sumsep, right=False).item()
        sub_idx = index - self.sumsep[group_id - 1] if group_id > 0 else index
        return self.group_sub_item(group_id, sub_idx)

    def group_sub_item(self, group_idx: int, sub_idx: int):
        token = self.sample_tokens_by_scene[group_idx][sub_idx]
        sample = self.nusc.get('sample', token)
        img = self.get_data(sample, self.cam_sensor_name)
        if self.resize_size is not None:
            RW, RH = self.resize_size
            img = img.resize([RW, RH], Image.Resampling.BILINEAR)
        _img = self.img_tran(img)
        return _img

    def get_data(self, sample_record: Dict,
                 camera_channel: Literal['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                                         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']) -> Image.Image:
        camera_token = sample_record['data'][camera_channel]
        cam = self.nusc.get('sample_data', camera_token)
        img_path = os.path.join(self.nusc.dataroot, cam['filename'])
        img: Image.Image = Image.open(img_path)
        return img


class NusceneDatasetSeqWrapper(Dataset):
    def __init__(self, root_dataset:NuSceneDataset, group_idx:int, scene_num:int):
        self.root_dataset = root_dataset
        self.group_idx = group_idx
        self.scene_num = scene_num

    def __len__(self):
        return self.scene_num
    
    def __getitem__(self, index:int) -> BaseDatasetOutput:
        return self.root_dataset.group_sub_item(self.group_idx, index)
    
    def collate_fn(self, *args, **argv) -> BatchedBaseDatasetOutput:
        return self.root_dataset.collate_fn(*args, **argv)


__classdict__ = {'kitti': BaseKITTIDataset, 'nuscenes': NuSceneDataset}
DATASET_TYPE = TypeVar('DATASET_TYPE', BaseKITTIDataset, NuSceneDataset)
       
class PerturbDataset(Dataset):
    def __init__(self,dataset:DATASET_TYPE,
                 max_deg:float,
                 max_tran:float,
                 min_deg:float=0.0,
                 min_tran:float=0.0,
                 mag_randomly = True,
                 disentangle = False,
                 file: Optional[Union[str, List[str]]] = None):
        self.dataset = dataset
        self.disentangle = disentangle
        self.file = file
        if self.file is not None:
            if isinstance(self.file, list):  # list of perturbation files
                if not all(isinstance(x, str) for x in self.file):
                    raise TypeError("all elements of `file` must be str")
                if not all(os.path.isfile(x) for x in self.file):
                    raise FileNotFoundError("If `file` is a list, all files must exist.")
                self.perturb = torch.from_numpy(np.loadtxt(self.file[0], dtype=np.float32))
                for file in self.file[1:]:
                    self.perturb = torch.cat([self.perturb, torch.from_numpy(np.loadtxt(file, dtype=np.float32))], dim=0)
                self.perturb.unsqueeze_(0)  # (N, 6) -> (1, N, 6)
            elif os.path.isfile(self.file):  # perturbation file
                self.perturb = torch.from_numpy(np.loadtxt(self.file, dtype=np.float32))[None,...]  # (1,N,6)
            else:  # non-exist perturbation file
                random_transform = transform.UniformTransformSE3(max_deg, max_tran, min_deg, min_tran, mag_randomly)
                perturb = random_transform.generate_transform(len(dataset))
                save_dir = Path(os.path.dirname(self.file))
                if not save_dir.exists():
                    save_dir.mkdir(parents=True)
                np.savetxt(self.file, perturb.cpu().detach().numpy(), fmt='%0.6f')
                self.perturb = perturb.unsqueeze(0)  # (1,N,6)
        else:
            self.transform = transform.UniformTransformSE3(max_deg, max_tran, min_deg, min_tran, mag_randomly)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index:Union[int, Tuple[int,int]]) -> PerturbDatasetOutput:
        data = self.dataset[index]
        if isinstance(index, Tuple):
            group_idx, sub_idx = index
            total_index = self.dataset.sumsep[group_idx] + sub_idx
        else:
            total_index = index
        extran = data['extran']  # (4,4)
        if self.file is None:  # randomly generate igt
            igt_x = self.transform.generate_transform(1)
            igt = se3.exp(igt_x).squeeze(0)
        else:
            igt = se3.exp(self.perturb[:,total_index,:]).squeeze(0)  # (1, 6) -> (1, 4, 4) -> (4, 4)
        if self.disentangle:
            init_extran = extran.clone()
            init_extran[:3, :3] = igt[:3, :3] @ extran[:3, :3]
            init_extran[:3, 3] = igt[:3, 3] + extran[:3, 3]
            # disentangle transformation, save seperately
            gt = torch.eye(4)
            gt[:3, :3] = igt[:3, :3].T
            gt[:3, 3] = -igt[:3, 3]
        else:
            init_extran = igt @ extran
            gt = transform.inv_pose(igt)
        new_data = dict(img=data['img'],pcd=data['pcd'], gt_extran=extran, init_extran=init_extran, pose_target=gt, camera_info=data['camera_info'],
                        group_idx=data['group_idx'], sub_idx=data['sub_idx'])
        return new_data
    
    @staticmethod
    def collate_fn(zipped_x:Iterable[PerturbDatasetOutput]) -> BatchedPerturbDatasetOutput:
        batch = dict()
        batch['group_idx'] = [x['group_idx'] for x in zipped_x]
        batch['sub_idx'] = [x['sub_idx'] for x in zipped_x]
        batch['img'] = torch.stack([x['img'] for x in zipped_x])
        batch['pcd'] = torch.stack([x['pcd'] for x in zipped_x])
        batch['init_extran'] = torch.stack([x['init_extran'] for x in zipped_x])
        batch['gt_extran'] = torch.stack([x['gt_extran'] for x in zipped_x])
        batch['pose_target'] = torch.stack([x['pose_target'] for x in zipped_x])
        batch['camera_info'] = zipped_x[0]['camera_info']
        batch['camera_info']['fx'] = torch.tensor([x['camera_info']['fx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['fy'] = torch.tensor([x['camera_info']['fy'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cx'] = torch.tensor([x['camera_info']['cx'] for x in zipped_x], dtype=torch.float32)
        batch['camera_info']['cy'] = torch.tensor([x['camera_info']['cy'] for x in zipped_x], dtype=torch.float32)
        return batch

if __name__ == "__main__":
    base_dataset = BaseKITTIDataset('data/kitti', seqs=['16','17','18'], skip_frame=1)
    dataset = PerturbDataset(base_dataset, 15, 0.15, True)
    print("dataset len:{}".format(len(base_dataset)))
    data = dataset[0]
    for key,value in data.items():
        if hasattr(value, 'shape'):
            shape = value.shape
        else:
            shape = value
        print('{key}: {shape}'.format(key=key, shape=shape))
    base_dataset = NuSceneDataset('v1.0-trainval','data/nuscenes',daylight=False)
    print("dataset len:{}".format(len(base_dataset)))
    data = base_dataset[0]
    for key,value in data.items():
        if hasattr(value, 'shape'):
            shape = value.shape
        else:
            shape = value
        print('{key}: {shape}'.format(key=key, shape=shape))