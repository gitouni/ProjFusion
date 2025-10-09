import numpy as np
from matplotlib import pyplot as plt
import argparse
from models.tools.cmsc import toMat, npproj
from pathlib import Path
import os
import open3d as o3d
import shutil
from tqdm import tqdm
from dataset import NuSceneDataset
from models.util.constant import IMAGENET_DEFAULT_MEAN as IMAGENET_MEAN
from models.util.constant import IMAGENET_DEFAULT_STD as IMAGENET_STD
from torchvision import transforms as Tf
import re
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",type=str,default="v1.0-test")
    parser.add_argument("--dataroot",type=str,default="data/nuscenes")
    parser.add_argument("--scene",type=str,default="scene-1036")
    parser.add_argument("--index",type=int,default=7)
    parser.add_argument("--x0_dir",type=str,default="{method}/{scene}/")
    parser.add_argument("--res_dir",type=str,default="fig/nusc/{scene_idx}-{index}/{tag}/")
    parser.add_argument("--method_dirs",type=str,default=[
        'experiments/calibnet/nusc/results/iterative_1_2025-02-25-21-30-52',
        'experiments/rggnet/nusc/results/iterative_1_2025-02-25-21-36-30',
        'experiments/lccnet/nusc/results/iterative_1_2025-02-25-21-33-35',
        'experiments/lccnet_mr5/nusc/results/mr_5_2025-03-30-15-52-27',
        'experiments/lccraft/nusc/results/iterative_1_2025-02-25-21-39-19',
        'experiments/calib_depth/nusc/results/bc_3_2025-03-31-00-32-07',
        'experiments/pool_3c_t/nusc_bc/results/ppo_10_2025-04-03-08-52-23',
        'experiments/pool_3c_t/nusc/results/ppo_10_2025-04-03-08-46-25'
    ])
    parser.add_argument("--tags",type=str,default=['calibnet','rggnet','lccnet','lccnet_mr5','lccraft','calibdepth','il','rl'])
    return parser.parse_args()

def loadpcd(name:str):
    if name.endswith('.bin'):
        return np.fromfile(name, dtype=np.float32).reshape(-1,4)[:,:3]
    elif name.endswith('.npy'):
        return np.load(name)
    else:
        return NotImplementedError()

def topcd(pcd_arr:np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr)
    return pcd

def viz_proj_pcd(proj_pcd:np.ndarray, r:np.ndarray, img:np.ndarray, save_name:str):
    H, W = img.shape[:2]
    u, v = proj_pcd[:,0], proj_pcd[:,1]
    plt.figure(figsize=(12,5),dpi=100,tight_layout=True)
    plt.axis([0,W,H,0])
    plt.imshow(img)
    plt.scatter([u],[v],c=[r],cmap='rainbow_r',alpha=0.5,s=2)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight')  # no padding
    plt.close()


def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def change_background_to_white(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    return False

def capture_img(vis, path:str):
    image = vis.capture_screen_float_buffer()
    plt.imsave(path,np.array(image))
    return False

if __name__ == "__main__":
    args = options()
    
    dataset = NuSceneDataset(version=args.version, dataroot=args.dataroot, scene_names=[args.scene], min_dist=0, pcd_sample_num=-1, extend_ratio=[2.5,2.5])
    data = dataset[args.index]
    RESTORE_IMAGE_MEAN = list(map(lambda x,y: -x/y, IMAGENET_MEAN, IMAGENET_STD))
    RESTORE_IMAGE_STD = list(map(lambda y:1/y, IMAGENET_STD))
    transform = Tf.Compose([
        Tf.Normalize(RESTORE_IMAGE_MEAN, RESTORE_IMAGE_STD),
        Tf.ToPILImage()
    ])
    image:np.ndarray = np.array(transform(data["img"]), dtype=np.uint8)  # (3, H, W) -> (H, W, 3)
    pcd:np.ndarray = data['pcd'].detach().transpose(0,1).numpy()  # (N, 3)
    gt_mat = data['extran'].detach().numpy()  # (4,4)
    camera_info = data['camera_info']
    fx, fy, cx, cy = camera_info['fx'], camera_info['fy'], camera_info['cx'], camera_info['cy']
    intran = np.array([[fx, 0, cx],
                      [0, fy ,cy],
                      [0, 0, 1]])
    for method_dir, tag in zip(args.method_dirs, args.tags):
        x0_dir = args.x0_dir.format(method=method_dir, scene=args.scene)
        res_dir = Path(args.res_dir.format(tag=tag, scene_idx=re.search("\d+",args.scene).group(), index=args.index))
        if res_dir.exists():
            shutil.rmtree(str(res_dir))
        res_dir.mkdir(parents=True)
        x0_files = sorted(os.listdir(x0_dir))
        img_hw = image.shape[:2]
        x0 = np.loadtxt(os.path.join(x0_dir, x0_files[args.index]))
        if np.ndim(x0) == 1:
            x0 = [x0]
        
        proj_pcd, _, depth = npproj(pcd, gt_mat, intran, img_hw, return_depth=True)
        viz_proj_pcd(proj_pcd, depth, image, str(res_dir.joinpath("gt.png")))
        for t, xt in tqdm(enumerate(x0), total=len(x0)):
            extran = toMat(xt[:3],xt[3:])
            proj_pcd, _, depth = npproj(pcd, extran, intran, img_hw, return_depth=True)
            viz_proj_pcd(proj_pcd, depth, image, str(res_dir.joinpath("%06d.png"%t)))