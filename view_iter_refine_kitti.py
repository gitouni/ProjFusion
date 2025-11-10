import numpy as np
from matplotlib import pyplot as plt
import argparse
from models.tools.cmsc import toMat, npproj
from pathlib import Path
import os
import open3d as o3d
import shutil
from tqdm import tqdm
from dataset import BaseKITTIDataset
from models.util.constant import IMAGENET_DEFAULT_MEAN as IMAGENET_MEAN
from models.util.constant import IMAGENET_DEFAULT_STD as IMAGENET_STD
from torchvision import transforms as Tf
import cv2


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir",type=str,default="data/kitti")
    parser.add_argument("--seq",type=str,default="13")
    parser.add_argument("--index",type=int,default=12)
    parser.add_argument("--x0_dir",type=str,default="{method}/seq_{seq}/")
    parser.add_argument("--res_dir",type=str,default="fig/kitti/{seq}-{index}/{tag}/")
    parser.add_argument("--method_dirs",type=str,default=[
        'experiments/kitti/projdualfusion_harmonic_lerr_iter3/results/projdualfusion_harmonic_lerr_iter3'
    ])
    parser.add_argument("--tags",type=str,default=['projdualfusion_harmonic_lerr_iter3'])
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

def viz_proj_pcd(proj_pcd:np.ndarray, r:np.ndarray, img:np.ndarray, save_name:str, cmap:str='rainbow_r'):
    H, W = img.shape[:2]
    u, v = proj_pcd[:,0], proj_pcd[:,1]
    plt.figure(figsize=(12,5),dpi=100,tight_layout=True)
    plt.axis([0,W,H,0])
    plt.imshow(img)
    plt.axis([-W,2 * W,2 * H,-H])
    plt.scatter([u],[v],c=[r],cmap=cmap,alpha=0.25,s=1)
    
    plt.xticks([-W, 0, W, 2*W], labels=['-W','0','W','2W'], fontsize=16)
    plt.yticks([-H, 0, H, 2*H], labels=['-H','0','H','2H'], fontsize=16)
    # plt.axis('off')
    plt.grid()
    plt.savefig(save_name, bbox_inches='tight')  # no padding
    plt.close()

def viz_proj_pcd_depthmap(proj_pcd: np.ndarray, r: np.ndarray, img: np.ndarray, save_name: str, cmap:str='gray_r'):
    """
    绘制深度图:
    - 背景为黑色
    - 投影点越亮代表距离越近
    - 输出灰度图，不依赖 plt
    """
    
    H, W = img.shape[:2]
    u, v = proj_pcd[:,0], proj_pcd[:,1]
    plt.figure(figsize=(12,5),dpi=100,tight_layout=True)
    plt.axis([0,W,H,0])
    ax = plt.gca()
    ax.set_facecolor("black") 
    plt.scatter([u],[v],c=[r],cmap=cmap,alpha=1.0,s=1)
    plt.xticks([], [])
    plt.yticks([], [])
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
    dataset = BaseKITTIDataset(basedir=args.basedir, seqs=[args.seq], min_dist=0, pcd_sample_num=-1, extend_ratio=[2.5, 2.5])
    data = dataset[args.index]
    RESTORE_IMAGE_MEAN = list(map(lambda x,y: -x / y, IMAGENET_MEAN, IMAGENET_STD))
    RESTORE_IMAGE_STD = list(map(lambda y: 1 / y, IMAGENET_STD))
    transform = Tf.Compose([
        Tf.Normalize(RESTORE_IMAGE_MEAN, RESTORE_IMAGE_STD),
        Tf.ToPILImage()
    ])
    image:np.ndarray = np.array(transform(data["img"]), dtype=np.uint8)  # (3, H, W) -> (H, W, 3)
    pcd:np.ndarray = data['pcd'].detach().numpy()  # (N, 3)
    gt_mat = data['extran'].detach().numpy()  # (4,4)
    camera_info = data['camera_info']
    fx, fy, cx, cy = camera_info['fx'], camera_info['fy'], camera_info['cx'], camera_info['cy']
    H, W = camera_info['sensor_h'], camera_info['sensor_w']
    intran = np.array([[fx, 0, cx],
                      [0, fy ,cy],
                      [0, 0, 1]])
    for method_dir, tag in zip(args.method_dirs, args.tags):
        x0_dir = args.x0_dir.format(method=method_dir, seq=args.seq)
        res_dir = Path(args.res_dir.format(tag=tag, seq=args.seq, index=args.index))
        if res_dir.exists():
            shutil.rmtree(str(res_dir))
        res_dir.mkdir(parents=True)
        x0_files = sorted(os.listdir(x0_dir))
        img_hw = image.shape[:2]
        x0 = np.loadtxt(os.path.join(x0_dir, x0_files[args.index]))
        if np.ndim(x0) == 1:
            x0 = [x0]
        
        proj_pcd, _, depth = npproj(pcd, gt_mat, intran, img_hw, return_depth=True, boundary=[-W, 3*W, -H, 3*H])
        viz_proj_pcd(proj_pcd, depth, image, str(res_dir.joinpath("gt.png")))
        proj_pcd, _, depth = npproj(pcd, gt_mat, intran, img_hw, return_depth=True)
        viz_proj_pcd_depthmap(proj_pcd, depth, image, str(res_dir.joinpath("depth_gt.png")))
        for t, xt in tqdm(enumerate(x0), total=len(x0)):
            extran = toMat(xt[:3],xt[3:])
            proj_pcd, _, depth = npproj(pcd, extran, intran, img_hw, return_depth=True, boundary=[-W, 3*W, -H, 3*H])
            viz_proj_pcd(proj_pcd, depth, image, str(res_dir.joinpath("%06d.png"%t)))
            proj_pcd, _, depth = npproj(pcd, extran, intran, img_hw, return_depth=True, boundary=[-W, 3*W, -H, 3*H])
            viz_proj_pcd_depthmap(proj_pcd, depth, image, str(res_dir.joinpath("depth_%06d.png"%t)))