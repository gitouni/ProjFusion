import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import argparse
from PIL import Image
from pathlib import Path
import os
import open3d as o3d
import pykitti
from scipy.spatial.transform import Rotation as R

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", type=str, default="data/kitti", help="KITTI dataset root")
    parser.add_argument("--sequence", type=str, default="13", help="Odometry sequence ID")
    parser.add_argument("--index", type=int, default=400, help="Frame index")
    parser.add_argument("--res_dir", type=str, default="fig/debug_gt_corr") # 修改输出目录
    parser.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png"], help="Output figure format")
    return parser.parse_args()

def topcd(pcd_arr: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr[:, :3])
    return pcd

def project_velo_to_image(pcd: np.ndarray, extran: np.ndarray, intran: np.ndarray, H: int, W: int, margin_ratio: float = 0.0):
    """
    额外返回 mask，以便后续提取对应的 3D 点
    """
    pcd_tf = extran[:3, :3] @ pcd.T + extran[:3, 3:4]
    proj_2d = intran @ pcd_tf
    u, v, w = proj_2d[0, :], proj_2d[1, :], proj_2d[2, :]
    u = u / w
    v = v / w
    u_lb = -margin_ratio * W
    u_ub = (1 + margin_ratio) * W
    v_lb = -margin_ratio * H
    v_ub = (1 + margin_ratio) * H
    
    # 生成掩码
    mask = (w > 0) & (u >= u_lb) & (u < u_ub) & (v >= v_lb) & (v < v_ub)
    
    # 返回 u, v, depth 以及 mask
    return u[mask], v[mask], w[mask], mask

def viz_color_proj(u: np.ndarray, v: np.ndarray, is_gt_inlier: np.ndarray, img: np.ndarray, save_name: str, margin_ratio: float = 0.0):
    """
    可视化投影结果：
    在GT外参下落在图像内部的点 -> 黑色
    其余点 -> 红色
    """
    H, W = img.shape[:2]
    plt.figure(figsize=(12, 12 * H / W), dpi=100)
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 先绘制红色（被GT剔除的点），再绘制黑色（GT内的点），避免红色噪点覆盖主体
    plt.scatter(u[~is_gt_inlier], v[~is_gt_inlier], c='red', alpha=0.8, s=5, label='Out of GT bounds')
    plt.scatter(u[is_gt_inlier], v[is_gt_inlier], c='black', alpha=0.8, s=5, label='Inside GT bounds')
    
    margin = margin_ratio / 2
    plt.axis([int(-margin * W), int((1 + margin) * W), int((1 + margin) * H), int(-margin * H)])
    
    # --- 样式设置 ---
    ax.xaxis.set_major_locator(ticker.MultipleLocator(W))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(H))
    plt.grid(True, which='major', linestyle='--', linewidth=1.5, color='gray', alpha=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Projection saved to: {save_name}")

if __name__ == "__main__":
    args = options()
    res_dir = Path(args.res_dir)
    res_dir.mkdir(exist_ok=True, parents=True)

    # 1. 数据加载
    dataset = pykitti.odometry(args.kitti_root, args.sequence)
    image_raw = dataset.get_cam2(args.index)
    intran = dataset.calib.K_cam2
    extran_gt = dataset.calib.T_cam2_velo # GT 外参
    
    RAW_H, RAW_W = image_raw.height, image_raw.width
    H, W = 375, 750
    # 调整内参以适应 resize 后的图像
    intran[0, :] *= W / RAW_W
    intran[1, :] *= H / RAW_H
    image_raw = image_raw.resize((W, H), resample=Image.BILINEAR)
    image_np = np.array(image_raw)
    
    pcd_full = dataset.get_velo(args.index)[:, :3] # [N, 3]

    # 保存原始参考图
    image_raw.save(res_dir / "image_ref.png")

    # ==========================================
    # 获取 GT 外参下的投影掩码 (用于区分红黑)
    # ==========================================
    print("\n--- Computing GT Mask ---")
    _, _, _, mask_gt = project_velo_to_image(pcd_full, extran_gt, intran, H, W, margin_ratio=0.0)
    num_mask_gt = np.sum(mask_gt)

    # ==========================================
    # 实验 1: 扰动外参下的投影与可视化 (正常视场)
    # ==========================================
    print("\n--- Processing Perturbed Parameters ---")
    euler_angles = [10, -10, 10] # 旋转扰动 (deg)
    perturb = np.eye(4)
    perturb[:3, :3] = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    perturb[:3, 3] = [0.4, -0.3, 0.45]
    extran_noise = perturb @ extran_gt
    print(f"Perturbation - Rotation (deg): {euler_angles}, Translation (m): {perturb[:3, 3]}")
    
    u_noise, v_noise, d_noise, mask_noise = project_velo_to_image(pcd_full, extran_noise, intran, H, W, margin_ratio=0.0)
    
    # 利用原掩码提取这些点在 GT 条件下是否在视场内
    is_gt_inlier_noise = mask_gt[mask_noise]
    num_mask_perturb = np.logical_and(mask_noise, mask_gt).sum()
    viz_color_proj(u_noise, v_noise, is_gt_inlier_noise, image_np, 
                   res_dir / "proj_perturbed_colored.{fmt}".format(fmt=args.fmt), margin_ratio=0.0)
    print("Proportion of points inside GT bounds under perturbation: {:.2f}%".format(num_mask_perturb / num_mask_gt * 100))
    # ==========================================
    # 实验 2: 扰动外参下的投影与可视化 (扩展视场)
    # ==========================================
    print("\n--- Processing Perturbed Parameters (Expand View) ---")
    margin_ratio = 2.0
    u_m, v_m, d_m, mask_m = project_velo_to_image(pcd_full, extran_noise, intran, H, W, margin_ratio)
    
    # 利用原掩码提取这些点在 GT 条件下是否在视场内
    is_gt_inlier_m = mask_gt[mask_m]
    num_mask_perturb_expanded = np.logical_and(mask_m, mask_gt).sum()
    viz_color_proj(u_m, v_m, is_gt_inlier_m, image_np, 
                   res_dir / "proj_perturbed_expand_colored.{fmt}".format(fmt=args.fmt), margin_ratio=margin_ratio)
    print("Proportion of points inside GT bounds under expanded view: {:.2f}%".format(num_mask_perturb_expanded / num_mask_gt * 100))