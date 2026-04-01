import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import argparse
from PIL import Image
from pathlib import Path
import os
import sys
import open3d as o3d
import pykitti
from scipy.spatial.transform import Rotation as R

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", type=str, default="data/kitti", help="KITTI dataset root")
    parser.add_argument("--sequence", type=str, default="13", help="Odometry sequence ID")
    parser.add_argument("--index", type=int, default=400, help="Frame index")
    parser.add_argument("--res_dir", type=str, default="fig/debug_occlusion") # 修改输出目录以便区分
    parser.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png"], help="Output figure format")
    return parser.parse_args()

def topcd(pcd_arr: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr[:, :3])
    return pcd

def project_velo_to_image(pcd: np.ndarray, extran: np.ndarray, intran: np.ndarray, H: int, W: int, margin_ratio: float = 0.0):
    """
    投影点云，并打印被过滤的点数和总点数
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
    
    # 生成掩码 (在相机前方，且在图像/裕量范围内)
    mask = (w > 0) & (u >= u_lb) & (u < u_ub) & (v >= v_lb) & (v < v_ub)
    
    # 打印要求的统计数据
    total_pts = pcd.shape[0]
    kept_pts = np.sum(mask)
    filtered_pts = total_pts - kept_pts
    
    print(f"Total points projected: {total_pts}")
    print(f"Points filtered by image bounds (and w<=0): {filtered_pts}")
    print(f"Points kept in view: {kept_pts}")
    
    # 返回 u, v, depth 以及 mask
    return u[mask], v[mask], w[mask], mask

def calc_occlusion(u: np.ndarray, v: np.ndarray, depth: np.ndarray):
    """
    计算深度图中的遮挡像素
    同一个像素若被赋值多次，深度最小的为未遮挡（False），其余的为被遮挡（True）
    """
    if len(u) == 0:
        return np.array([], dtype=bool), 0.0

    # 1. 转化为像素整数坐标
    u_int = np.floor(u).astype(int)
    v_int = np.floor(v).astype(int)
    coords = np.vstack((u_int, v_int)).T
    
    # 2. 找到所有唯一的像素位置，以及原始坐标对应的唯一像素索引
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
    
    # 3. 找到每个像素位置上的最小深度
    min_depths = np.full(unique_coords.shape[0], np.inf)
    np.minimum.at(min_depths, inverse_indices, depth)
    
    # 4. 判断遮挡：如果该点的深度大于该像素位置的最小深度 (加入1e-4的浮点容差)，则判定为被遮挡
    occluded_mask = depth > (min_depths[inverse_indices] + 1e-4)
    
    # 5. 计算遮挡比率
    percentage = np.sum(occluded_mask) / len(occluded_mask) * 100.0
    
    return occluded_mask, percentage

def viz_occlusion(u: np.ndarray, v: np.ndarray, occluded_mask: np.ndarray, img: np.ndarray, save_name: str, margin_ratio: float = 0.0):
    """
    可视化遮挡分布
    未遮挡点(False) -> 黑色
    被遮挡点(True) -> 红色
    """
    H, W = img.shape[:2]
    plt.figure(figsize=(12, 12 * H / W), dpi=100)
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 绘制未被遮挡的点 (离相机最近的表面)
    plt.scatter(u[~occluded_mask], v[~occluded_mask], c='black', alpha=0.5, s=5, label='Unoccluded')
    
    # 绘制被遮挡的点 (被前面点挡住的背景点)
    plt.scatter(u[occluded_mask], v[occluded_mask], c='red', alpha=0.8, s=5, label='Occluded')
    
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
    
    # plt.legend(loc='upper right')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Occlusion Map saved to: {save_name}")

def viz_proj_pcd(u: np.ndarray, v: np.ndarray, depth: np.ndarray, img: np.ndarray, save_name: str, face_color: str = 'white', cmap: str = "rainbow_r", margin_ratio: float = 0.0):
    H, W = img.shape[:2]
    plt.figure(figsize=(12, 12 * H / W), dpi=100)
    ax = plt.gca()
    ax.set_facecolor(face_color)
    plt.scatter(u, v, c=np.sqrt(depth), cmap=cmap, alpha=0.8, s=10)
    margin = margin_ratio / 2
    plt.axis([int(-margin * W), int((1 + margin) * W), int((1 + margin) * H), int(-margin * H)])
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
    # 实验 1: GT 外参下的投影与深度遮挡分析
    # ==========================================
    print("\n--- Processing GT Parameters ---")
    u_gt, v_gt, d_gt, mask_gt = project_velo_to_image(pcd_full, extran_gt, intran, H, W)
    
    # 计算深度遮挡
    occluded_gt, ratio_gt = calc_occlusion(u_gt, v_gt, d_gt)
    print(f"GT Condition - Occlusion Ratio: {ratio_gt:.2f}%")
    
    # 可视化
    viz_proj_pcd(u_gt, v_gt, d_gt, image_np, res_dir / "proj_gt.{fmt}".format(fmt=args.fmt))
    viz_occlusion(u_gt, v_gt, occluded_gt, image_np, res_dir / "occlusion_gt.{fmt}".format(fmt=args.fmt))

    # ==========================================
    # 扩展视场测试 (Margin Ratio)
    # ==========================================
    print("\n--- Processing GT Parameters (Expand View) ---")
    margin_ratio = 2.0
    u_m, v_m, d_m, mask_m = project_velo_to_image(pcd_full, extran_gt, intran, H, W, margin_ratio)
    occluded_m, ratio_m = calc_occlusion(u_m, v_m, d_m)
    print(f"GT Condition (Wide View) - Occlusion Ratio: {ratio_m:.2f}%")
    
    viz_occlusion(u_m, v_m, occluded_m, image_np, res_dir / "occlusion_perturbed_gt.{fmt}".format(fmt=args.fmt), margin_ratio=margin_ratio)
    # ==========================================
    # 实验 2: 扰动外参下的投影与深度遮挡分析
    # ==========================================
    print("\n--- Processing Perturbed Parameters ---")
    perturb = np.eye(4)
    perturb[:3, :3] = R.from_rotvec([-0.05, 0.05, -0.2]).as_matrix()
    extran_noise = perturb @ extran_gt
    
    u_noise, v_noise, d_noise, mask_noise = project_velo_to_image(pcd_full, extran_noise, intran, H, W)
    
    occluded_noise, ratio_noise = calc_occlusion(u_noise, v_noise, d_noise)
    print(f"Perturbed Condition - Occlusion Ratio: {ratio_noise:.2f}%")
    
    viz_proj_pcd(u_noise, v_noise, d_noise, image_np, res_dir / "proj_perturbed.{fmt}".format(fmt=args.fmt))
    viz_occlusion(u_noise, v_noise, occluded_noise, image_np, res_dir / "occlusion_perturbed.{fmt}".format(fmt=args.fmt))
    
    # ==========================================
    # 扩展视场测试 (Margin Ratio)
    # ==========================================
    print("\n--- Processing Perturbed Parameters (Expand View) ---")
    margin_ratio = 2.0
    u_m, v_m, d_m, mask_m = project_velo_to_image(pcd_full, extran_noise, intran, H, W, margin_ratio)
    
    occluded_m, ratio_m = calc_occlusion(u_m, v_m, d_m)
    print(f"Perturbed (Wide View) - Occlusion Ratio: {ratio_m:.2f}%")
    
    viz_occlusion(u_m, v_m, occluded_m, image_np, res_dir / "occlusion_perturbed_expand.{fmt}".format(fmt=args.fmt), margin_ratio=margin_ratio)