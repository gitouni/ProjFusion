import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import argparse
from PIL import Image
from pathlib import Path
import os
import sys
import open3d as o3d
from functools import partial
import pykitti
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree  # 引入 KDTree

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", type=str, default="data/kitti", help="KITTI dataset root")
    parser.add_argument("--sequence", type=str, default="13", help="Odometry sequence ID")
    parser.add_argument("--index", type=int, default=400, help="Frame index")
    parser.add_argument("--res_dir", type=str, default="fig/debug_nn") # 修改输出目录以便区分
    return parser.parse_args()

def topcd(pcd_arr: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr[:, :3])
    return pcd

def project_velo_to_image(pcd: np.ndarray, extran: np.ndarray, intran: np.ndarray, H: int, W: int, margin_ratio: float = 0.0):
    """
    修改：额外返回 mask，以便后续提取对应的 3D 点
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
    
    print("Mask sum:", np.sum(mask), "/", pcd.shape[0])
    # 返回 u, v, depth 以及 mask
    return u[mask], v[mask], w[mask], mask

def calc_nn_consistency(pcd_3d_subset: np.ndarray, u: np.ndarray, v: np.ndarray):
    """
    计算 3D 最近邻与 2D 投影最近邻的一致性
    pcd_3d_subset: [M, 3] 经过 mask 筛选后的 3D 点
    u, v: [M] 对应的 2D 坐标
    """
    if pcd_3d_subset.shape[0] < 2:
        return np.zeros(pcd_3d_subset.shape[0], dtype=bool), 0.0

    # 1. 构建 2D 坐标数组 [M, 2]
    pts_2d = np.vstack((u, v)).T
    
    # 2. 构建 KDTree
    tree_3d = cKDTree(pcd_3d_subset)
    tree_2d = cKDTree(pts_2d)
    
    # 3. 查询最近邻 (k=2, 因为第1个最近邻是点自己)
    # dists, indices: [M, 2]
    _, idx_3d = tree_3d.query(pcd_3d_subset, k=2)
    _, idx_2d = tree_2d.query(pts_2d, k=2)
    
    # 4. 获取第2个点的索引 (即真正的最近邻)
    nn_3d = idx_3d[:, 1]
    nn_2d = idx_2d[:, 1]
    
    # 5. 比较索引是否相同
    # mismatch 为 True 表示：在3D里的最近邻，投影到2D后不再是最近邻了
    mismatch_mask = (nn_3d != nn_2d)
    
    percentage = np.sum(mismatch_mask) / len(mismatch_mask) * 100.0
    return mismatch_mask, percentage

def viz_nn_changes(u: np.ndarray, v: np.ndarray, mismatch_mask: np.ndarray, img: np.ndarray, save_name: str, margin_ratio: float = 0.0):
    """
    可视化最近邻变化图
    一致的点(False) -> 灰色/淡色
    不一致的点(True) -> 红色/醒目色
    """
    H, W = img.shape[:2]
    plt.figure(figsize=(12, 12 * H / W), dpi=100)
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 绘制一致的点 (背景噪音，设为黑色)
    plt.scatter(u[~mismatch_mask], v[~mismatch_mask], c='black', alpha=0.5, s=5, label='Preserved')
    
    # 绘制不一致的点 (突变点，设为红色或彩虹色)
    # 这里使用红色突出显示
    plt.scatter(u[mismatch_mask], v[mismatch_mask], c='red', alpha=0.8, s=5, label='Changed')
    
    margin = margin_ratio / 2
    plt.axis([int(-margin * W), int((1 + margin) * W), int((1 + margin) * H), int(-margin * H)])
    
    # --- 样式设置 (保留你的网格逻辑) ---
    ax.xaxis.set_major_locator(ticker.MultipleLocator(W))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(H))
    plt.grid(True, which='major', linestyle='--', linewidth=1.5, color='gray', alpha=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 添加图例和标题显示百分比
    # plt.legend(loc='upper right')
    
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"NN Change Map saved to: {save_name}")

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
    # 实验 1: GT 外参下的投影与最近邻变化分析
    # ==========================================
    print("\n--- Processing GT Parameters ---")
    u_gt, v_gt, d_gt, mask_gt = project_velo_to_image(pcd_full, extran_gt, intran, H, W)
    
    # 提取只有在视场内的 3D 点，确保索引对齐
    pcd_subset_gt = pcd_full[mask_gt] 
    
    # 计算最近邻不匹配
    mismatch_gt, ratio_gt = calc_nn_consistency(pcd_subset_gt, u_gt, v_gt)
    print(f"GT Condition - NN Mismatch Ratio: {ratio_gt:.2f}%")
    
    # 可视化
    viz_proj_pcd(u_gt, v_gt, d_gt, image_np, res_dir / "proj_gt.png")
    viz_nn_changes(u_gt, v_gt, mismatch_gt, image_np, res_dir / "nn_change_gt.png")

    # ==========================================
    # 实验 2: 扰动外参下的投影与最近邻变化分析
    # ==========================================
    print("\n--- Processing Perturbed Parameters ---")
    perturb = np.eye(4)
    perturb[:3, :3] = R.from_rotvec([-0.05, 0.05, -0.2]).as_matrix()
    extran_noise = perturb @ extran_gt
    
    u_noise, v_noise, d_noise, mask_noise = project_velo_to_image(pcd_full, extran_noise, intran, H, W)
    
    pcd_subset_noise = pcd_full[mask_noise]
    
    mismatch_noise, ratio_noise = calc_nn_consistency(pcd_subset_noise, u_noise, v_noise)
    print(f"Perturbed Condition - NN Mismatch Ratio: {ratio_noise:.2f}%")
    
    viz_proj_pcd(u_noise, v_noise, d_noise, image_np, res_dir / "proj_perturbed.png")
    viz_nn_changes(u_noise, v_noise, mismatch_noise, image_np, res_dir / "nn_change_perturbed.png")
    
    # 扩展视场测试 (Margin Ratio)
    margin_ratio = 2.0
    u_m, v_m, d_m, mask_m = project_velo_to_image(pcd_full, extran_noise, intran, H, W, margin_ratio)
    pcd_sub_m = pcd_full[mask_m]
    mismatch_m, ratio_m = calc_nn_consistency(pcd_sub_m, u_m, v_m)
    print(f"Perturbed (Wide View) - NN Mismatch Ratio: {ratio_m:.2f}%")
    viz_nn_changes(u_m, v_m, mismatch_m, image_np, res_dir / "nn_change_perturbed_expand.png", margin_ratio=margin_ratio)

    # sys.exit(0)