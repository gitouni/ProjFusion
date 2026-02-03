import numpy as np
from matplotlib import pyplot as plt
import argparse
from PIL import Image
from pathlib import Path
import os
import open3d as o3d
from functools import partial
import pykitti  # 确保已安装: pip install pykitti

def options():
    parser = argparse.ArgumentParser()
    # 修改为 pykitti 需要的路径结构
    parser.add_argument("--kitti_root", type=str, default="data/kitti", help="KITTI dataset root")
    parser.add_argument("--sequence", type=str, default="13", help="Odometry sequence ID")
    parser.add_argument("--index", type=int, default=400, help="Frame index")
    parser.add_argument("--res_dir", type=str, default="fig/debug")
    return parser.parse_args()

def topcd(pcd_arr: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr[:, :3])
    return pcd

def project_velo_to_image(pcd: np.ndarray, extran: np.ndarray, intran: np.ndarray, H: int, W: int):
    """
    将点云投影到图像平面
    pcd: [N, 3] 点云数据 (x, y, z)
    extran: [4, 4] 外参矩阵
    intran: [3, 3] 内参矩阵
    返回:
        proj_2d: [N, 2] 投影后的二维坐标 (u, v)
        depth: [N] 深度值
    """
    pcd_tf = extran[:3, :3] @ pcd.T + extran[:3, 3:4]  # [3, N]
    # 投影到图像平面
    proj_2d = intran @ pcd_tf  # [3, N]
    u, v, w = proj_2d[0, :], proj_2d[1, :], proj_2d[2, :]
    u = u / w
    v = v / w
    mask = (w > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    print("mask sum:", np.sum(mask), "/", pcd.shape[0])
    return u[mask], v[mask], w[mask]


def viz_proj_pcd(u: np.ndarray, v: np.ndarray, depth: np.ndarray, img: np.ndarray, save_name: str):
    H, W = img.shape[:2]
    plt.figure(figsize=(12, 5), dpi=100, tight_layout=True)
    # plt.imshow(img)
    plt.scatter(u, v, c=np.sqrt(depth), cmap='rainbow_r', alpha=0.8, s=10)
    plt.axis([0, W, H, 0])
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Projection saved to: {save_name}")


def depth_proj_pcd(u: np.ndarray, v: np.ndarray, depth: np.ndarray, img: np.ndarray, save_name: str):
    H, W = img.shape[:2]
    
    # 1. 设置画布背景为黑色 (facecolor='black')
    fig = plt.figure(figsize=(12, 5), dpi=100, tight_layout=True, facecolor='black')
    
    # 获取当前坐标轴并设置背景为黑色
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # 2. 修改 scatter 的颜色映射
    # cmap='gray': 数值越小越黑，数值越大越白
    # 如果你希望“近处白，远处黑”，请使用 cmap='gray_r' (反转灰度)
    plt.scatter(u, v, c=np.sqrt(depth), cmap='gray', alpha=0.8, s=10)
    
    # 设置坐标轴范围
    plt.axis([0, W, H, 0])
    
    # 关闭坐标轴显示 (刻度、边框等)
    plt.axis('off')
    
    # 3. 保存图片时确保背景色一致
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, facecolor='black')
    
    plt.close()
    print(f"Depth saved to: {save_name}")

# --- Open3D 回调函数 ---
def capture_all(vis, res_dir, pcd_full, dataset, index, image_np):
    """
    按下 S 键触发：保存当前视角、原始点云、投影图
    """
    # 1. 保存当前 Open3D 视角的渲染图
    screen_path = os.path.join(res_dir, f"viewpoint_{index}.png")
    image = vis.capture_screen_float_buffer()
    plt.imsave(screen_path, np.array(image))

    return False

def increase_point_size(vis):
    opt = vis.get_render_option()
    opt.point_size += 1.0
    return False

def decrease_point_size(vis):
    opt = vis.get_render_option()
    opt.point_size = max(opt.point_size - 1.0, 1.0)
    return False

if __name__ == "__main__":
    args = options()
    res_dir = Path(args.res_dir)
    res_dir.mkdir(exist_ok=True, parents=True)

    # --- 使用 pykitti 加载数据 ---
    # 确保路径结构为: data/kitti/sequences/13/image_2, data/kitti/sequences/13/velodyne 等
    dataset = pykitti.odometry(args.kitti_root, args.sequence)
    
    # 加载特定帧
    image_raw = dataset.get_cam2(args.index) # PIL Image
    image_np = np.array(image_raw) # H, W, 3
    pcd_full = dataset.get_velo(args.index) # [N, 4] (x, y, z, intensity)
    
    # 保存原始参考图
    image_raw.save(res_dir / f"image_{args.index}.png")
    # 2. 生成并保存投影图 (Projection)
    proj_path = os.path.join(res_dir, f"projection_{args.index}.png")
    depth_path = os.path.join(res_dir, f"depth_{args.index}.png")
    intran = dataset.calib.K_cam2  # 内参矩阵
    extran = dataset.calib.T_cam2_velo  # 外参矩阵
    H, W = image_np.shape[:2]
    u, v, depth = project_velo_to_image(pcd_full[:, :3], extran, intran, H, W)
    viz_proj_pcd(u, v, depth, image_np, proj_path)
    depth_proj_pcd(u, v, depth, image_np, depth_path)
    
    print(f"Captured screen and projection for index {args.index}")
    # --- Open3D 可视化准备 ---
    pcd_o3d = topcd(pcd_full[:, :3])
    pcd_o3d, _ = pcd_o3d.remove_statistical_outlier(nb_neighbors=10, std_ratio=5.0)
    # --- 新增：按 Z 轴高度上色 ---
    points = np.asarray(pcd_o3d.points)  # filtered points
    z_values = points[:, 2]  # 获取 Z 坐标
    z_min = np.min(z_values)
    z_max = np.max(z_values)

    # 归一化 Z 值到 [0, 1]
    z_norm = (z_values - z_min) / (z_max - z_min)

    # 使用 matplotlib 的颜色映射（例如 'jet' 或 'viridis'）
    # .colors 属性会返回 (N, 4)，我们只取前 3 列 (RGB)
    colors = plt.get_cmap('jet')(z_norm)[:, :3] 

    # 将颜色赋给点云对象
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    # ---------------------------
    # 绑定按键
    key_to_callback = {
        ord("K"): lambda vis: [setattr(vis.get_render_option(), 'background_color', [0, 0, 0]), False][1],
        ord("B"): lambda vis: [setattr(vis.get_render_option(), 'background_color', [1, 1, 1]), False][1],
        ord("S"): partial(capture_all, res_dir=args.res_dir, pcd_full=pcd_full[:, :3], dataset=dataset, index=args.index, image_np=image_np),
        ord("+"): increase_point_size,
        ord("-"): decrease_point_size,
    }

    print("Controls: 'S' to save all, '+'/-' to resize points, 'K'/'B' for BG color")
    o3d.visualization.draw_geometries_with_key_callbacks([pcd_o3d], key_to_callback)