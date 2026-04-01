import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from pathlib import Path
def main():
    # ================= 配置区域 =================
    # 请修改为您本地 nuScenes 数据集的根目录
    dataroot = 'data/nuscenes/'
    version = 'v1.0-test'  # 指定使用 test 集
    res_dir = Path('fig/speed_analysis/nusc/') # 结果保存目录
    res_dir.mkdir(exist_ok=True, parents=True)
    # ===========================================

    print(f"Initializing nuScenes {version}...")
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    except Exception as e:
        print(f"Error initializing nuScenes: {e}")
        print("Please ensure you have downloaded the v1.0-test metadata and blobs.")
        return

    all_speeds = []
    
    print(f"Processing {len(nusc.scene)} scenes...")

    # 遍历所有场景 (Scene)
    for scene in nusc.scene:
        # 获取该场景的第一个 sample token
        first_sample_token = scene['first_sample_token']
        
        # 从第一个 sample 开始遍历整个链表
        current_sample_token = first_sample_token
        
        while current_sample_token != '':
            # 获取当前 sample 数据
            current_sample = nusc.get('sample', current_sample_token)
            
            # 获取下一个 sample 的 token
            next_sample_token = current_sample['next']
            
            # 如果没有下一帧，说明到了场景末尾，跳出
            if next_sample_token == '':
                break
            
            # 获取下一个 sample 数据
            next_sample = nusc.get('sample', next_sample_token)
            
            # --- 获取位姿 ---
            # 这里的逻辑是：通过 sample -> LIDAR_TOP -> ego_pose
            # 为什么通过 LIDAR_TOP？因为它是关键传感器，其时间戳通常作为 sample 的基准
            
            # 1. 获取 LIDAR_TOP 的 token
            lidar_token_curr = current_sample['data']['LIDAR_TOP']
            lidar_token_next = next_sample['data']['LIDAR_TOP']
            
            # 2. 获取 sample_data 记录
            sd_curr = nusc.get('sample_data', lidar_token_curr)
            sd_next = nusc.get('sample_data', lidar_token_next)
            
            # 3. 获取 ego_pose 记录
            pose_curr = nusc.get('ego_pose', sd_curr['ego_pose_token'])
            pose_next = nusc.get('ego_pose', sd_next['ego_pose_token'])
            
            # --- 计算距离 ---
            # 提取平移向量 (Translation) [x, y, z]
            t1 = np.array(pose_curr['translation'])
            t2 = np.array(pose_next['translation'])
            
            # 计算 2D 欧氏距离 (忽略高度变化，更符合车辆平面运动逻辑)
            dist = np.linalg.norm(t2[:2] - t1[:2])
            
            # --- 计算时间 ---
            # nuScenes 时间戳单位是微秒 (microseconds)
            time_curr = pose_curr['timestamp']
            time_next = pose_next['timestamp']
            
            dt = (time_next - time_curr) / 1e6 # 转换为秒
            
            # --- 计算速度 ---
            if dt > 1e-3: # 防止除零
                v = dist / dt # m/s
                
                # 简单的异常值过滤 (例如 > 60m/s 即 216km/h 可能为异常定位跳变)
                if v < 60.0:
                    all_speeds.append(v)
            
            # 移动到下一帧
            current_sample_token = next_sample_token
    np.savetxt(res_dir / "speeds.txt", all_speeds)

    # 转换单位 (可选: m/s -> km/h)
    # all_speeds_kmh = [s * 3.6 for s in all_speeds]

    print(f"Total speed samples collected: {len(all_speeds)}")
    
    if all_speeds:
        mean_speed = np.mean(all_speeds)
        max_speed = np.max(all_speeds)
        print(f"Global Mean Speed: {mean_speed:.2f} m/s")
        print(f"Global Max Speed: {max_speed:.2f} m/s")
        lower_5m_per_second = sum(1 for s in all_speeds if s < 5.0)
        print(f"Overall Speeds < 5 m/s count: {lower_5m_per_second / len(all_speeds) * 100: .2f}%")
        lower_8m_per_second = sum(1 for s in all_speeds if s < 8.0)
        print(f"Overall Speeds < 8 m/s count: {lower_8m_per_second / len(all_speeds) * 100: .2f}%")
        lower_10m_per_second = sum(1 for s in all_speeds if s < 10.0)
        print(f"Overall Speeds < 10 m/s count: {lower_10m_per_second / len(all_speeds) * 100: .2f}%")

        # --- 绘图 ---
        plt.figure(figsize=(12, 6))
        
        # 子图1: 直方图 (展示整体分布形状)
        plt.subplot(1, 2, 1)
        plt.hist(all_speeds, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Speed Histogram (nuScenes v1.0-test)')
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 子图2: 箱线图 (展示统计特征)
        plt.subplot(1, 2, 2)
        plt.boxplot(all_speeds, patch_artist=True, 
                    boxprops=dict(facecolor='lightgreen'))
        plt.title('Speed Boxplot')
        plt.ylabel('Speed (m/s)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(res_dir / 'nuscenes_speed_distribution.pdf', bbox_inches='tight')
        print(f"Plot saved to {res_dir / 'nuscenes_speed_distribution.pdf'}")
        plt.show()
    else:
        print("No speed data calculated. Please check dataset path and version.")

if __name__ == "__main__":
    main()