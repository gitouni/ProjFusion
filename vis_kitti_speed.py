import os
import numpy as np
import matplotlib.pyplot as plt
import pykitti
from pathlib import Path
def main():
    # ================= 配置区域 =================
    # 请修改为您本地 KITTI 数据集的根目录
    # 目录结构应包含 sequences/13/times.txt 等
    basedir = 'data/kitti' 
    res_dir = Path('fig/speed_analysis/kitti/') # 结果保存目录
    res_dir.mkdir(exist_ok=True, parents=True)
    # 定义需要处理的序列
    sequences = ['13', '14', '15', '16', '18']
    
    # 定义自定义位姿文件的路径字典
    # 请将路径替换为您实际的位姿文件路径
    custom_pose_dict = {
        '13': 'data/kitti/lidar_poses/13.txt',
        '14': 'data/kitti/lidar_poses/14.txt',
        '15': 'data/kitti/lidar_poses/15.txt',
        '16': 'data/kitti/lidar_poses/16.txt',
        '18': 'data/kitti/lidar_poses/18.txt'
    }
    # ===========================================

    # 用于存储每个序列的速度数据列表
    all_speeds = []
    valid_sequences = []

    for seq in sequences:
        print(f"Processing sequence {seq}...")
        
        # 1. 使用 pykitti 加载数据集信息 (主要为了获取 times)
        try:
            # pykitti.odometry 会自动寻找 basedir/sequences/{seq} 下的文件
            data_obj = pykitti.odometry(basedir, seq)
        except Exception as e:
            print(f"Error loading pykitti for seq {seq}: {e}")
            continue

        # 2. 加载自定义位姿 (您提供的代码片段)
        lidar_poses = []
        pose_file = custom_pose_dict.get(seq)
        
        if pose_file and os.path.exists(pose_file):
            print(f"Loading custom LiDAR poses for seq {seq} from {pose_file}")
            try:
                poses_velo = []
                with open(pose_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        values = [float(v) for v in line.strip().split()]
                        tf = np.eye(4)
                        # KITTI 格式通常是 12 个数，重塑为 3x4
                        tf[:3, :] = np.array(values).reshape(3, 4)
                        poses_velo.append(tf)
                
                # 检查位姿数量与 pykitti 加载的帧数是否一致
                if len(poses_velo) == len(data_obj):
                    lidar_poses = poses_velo # velodyne 坐标系下的 Pose 列表
                else:
                    print(f"Warning: Custom pose length ({len(poses_velo)}) mismatch for seq {seq} (expected {len(data_obj)}).")
                    # 如果长度不一致，通常无法一一对应时间戳计算，选择跳过或截断
                    # 这里选择跳过以保证数据严谨性
                    continue
            except Exception as e:
                print(f"Error reading pose file for seq {seq}: {e}")
                continue
        else:
            print(f"Pose file not found for seq {seq}")
            continue

        # 3. 计算速度
        # 速度 v = distance / time_diff
        # distance = norm(pos_i+1 - pos_i)
        
        seq_speeds = []
        times = data_obj.timestamps # 获取时间戳列表 (单位: 秒)
        
        for i in range(len(lidar_poses) - 1):
            # 获取当前帧和下一帧的平移向量 (位置)
            # tf[:3, 3] 是 x, y, z 坐标
            p1 = lidar_poses[i][:3, 3]
            p2 = lidar_poses[i+1][:3, 3]
            
            t1 = times[i]
            t2 = times[i+1]
            dt_obj = t2 - t1          # 两个 timedelta 相减
            dt = dt_obj.total_seconds() # 转换为浮点数秒 (e.g., 0.1032)
                        
            # 计算欧氏距离
            dist = np.linalg.norm(p2 - p1)
            
            # 避免除以零 (虽然 KITTI 数据通常 dt ~ 0.1s)
            if dt > 1e-6:
                v = dist / dt # 单位: m/s
                # 简单的异常值过滤 (可选): 比如过滤掉 > 100m/s 的异常点
                # if v < 100: 
                seq_speeds.append(v)
        
        # 将该序列的速度数据加入总列表
        all_speeds.append(seq_speeds)
        valid_sequences.append(seq)
        
        # 打印一些统计信息
        if seq_speeds:
            print(f"  Seq {seq}: Mean Speed = {np.mean(seq_speeds):.2f} m/s, Max = {np.max(seq_speeds):.2f} m/s")

    # 4. 绘制箱线图
    if all_speeds:
        for speed_list, seq in zip(all_speeds, valid_sequences):
            np.savetxt(res_dir / f"speeds_seq_{seq}.txt", speed_list)
            lower_6m_per_second = sum(1 for s in speed_list if s < 6.0)
            print(f"Sequence {seq}: Speeds < 6 m/s count: {lower_6m_per_second / len(speed_list) * 100: .2f}%")
            lower_10m_per_second = sum(1 for s in speed_list if s < 10.0)
            print(f"Sequence {seq}: Speeds < 10 m/s count: {lower_10m_per_second / len(speed_list) * 100: .2f}%")
        plt.figure(figsize=(10, 6))
        
        # 绘制箱线图
        # patch_artist=True 允许填充颜色
        bplot = plt.boxplot(all_speeds, 
                            labels=valid_sequences, 
                            patch_artist=True,
                            showfliers=True) # showfliers=False 可以隐藏离群点
        
        # 设置颜色 (可选)
        colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'lightgrey']
        for patch, color in zip(bplot['boxes'], colors * 2): # 循环使用颜色
            patch.set_facecolor(color)
            
        plt.title('Vehicle Speed Distribution (Sequences 13, 14, 15, 16, 18)')
        plt.xlabel('Sequence ID')
        plt.ylabel('Speed (m/s)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存或显示
        plt.savefig(res_dir / 'kitti_speed_distribution.pdf', bbox_inches='tight')
        print(f"Plot saved to {res_dir / 'kitti_speed_distribution.pdf'}")
        plt.show()
    else:
        print("No valid data to plot.")

if __name__ == "__main__":
    main()