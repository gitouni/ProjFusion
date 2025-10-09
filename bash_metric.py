import subprocess

if __name__ == "__main__":
    python_path = '/home/ouni/anaconda3/envs/pytorch/bin/python'
    gt_dirs = ['cache/kitti_gt','cache/nuscenes_gt']
    log_dirs = ['log/kitti/lccnet_mr5.json',
                'log/nusc/lccnet_mr5.json']
    pred_dir_roots = ['/home/ouni/CODE/Research/AutoCalib/denoising_calib/experiments/kitti/mr_5/lccnet/results/mr_5_2025-03-30-15-47-03',
                      '/home/ouni/CODE/Research/AutoCalib/denoising_calib/experiments/nuscenes/mr_5/lccnet/results/mr_5_2025-03-30-15-52-27']
    for log_dir, pred_dir_root, gt_dir in zip(log_dirs, pred_dir_roots, gt_dirs):
        process = subprocess.Popen([python_path,'metrics.py', '--gt_dir',gt_dir, '--pred_dir_root',pred_dir_root, '--log_file',log_dir])
        process.wait()  # must be serialized