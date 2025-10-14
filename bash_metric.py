import subprocess

if __name__ == "__main__":
    python_path = '/home/bit/anaconda3/envs/pytorch/bin/python'
    gt_dir = 'cache/kitti_gt'
    log_dirs = ['log/kitti/projdualfusion.json',
                'log/kitti/projdualfusion_harmonic.json',
                'log/kitti/projdualfusion_harmonic_resnet.json',
                'log/kitti/projfusion.json',
                'log/kitti/projfusion_harmonic.json']
    pred_dir_roots = ['experiments/kitti/projdualfusion/results/projdualfusion',
                      'experiments/kitti/projdualfusion_harmonic/results/projdualfusion_harmonic',
                      'experiments/kitti/projdualfusion_harmonic_resnet/results/projdualfusion_harmonic_resnet',
                      'experiments/kitti/projfusion/results/projfusion',
                      'experiments/kitti/projfusion_harmonic/results/projfusion_harmonic']
    for log_dir, pred_dir_root in zip(log_dirs, pred_dir_roots):
        process = subprocess.Popen([python_path,'metrics.py', '--gt_dir',gt_dir, '--pred_dir_root',pred_dir_root, '--log_file',log_dir])
        process.wait()  # must be serialized

    gt_dir = 'cache/nuscenes_gt'
    log_dirs = ['log/nusc/projdualfusion.json',
                'log/nusc/projdualfusion_harmonic.json',
                'log/nusc/projdualfusion_harmonic_resnet.json',
                'log/nusc/projfusion.json',
                'log/nusc/projfusion_harmonic.json']
    pred_dir_roots = ['experiments/nusc/projdualfusion/results/projdualfusion',
                      'experiments/nusc/projdualfusion_harmonic/results/projdualfusion_harmonic',
                      'experiments/nusc/projdualfusion_harmonic_resnet/results/projdualfusion_harmonic_resnet',
                      'experiments/nusc/projfusion/results/projfusion',
                      'experiments/nusc/projfusion_harmonic/results/projfusion_harmonic']
    for log_dir, pred_dir_root in zip(log_dirs, pred_dir_roots):
        process = subprocess.Popen([python_path,'metrics.py', '--gt_dir',gt_dir, '--pred_dir_root',pred_dir_root, '--log_file',log_dir])
        process.wait()  # must be serialized