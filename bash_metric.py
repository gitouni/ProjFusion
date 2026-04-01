import subprocess

if __name__ == "__main__":
    python_path = '/home/bit/anaconda3/envs/pytorch/bin/python'
    # gt_dir = 'cache/kitti_gt'
    # log_dirs = ['log/kitti/projdualfusion.json',
    #             'log/kitti/projdualfusion_harmonic.json',
    #             'log/kitti/projdualfusion_harmonic_resnet.json',
    #             'log/kitti/projfusion.json',
    #             'log/kitti/projfusion_harmonic.json']
    # pred_dir_roots = ['experiments/kitti/projdualfusion/results/projdualfusion',
    #                   'experiments/kitti/projdualfusion_harmonic/results/projdualfusion_harmonic',
    #                   'experiments/kitti/projdualfusion_harmonic_resnet/results/projdualfusion_harmonic_resnet',
    #                   'experiments/kitti/projfusion/results/projfusion',
    #                   'experiments/kitti/projfusion_harmonic/results/projfusion_harmonic']
    # for log_dir, pred_dir_root in zip(log_dirs, pred_dir_roots):
    #     process = subprocess.Popen([python_path,'metrics.py', '--gt_dir',gt_dir, '--pred_dir_root',pred_dir_root, '--log_file',log_dir])
    #     process.wait()  # must be serialized

    # gt_dir = 'cache/nuscenes_gt'
    # log_dirs = ['log/nusc/projdualfusion.json',
    #             'log/nusc/projdualfusion_harmonic.json',
    #             'log/nusc/projdualfusion_harmonic_resnet.json',
    #             'log/nusc/projfusion.json',
    #             'log/nusc/projfusion_harmonic.json']
    # pred_dir_roots = ['experiments/nusc/projdualfusion/results/projdualfusion',
    #                   'experiments/nusc/projdualfusion_harmonic/results/projdualfusion_harmonic',
    #                   'experiments/nusc/projdualfusion_harmonic_resnet/results/projdualfusion_harmonic_resnet',
    #                   'experiments/nusc/projfusion/results/projfusion',
    #                   'experiments/nusc/projfusion_harmonic/results/projfusion_harmonic']
    # for log_dir, pred_dir_root in zip(log_dirs, pred_dir_roots):
    #     process = subprocess.Popen([python_path,'metrics.py', '--gt_dir',gt_dir, '--pred_dir_root',pred_dir_root, '--log_file',log_dir])
    #     process.wait()  # must be serialized

    gt_dir = 'cache/kitti_gt'
    log_dirs = ['log/kitti_r15_t0.15/projdualfusion_harmonic_r15_t0.15.json',
                'log/kitti_r10_t0.25/projdualfusion_harmonic_r10_t0.25.json',
                'log/kitti_r10_t0.5/projdualfusion_harmonic_r10_t0.5.json']
    pred_dir_roots = ['experiments/kitti/projdualfusion_harmonic_r15_t0.15/results/projdualfusion_harmonic_r15_t0.15',
                      'experiments/kitti/projdualfusion_harmonic_r10_t0.25/results/projdualfusion_harmonic_r10_t0.25',
                      'experiments/kitti/projdualfusion_harmonic_r10_t0.5/results/projdualfusion_harmonic_r10_t0.5']
    for log_dir, pred_dir_root in zip(log_dirs, pred_dir_roots):
        process = subprocess.Popen([python_path,'metrics.py', '--gt_dir',gt_dir, '--pred_dir_root',pred_dir_root, '--log_file',log_dir])
        process.wait()  # must be serialized

    gt_dir = 'cache/nuscenes_gt'
    log_dirs = ['log/nusc_r15_t0.15/projdualfusion_harmonic_r15_t0.15.json',
                'log/nusc_r10_t0.25/projdualfusion_harmonic_r10_t0.25.json',
                'log/nusc_r10_t0.5/projdualfusion_harmonic_r10_t0.5.json']
    pred_dir_roots = ['experiments/nusc/projdualfusion_harmonic_r15_t0.15/results/projdualfusion_harmonic_r15_t0.15',
                      'experiments/nusc/projdualfusion_harmonic_r10_t0.25/results/projdualfusion_harmonic_r10_t0.25',
                      'experiments/nusc/projdualfusion_harmonic_r10_t0.5/results/projdualfusion_harmonic_r10_t0.5']
    for log_dir, pred_dir_root in zip(log_dirs, pred_dir_roots):
        process = subprocess.Popen([python_path,'metrics.py', '--gt_dir',gt_dir, '--pred_dir_root',pred_dir_root, '--log_file',log_dir])
        process.wait()  # must be serialized

    gt_dir = 'cache/kitti_gt'
    log_dirs = [
        'log/ablation/projdualfusion_harmonic_r10_t0.5.json',
        'log/ablation/projdualfusion_harmonic_f2_r10_t0.5.json',
        'log/ablation/projdualfusion_harmonic_f10_r10_t0.5.json',
        'log/ablation/projdualfusion_harmonic_m0_mask_r10_t0.5.json',
        'log/ablation/projdualfusion_rope_r10_t0.5.json',
        'log/ablation/projfusion_harmonic_r10_t0.5.json',
        'log/ablation/projdualfusion_harmonic_resnet_r10_t0.5.json',
        'log/ablation/projdualfusion_r10_t0.5.json',
        'log/ablation/projdualfusion_concat_r10_t0.5.json',
        'log/ablation/projdualfusion_harmonic_depth_r10_t0.5.json']
    pred_dir_roots = [
        'experiments/kitti/projdualfusion_harmonic_r10_t0.5/results/projdualfusion_harmonic_r10_t0.5',
        'experiments/kitti/projdualfusion_harmonic_f2_r10_t0.5/results/projdualfusion_harmonic_f2_r10_t0.5',
        'experiments/kitti/projdualfusion_harmonic_f10_r10_t0.5/results/projdualfusion_harmonic_f10_r10_t0.5',
        'experiments/kitti/projdualfusion_harmonic_m0_mask_r10_t0.5/results/projdualfusion_harmonic_m0_mask_r10_t0.5',
        'experiments/kitti/projdualfusion_rope_r10_t0.5/results/projdualfusion_rope_r10_t0.5',
        'experiments/kitti/projfusion_harmonic_r10_t0.5/results/projfusion_harmonic_r10_t0.5',
        'experiments/kitti/projdualfusion_harmonic_resnet_r10_t0.5/results/projdualfusion_harmonic_resnet_r10_t0.5',
        'experiments/kitti/projdualfusion_r10_t0.5/results/projdualfusion_r10_t0.5',
        'experiments/kitti/projdualfusion_concat_r10_t0.5/results/projdualfusion_concat_r10_t0.5',
        'experiments/kitti/projdualfusion_harmonic_depth_r10_t0.5/results/projdualfusion_harmonic_depth_r10_t0.5']
    for log_dir, pred_dir_root in zip(log_dirs, pred_dir_roots):
        process = subprocess.Popen([python_path,'metrics.py', '--gt_dir',gt_dir, '--pred_dir_root',pred_dir_root, '--log_file',log_dir])
        process.wait()  # must be serialized
