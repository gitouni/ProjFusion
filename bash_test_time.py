import subprocess

if __name__ == "__main__":
    # configs = [
    #     'experiments/kitti/projdualfusion_harmonic_r10_t0.5/log/projdualfusion_harmonic.yml',
    #     'experiments/kitti/projdualfusion_harmonic_f2_r10_t0.5/log/projdualfusion_harmonic_f2.yml',
    #     'experiments/kitti/projdualfusion_harmonic_f10_r10_t0.5/log/projdualfusion_harmonic_f10.yml',
    #     'experiments/kitti/projfusion_harmonic_r10_t0.5/log/projfusion_harmonic.yml',
    #     'experiments/kitti/projdualfusion_r10_t0.5/log/projdualfusion.yml',
    #     'experiments/kitti/projdualfusion_harmonic_resnet_r10_t0.5/log/projdualfusion_harmonic_resnet.yml']
    configs = [
        'experiments/kitti/projdualfusion_harmonic_m0_mask_r10_t0.5/log/projdualfusion_harmonic.yml',
        'experiments/kitti/projdualfusion_rope_r10_t0.5/log/projdualfusion_rope.yml'
    ]
    python_path = '/home/bit/anaconda3/envs/pytorch/bin/python'
    for config in configs:
        process = subprocess.Popen([python_path,'test.py','--config', config, '--batch_size', '1', '--no_write_to_file', '--run_iter', '3'])
        process.wait()  # must be serialized