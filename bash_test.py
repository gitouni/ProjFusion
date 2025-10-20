import subprocess

if __name__ == "__main__":
    configs = ['experiments/kitti/projdualfusion_harmonic_attn/log/projdualfusion_harmonic_attn.yml',
               'experiments/nusc/projdualfusion_harmonic_attn/log/projdualfusion_harmonic_attn.yml']
    python_path = '/home/bit/anaconda3/envs/pytorch/bin/python'
    for config in configs:
        process = subprocess.Popen([python_path,'test.py','--config', config])
        process.wait()  # must be serialized