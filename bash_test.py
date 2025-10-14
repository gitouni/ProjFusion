import subprocess

if __name__ == "__main__":
    configs = ['experiments/nusc/projdualfusion/log/projdualfusion.yml',
               'experiments/nusc/projdualfusion_harmonic/log/projdualfusion_harmonic.yml',
               'experiments/nusc/projdualfusion_harmonic_resnet/log/projdualfusion_harmonic_resnet.yml',
               'experiments/nusc/projfusion/log/projfusion.yml',
               'experiments/nusc/projfusion_harmonic/log/projfusion_harmonic.yml']
    python_path = '/home/bit/anaconda3/envs/pytorch/bin/python'
    for config in configs:
        process = subprocess.Popen([python_path,'test.py','--config',config])
        process.wait()  # must be serialized