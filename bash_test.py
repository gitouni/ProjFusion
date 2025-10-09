import subprocess

if __name__ == "__main__":
    configs = ['experiments/pool_3c_t/kitti_new/log/pool_3c_t.yml']
    python_path = '/home/ouni/anaconda3/envs/pytorch/bin/python'
    for config in configs:
        process = subprocess.Popen([python_path,'test.py','--config',config])
        process.wait()  # must be serialized