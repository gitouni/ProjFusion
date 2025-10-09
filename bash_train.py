import subprocess

if __name__ == "__main__":
    python_path = '/home/ouni/anaconda3/envs/pytorch/bin/python'
    script_path = 'train.py'
    model_config = 'cfg/model/pool_3c_t.yml'
    dataset_config = 'cfg/dataset/kitti.yml'
    # process = subprocess.Popen([python_path,'train.py',
    #     '--dataset_config',dataset_config,'--model_config',model_config,
    #     '--base_dir', 'pool_3c_t', '--task_name', 'kitti3', '--rl_steps','10'])
    # process.wait()  # must be serialized
    # model_config = 'cfg/model/pool_3c_t_bc.yml'
    # process = subprocess.Popen([python_path, script_path,
    #     '--dataset_config',dataset_config,'--model_config',model_config,
    #     '--base_dir', 'pool_3c_t', '--task_name', 'kitti_bc3', '--rl_steps','10'])
    # process.wait()  # must be serialized
    model_config = 'cfg/model_nuscenes/pool_3c_t.yml'
    dataset_config = 'cfg/dataset/nusc.yml'
    process = subprocess.Popen([python_path,script_path,
        '--dataset_config',dataset_config,'--model_config',model_config,
        '--base_dir', 'pool_3c_t', '--task_name', 'nusc_new', '--rl_steps','10','--resume','experiments/pool_3c_t/nusc_new/checkpoint/last_model.pth'])
    process.wait()  # must be serialized
    model_config = 'cfg/model_nuscenes/pool_3c_t_bc.yml'
    process = subprocess.Popen(['python','train.py',
        '--dataset_config',dataset_config,'--model_config',model_config,
        '--base_dir', 'pool_3c_t', '--task_name', 'nusc_new_bc', '--rl_steps','10'])
    process.wait()  # must be serialized
