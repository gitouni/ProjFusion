import numpy as np
import torch
from torchinfo import summary
from itertools import islice  # to skip batches
from tqdm import tqdm
import argparse
from models.environment.environment import Env
from models.loss import se3_err, se3_reduce, se3_rmse
from models.model import EBM, EBMMode
from models.util import se3
from dataset import PerturbDataset, BatchedPerturbDatasetOutput
from dataset import __classdict__ as DatasetDict, DATASET_TYPE
from torch.utils.data import DataLoader
import yaml
from typing import Dict, Iterable, Generator, Tuple, List
from logging import Logger
from pathlib import Path
from copy import deepcopy
import shutil
import logging
from core.logger import LogTracker, fmt_time
from core.tools import load_checkpoint_model_only
from models.tools.csrc import wrapped_fps
from einops import rearrange, repeat
import math
from accelerate import Accelerator

def get_dataloader(test_dataset_argv:Iterable[Dict],
        test_dataloader_argv:Dict, dataset_type:str) -> Tuple[List[str], List[Generator[BatchedPerturbDatasetOutput, None, None]], List[Env]]:
    name_list = []
    dataloader_list = []
    env_list = []
    data_class:DATASET_TYPE = DatasetDict[dataset_type]
    if isinstance(test_dataset_argv, list):
        for dataset_argv in test_dataset_argv:
            name_list.append(dataset_argv['name'])
            base_dataset = data_class(**dataset_argv['base'])
            dataset = PerturbDataset(base_dataset, **dataset_argv['main'])
            if hasattr(dataset, 'collate_fn'):
                test_dataloader_argv['collate_fn'] = getattr(dataset, 'collate_fn')
            dataloader = DataLoader(dataset, **test_dataloader_argv)
            dataloader_list.append(dataloader)
            env_list.append(Env(**dataset_argv['main']))
    else:
        assert hasattr(data_class, 'split_dataset'), '{} must has the function \"split_dataset\"'.format(data_class.__class__.__name__)
        root_dataset:DATASET_TYPE = data_class(**test_dataset_argv['base'])
        for base_dataset, name in root_dataset.split_dataset():
            main_args = deepcopy(test_dataset_argv['main'])
            if 'file' in main_args:
                main_args['file'] = main_args['file'].format(name=name)
            dataset = PerturbDataset(base_dataset, **main_args)
            if hasattr(dataset, 'collate_fn'):
                test_dataloader_argv['collate_fn'] = getattr(dataset, 'collate_fn')
            dataloader = DataLoader(dataset, **test_dataloader_argv)
            dataloader_list.append(dataloader)
            name_list.append(name)
            env_list.append(Env(**dataset_argv['main']))
    return name_list, dataloader_list, env_list

def to_npy(x0:torch.Tensor) -> np.ndarray:
    return x0.detach().cpu().numpy()

@torch.no_grad()
def test_seq(env:Env, model:EBM, logger:Logger, loader:Generator[BatchedPerturbDatasetOutput, None, None], fps_sample_num:int,
             res_dir:Path, log_per_iter:int, repeat_perturb:int, num_perturbations:int, device:torch.device, cnt:int=0):
    model.eval()
    progress = tqdm(loader, total=len(loader))
    log_tracker = LogTracker('rot_err','tsl_err','se3_loss','best_possible_se3', phase='test')
    num_batches = len(loader)
    if isinstance(res_dir, Path):
        res_dir = res_dir
    else:
        res_dir = Path(res_dir)
    # for _ in range(cnt):
    #     progress.__iter__().__next__()  # skip cnt batches
    # bnd_se3 = np.array([env.max_deg * math.pi / 180, env.max_deg * math.pi / 180, env.max_deg * math.pi / 180, env.max_tran, env.max_tran, env.max_tran])
    for i, data in enumerate(islice(progress, cnt, None)):
        img = data['img'].to(device)
        pcd = data['pcd'].to(device)
        # pcd = wrapped_fps(pcd, fps_sample_num)
        # init_extran = data['init_extran'].to(device)
        gt_extran = data['gt_extran'].to(device)
        init_extran = env.perturb(gt_extran, 1, concat_input=False).squeeze(1)  # init_extran (B, 4, 4)
        # pose_target_x = se3.log(data['pose_target']).cpu().detach().numpy()
        # within_range = np.logical_and(-bnd_se3[None,...] < pose_target_x, pose_target_x < bnd_se3[None,...])
        # print(pose_target_x)
        # print(within_range)
        camera_info = data['camera_info']
        se3_xlist = [to_npy(se3.log(init_extran))]
        curr_extran = init_extran
        with model.state_emb.model_buffer_manager(img, pcd):  # store and clear buffer automatically
            total_best_extran = None
            best_logits = None
            query_gt_extran = None
            last_err = None
            for _ in range(repeat_perturb):
                query_curr_extran = env.perturb(curr_extran, num_perturbations, concat_input=False)  # (B, G, 4, 4)
                logits, best_idx, best_Tcl = model(img, pcd, query_curr_extran, camera_info, mode=EBMMode.LOGIT_IDX)  # (B, G), (B,), (B, 4, 4)
                # with torch.no_grad():
                #     best_Tcl_s1_x = se3.log(best_Tcl_s1).detach().clone()
                # best_Tcl = model.optimize_Tcl(img, pcd, best_Tcl_s1_x, camera_info, num_steps=20)
                best_idx_expand = best_idx.view(-1, 1)  # (B,) -> (B, 1)
                # TODO: cluster N local maximum possible points
                curr_logits = torch.gather(logits, 1, best_idx_expand).squeeze(1)  # (B, G) -> (B,)
                if best_logits is None:
                    best_logits = curr_logits
                    total_best_extran = best_Tcl
                else:  # update the best extran throughout a single batch
                    replace_idx = curr_logits > best_logits  # (B,)
                    best_logits[replace_idx] = curr_logits[replace_idx]
                    total_best_extran[replace_idx, ...] = best_Tcl[replace_idx, ...]
                # curr_extran = total_best_extran.clone()  # iteration from current best Tcl
                B, G = query_curr_extran.shape[:2]
                batch_err = se3_reduce(*se3_err(rearrange(query_curr_extran, 'b g ... -> (b g) ...'), repeat(gt_extran, 'b ... -> (b g) ...', g=G)))
                batch_err = rearrange(batch_err, '(b g) ... -> b g ...',b=B, g=G)  # (B, G)
                best_err_idx = torch.argmin(batch_err, dim=1)  # (B,)
                err = torch.gather(batch_err, 1, best_err_idx.view(-1, 1))  # (B,)
                best_extran = torch.gather(query_curr_extran, 1, best_err_idx.view(-1, 1, 1, 1).expand(-1, -1, 4, 4))  # (B, 4, 4)
                if last_err is None:
                    last_err = err  # (B, )
                    query_gt_extran = best_extran
                else:
                    replace_idx = err < last_err
                    query_gt_extran[replace_idx] = best_extran[replace_idx]
                    last_err[replace_idx] = err[replace_idx]
                curr_extran = total_best_extran.clone()
                se3_xlist.append(to_npy(se3.log(total_best_extran)))
        batched_x0_list = np.stack(se3_xlist, axis=1)  # (B, K, 6)
        for x0 in batched_x0_list:
            np.savetxt(res_dir.joinpath("%06d.txt"%cnt), x0)
            cnt += 1
        rot_err, tsl_err = se3_err(total_best_extran, gt_extran)  # set the Tcl with the largest logits as the predicted one
        se3_loss = se3_reduce(rot_err, tsl_err)
        log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())
        log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())
        log_tracker.update('se3_loss', se3_loss.mean().item())
        log_tracker.update('best_possible_se3', last_err.mean().item())
        if (i+1) % log_per_iter == 0 or i+1 == len(loader):
            logger.info("\tBatch {}|{} {}".format(i+1, num_batches, log_tracker.result()))
    return log_tracker.result()

def main(config:Dict, resume:bool):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    dataset_argv = config['dataset']['test']
    pcd_sample_num = config['dataset']['train']['dataset']['train']['base']['pcd_sample_num']
    dataset_type = config['dataset']['type']
    name_list, dataloader_list, env_list = get_dataloader(dataset_argv['dataset'], dataset_argv['dataloader'], dataset_type)
    model = EBM(**config['model']).to(device)
    # summary(model)
    # exit(0)
    run_argv = config['run']
    path_argv = config['path']
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True, parents=True)
    experiment_dir = experiment_dir.joinpath(path_argv['name'])
    experiment_dir.mkdir(exist_ok=True)
    # checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    # checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    # if isinstance(config_path, str):
    #     shutil.copyfile(config_path, str(log_dir.joinpath(os.path.basename(config_path))))  # copy the config file
    # else:
    #     for path in config_path:
    #         shutil.copyfile(path, str(log_dir.joinpath(os.path.basename(path))))  # copy the config file
    # logger
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'a' if path_argv['resume'] is not None else 'w'
    file_handler = logging.FileHandler(str(log_dir) + '/test_{}_{}.log'.format(path_argv['name'], fmt_time()), mode=logger_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Testing args')
    logger.info(args)
    if path_argv['pretrain'] is not None:
        load_checkpoint_model_only(path_argv['pretrain'], model)
        logger.info("Loaded checkpoint from {}".format(path_argv['pretrain']))
    else:
        raise FileNotFoundError("'pretrain' cannot be set to 'None' during test-time")
    record_list = []
    res_dir = experiment_dir.joinpath(path_argv['results']).joinpath(path_argv['name'])
    res_dir.mkdir(exist_ok=True, parents=True)
    for name, dataloader, env in zip(name_list, dataloader_list, env_list):
        sub_res_dir = res_dir.joinpath(name)
        if not sub_res_dir.exists():
            sub_res_dir.mkdir()
            cnt = 0
        else:
            if resume:
                cnt = len([file for file in sub_res_dir.iterdir() if str(file).endswith('.txt')])
            else:
                shutil.rmtree(str(sub_res_dir))
                sub_res_dir.mkdir()
                cnt = 0   
        record = test_seq(env, model, logger, dataloader, pcd_sample_num, sub_res_dir,
            run_argv['log_per_iter'], run_argv['test_repeat_perturb'], run_argv['test_num_perturbations'], device, cnt)
        logger.info("{}: {}".format(name, record))
        record_list.append([name, record])
    logger.info("Summary:")  # view in the bottom
    for name, record in record_list:
        logger.info("{}: {}".format(name, record))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/pool_t/kitti/log/ebm_nq.yml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config:Dict  = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    main(config, args.resume)
