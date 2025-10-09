import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse
from typing import Literal, Generator, Tuple, Dict, Union, Iterable
from dataset import BaseKITTIDataset, NuSceneDataset, SeqBatchSampler, BatchedBaseDatasetOutput
from torch.utils.data import DataLoader
import yaml
from logging import Logger
from pathlib import Path
import logging
import torch.nn as nn
from einops import rearrange, repeat
from models.lr_scheduler import get_lr_scheduler, get_optimizer
from core.tools import load_checkpoint, save_checkpoint
from core.logger import LogTracker, fmt_time, print_warning
from models.pointgpt.utils.misc import worker_init_fn
from models.environment.environment import Env
from models.loss import se3_err, se3_reduce, se3_rmse
from models.model import EBMMode, DualEBM
from models.util.constant import PredictMode

def unique_append(s:str, arr:list):
    if not s in arr:
        arr.append(s)

def get_dataloader(dataset_type:Literal['kitti','nuscenes'], train_base_dataset_argv:Dict, val_base_dataset_argv:Dict, 
        train_dataloader_argv:Dict, val_dataloader_argv:Dict) -> Tuple[Generator[BatchedBaseDatasetOutput, None, None], Generator[BatchedBaseDatasetOutput, None, None]]:
    if dataset_type == 'kitti':
        train_base_dataset = BaseKITTIDataset(**train_base_dataset_argv)
        val_base_dataset = BaseKITTIDataset(**val_base_dataset_argv)
    elif dataset_type == 'nuscenes':
        train_base_dataset = NuSceneDataset(**train_base_dataset_argv)
        val_base_dataset = NuSceneDataset(**val_base_dataset_argv)
    else:
        raise NotImplementedError("dataset_type:{} unrecognized.".format(dataset_type))
    if 'batch_sampler' in train_dataloader_argv:
        train_dataloader_argv['batch_sampler'] = SeqBatchSampler(*train_base_dataset.get_seq_params(), **train_dataloader_argv['batch_sampler'])
    if 'batch_sampler' in val_dataloader_argv:
        val_dataloader_argv['batch_sampler'] = SeqBatchSampler(*val_base_dataset.get_seq_params(), **val_dataloader_argv['batch_sampler'])
    if hasattr(train_base_dataset, 'collate_fn'):
        train_dataloader_argv['collate_fn'] = getattr(train_base_dataset, 'collate_fn')
    if hasattr(val_base_dataset, 'collate_fn'):
        val_dataloader_argv['collate_fn'] = getattr(val_base_dataset, 'collate_fn')
    train_dataloader = DataLoader(train_base_dataset, **train_dataloader_argv, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_base_dataset, **val_dataloader_argv, worker_init_fn=worker_init_fn)
    return train_dataloader, val_dataloader

@torch.inference_mode()
def val_epoch(env:Env, model:DualEBM, logger:Logger, loader:Generator[BatchedBaseDatasetOutput, None, None], log_per_iter:int,
        repeat_perturb:int, num_perturbations:int, device:torch.device) -> float:
    model.eval()
    progress = tqdm(loader, total=len(loader))
    log_tracker = LogTracker('rot_err','tsl_err','se3_loss', phase='val')
    num_batches = len(loader)
    for i, data in enumerate(progress):
        img = data['img'].to(device)
        pcd = data['pcd'].to(device)
        gt_extran = data['extran'].to(device)  # (B, 4, 4)
        camera_info = data['camera_info']
        global_best_Tcl = None
        best_logits = None
        query_gt_extran = None
        last_err = None
        curr_extran = env.perturb(gt_extran, 1, concat_input=False).squeeze(1)  # init_extran (B, 4, 4)
        with model.state_emb.model_buffer_manager(img, pcd):  # store and clear buffer automatically
            for _ in range(repeat_perturb):
                query_rot_curr_extran = env.perturb(curr_extran, num_perturbations, concat_input=False, add_rot_perturb=True, add_tsl_perturb=False)  # (B, G, 4, 4)
                _, _, best_rot_Tcl = model(img, pcd, query_rot_curr_extran, camera_info, predict_mode=PredictMode.RotOnly, mode=EBMMode.LOGIT_IDX)  # (B, G), (B,), (B, 4, 4)
                query_curr_extran = env.perturb(best_rot_Tcl, num_perturbations, concat_input=False, add_rot_perturb=False, add_tsl_perturb=True)  # (B, G, 4, 4) 
                logits, best_idx, best_Tcl = model(img, pcd, query_curr_extran, camera_info, predict_mode=PredictMode.TslOnly, mode=EBMMode.LOGIT_IDX)  # (B, G), (B,), (B, 4, 4)
                # with torch.no_grad():
                #     best_Tcl_s1_x = se3.log(best_Tcl_s1).detach().clone()
                # best_Tcl = model.optimize_Tcl(img, pcd, best_Tcl_s1_x, camera_info, num_steps=20)
                best_idx_expand = best_idx.view(-1, 1)  # (B,) -> (B, 1)
                # TODO: cluster N local maximum possible points
                curr_logits = torch.gather(logits, 1, best_idx_expand).squeeze(1)  # (B, G) -> (B,)
                if best_logits is None:
                    best_logits = curr_logits
                    global_best_Tcl = best_Tcl
                else:  # update the best extran throughout a single batch
                    replace_idx = curr_logits > best_logits  # (B,)
                    best_logits[replace_idx] = curr_logits[replace_idx]
                    global_best_Tcl[replace_idx, ...] = best_Tcl[replace_idx, ...]
                # curr_extran = global_best_Tcl.clone()  # iteration from current best Tcl
                B, G = query_curr_extran.shape[:2]
                # compute best possible Tcl
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
                curr_extran = global_best_Tcl.clone()  # (B, 4, 4)
        rot_err, tsl_err = se3_err(global_best_Tcl, gt_extran)  # set the Tcl with the largest logits as the predicted one
        se3_loss = se3_reduce(rot_err, tsl_err)
        log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())
        log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())
        log_tracker.update('se3_loss', se3_loss.mean().item())
        log_tracker.update('best_possible_se3', last_err.mean().item())
        if (i+1) % log_per_iter == 0 or i+1 == len(loader):
            logger.info("\tBatch {}|{} {}".format(i+1, num_batches, log_tracker.result()))
    return log_tracker.avg('se3_loss')

def main(config:Dict, config_path:Union[str, Iterable[str]]):
    run_argv = config['run']
    path_argv = config['path']
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True, parents=True)
    experiment_dir = experiment_dir.joinpath(path_argv['name'])
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    yaml.safe_dump(config, open(str(log_dir.joinpath(os.path.basename(config_path))),'w'))
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = config['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    dataset_argv = config['dataset']['train']  # train and val
    train_dataloader, val_dataloader = get_dataloader(config['dataset']['type'], dataset_argv['dataset']['train']['base'], dataset_argv['dataset']['val']['base'], 
        dataset_argv['dataloader']['args'], dataset_argv['dataloader']['val_args'])
    env = Env(**config['dataset']['env']['train'])
    model = DualEBM(**config['model']).to(device)
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), config['optimizer']['type'], **config['optimizer']['args'])
    clip_grad = config['optimizer']['max_grad']
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
    # logger
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'a'  # do not need write because logs have different names
    file_handler = logging.FileHandler(str(log_dir) + '/train_{}.log'.format(fmt_time()), mode=logger_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Training args')
    logger.info(args)
    
    if path_argv['resume'] is not None:
        start_epoch, best_loss, _ = load_checkpoint(path_argv['resume'], model, optimizer, scheduler)
        logger.info("Loaded checkpoint from {}, Start from Epoch {}".format(path_argv['resume'], start_epoch))
        if path_argv['resume_eval_first']:
            with env.set_mag_attr_temp(**config['dataset']['env']['val']):
                val_epoch(env, model, logger, val_dataloader, run_argv['log_per_iter'], run_argv['val_repeat_perturb'], run_argv['val_num_perturbations'], device)
    else:
        start_epoch = 1
        best_loss = float('inf')
        logger.info("Start from scratch")
    
    for epoch in range(start_epoch, run_argv['n_epoch'] + 1):  # start epoch start from 1
        log_tracker_keys = ['rot_ce', 'rot_err','tsl_ce', 'tsl_err']
        log_tracker = LogTracker(*log_tracker_keys, phase='train')
        # -- train
        model.train()
        progress = tqdm(train_dataloader, total=len(train_dataloader))  # dataloader with tqdm
        for bi, data in enumerate(progress):
            img:torch.Tensor = data['img'].to(device)
            pcd:torch.Tensor = data['pcd'].to(device)
            camera_info = data['camera_info'] # device transfer will automatically run in the forward process
            gt_extran:torch.Tensor = data['extran'].to(device)
            with model.state_emb.model_buffer_manager(img, pcd):  # store and clear buffer automatically
                query_extran = env.perturb(gt_extran, run_argv['train_num_perturbations'], concat_input=True, add_rot_perturb_to_gt=True, add_tsl_perturb_to_gt=True)  # (B, G, 4, 4)
                logits, labels, _, best_Tcl, _ = model(img, pcd, query_extran, camera_info, predict_mode=PredictMode.Both, mode=EBMMode.LOGIT_LABEL_IDX_STATE) # (B, N), (B, ), (B, ), (B, 4, 4)
                (rot_logits, rot_labels, rot_best_Tcl), (tsl_logits, tsl_labels, tsl_best_Tcl) = zip(logits, labels, best_Tcl)
                loss_rot = F.cross_entropy(rot_logits, rot_labels, reduction='mean')
                loss_tsl = F.cross_entropy(tsl_logits, tsl_labels, reduction='mean')
                loss = loss_rot + loss_tsl
                optimizer.zero_grad()
                loss.backward()
                try:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=2, error_if_nonfinite=True)  # avoid gradient explosion
                except Exception as e:
                    logger.warning('Found nan Loss, skip this batch. Raw Exception:{}'.format(e))
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                log_tracker.update('rot_ce', loss_rot.item())
                log_tracker.update('tsl_ce', loss_tsl.item())
            with torch.inference_mode():
                rot_err, _ = se3_err(rot_best_Tcl, gt_extran)  # (B, 3), (B, 3)
                _, tsl_err = se3_err(tsl_best_Tcl, gt_extran)  # (B, 3), (B, 3)
                log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())  #  average through batch
                log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())
            if (bi + 1) % run_argv['log_per_iter'] == 0:
                logger.info("\tBatch {}|{}: {}".format(bi+1, len(progress), log_tracker.result()))
                # logger.debug("\tBatch {}|{}: {}".format(bi+1, len(progress), traj_tracker.result()))
        scheduler.step()
        logger.info("Epoch {}|{}: {}".format(epoch, run_argv['n_epoch'], log_tracker.result()))
        if epoch % run_argv['val_per_epoch'] == 0:
            with env.set_mag_attr_temp(**config['dataset']['env']['val']):
                val_loss = val_epoch(env, model, logger, val_dataloader, run_argv['log_per_iter'], run_argv['val_repeat_perturb'], run_argv['val_num_perturbations'], device)
            if val_loss < best_loss:
                logger.info("Find Best Model at Epoch {} prev | curr best loss: {} | {}".format(epoch, best_loss, val_loss))
                best_loss = val_loss
                save_checkpoint(checkpoints_dir.joinpath('best_model.pth'), epoch, best_loss, model, model.state_dict_keys, optimizer, scheduler)
        save_checkpoint(checkpoints_dir.joinpath('last_model.pth'), epoch, best_loss, model, model.state_dict_keys, optimizer, scheduler)
        # torch.cuda.empty_cache()  # clear cache caused by dynamic memory caused by soid_buffer and buffer

def str2bool(s:str) -> bool:
    if s in ['true', 'false']:
        return s == 'true'
    try:
        s_ = int(s)
        return s_ > 0
    except ValueError:
        print_warning("unrecognized param:{}, return false".format(s))
        return False
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config",type=str,default="cfg/dataset/kitti_debug.yml")
    parser.add_argument("--model_config",type=str,default="cfg/model/ebm_nq_daul_attn.yml")
    parser.add_argument("--base_dir",type=str,default="attn_t")
    parser.add_argument("--task_name",type=str,default="kitti_rot_tsl_debug")
    parser.add_argument("--resume",type=str,default=None)
    parser.add_argument("--tr_num", type=int, default=256)
    parser.add_argument("--val_repeat", type=int, default=2)
    parser.add_argument("--val_sub_repeat", type=int, default=1)
    parser.add_argument("--val_num", type=int, default=256)
    args = parser.parse_args()
    dataset_config:Dict = yaml.load(open(args.dataset_config,'r'), yaml.SafeLoader)
    config:Dict  = yaml.load(open(args.model_config,'r'), yaml.SafeLoader)
    base_config:Dict  = yaml.load(open(config['base'],'r'), yaml.SafeLoader)
    config.update(base_config)
    config.update(dataset_config)
    config['path']['name'] = args.task_name
    config['path']['base_dir'] = config['path']['base_dir'].format(base_dir=args.base_dir)
    config['path']['pretrain'] = config['path']['pretrain'].format(base_dir=args.base_dir, task_name=args.task_name)
    if args.resume:
        config['path']['resume'] = args.resume
    config['run']['train_num_perturbations'] = args.tr_num
    config['run']['val_repeat_perturb'] = args.val_repeat
    config['run']['val_repeat_sub_perturb'] = args.val_sub_repeat
    config['run']['val_num_perturbations'] = args.val_num
    PREDICT_MODE = PredictMode.Both
    # args.rot_only == True and args.tsl_only == True, there will be two gt_mat in one sampling (B, 2, 4, 4) + (B, G-2, 4, 4), rotation gt first
    main(config, args.model_config)
