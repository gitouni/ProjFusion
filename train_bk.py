import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse
from models.environment.environment import Env
from models.loss import se3_err, se3_reduce, se3_rmse
from models.model import EBM, EBMMode
from typing import Literal, Generator, Tuple
from dataset import BaseKITTIDataset, NuSceneDataset, SeqBatchSampler, BatchedBaseDatasetOutput
from torch.utils.data import DataLoader
import yaml
from typing import Dict, Union, Iterable
from logging import Logger
from pathlib import Path
import logging
import torch.nn as nn
from models.lr_scheduler import get_lr_scheduler, get_optimizer
from core.tools import load_checkpoint, save_checkpoint
from core.logger import LogTracker, fmt_time, print_warning
from models.pointgpt.utils.misc import worker_init_fn

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
def val_epoch(env:Env, model:EBM, logger:Logger, loader:Generator[BatchedBaseDatasetOutput, None, None], log_per_iter:int,
        repeat_perturb:int, num_perturbations:int, device:torch.device) -> float:
    model.eval()
    progress = tqdm(loader, total=len(loader))
    log_tracker = LogTracker('rot_err','tsl_err','se3_loss', phase='val')
    num_batches = len(loader)
    for i, data in enumerate(progress):
        img = data['img'].to(device)
        pcd = data['pcd'].to(device)
        gt_extran = data['extran'].to(device)
        camera_info = data['camera_info']
        with model.state_emb.model_buffer_manager(img, pcd):  # store and clear buffer automatically
            query_extran = None
            query_logits = None
            for _ in range(repeat_perturb):
                curr_extran = env.perturb(gt_extran, num_perturbations, concat_input=False)  # (B, G, 4, 4)
                logits, best_idx, best_Tcl = model(img, pcd, curr_extran, camera_info, mode=EBMMode.LOGIT_IDX)  # (B, G), (B,), (B, 4, 4)
                best_idx_expand = best_idx.view(-1, 1)  # (B,) -> (B, 1)
                best_logits = torch.gather(logits, 1, best_idx_expand).squeeze(1)  # (B, G) -> (B,)
                if query_logits is None:
                    query_logits = best_logits
                    query_extran = best_Tcl
                else:  # update the best extran throughout a single batch
                    replace_idx = best_logits > query_logits  # (B,)
                    query_logits[replace_idx] = best_logits[replace_idx]
                    query_extran[replace_idx, ...] = best_Tcl[replace_idx, ...]
        rot_err, tsl_err = se3_err(query_extran, gt_extran)  # set the Tcl with the largest logits as the predicted one
        se3_loss = se3_reduce(rot_err, tsl_err)
        log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())
        log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())
        log_tracker.update('se3_loss', se3_loss.mean().item())
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
    env = Env(**config['dataset']['env'])
    model = EBM(**config['model']).to(device)
    if model.use_queue:
        for key in ['queue','ptr','ema']:
            unique_append(key, model.state_dict_keys)
        
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
            val_epoch(env, model, logger, val_dataloader, run_argv['log_per_iter'], run_argv['val_repeat_perturb'], run_argv['val_num_perturbations'], device)
    else:
        start_epoch = 1
        best_loss = float('inf')
        logger.info("Start from scratch")
    
    for epoch in range(start_epoch, run_argv['n_epoch']+1):  # start epoch start from 1
        log_tracker_keys = ['ce','loss','rot_err','tsl_err','se3']
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
                curr_extran = env.perturb(gt_extran, run_argv['train_num_perturbations'], concat_input=True)  # (B, G, 4, 4)
                logits, labels, _, best_Tcl, state = model(img, pcd, curr_extran, camera_info, mode=EBMMode.LOGIT_LABEL_IDX_STATE) # (B, N), (B, ), (B, ), (B, 4, 4)
                # -- clone term
                loss_ce = F.cross_entropy(logits, labels, reduction='mean')
                # loss_regular = F.l1_loss(logits[:,0], torch.zeros_like(logits[:,0]), reduction='mean')
                loss = loss_ce
                log_tracker.update('ce', loss_ce.item())
                log_tracker.update('loss', loss.item())
                optimizer.zero_grad()
                loss.backward()
                try:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=2, error_if_nonfinite=True)  # avoid gradient explosion
                except Exception as e:
                    logger.warning('Found nan Loss, skip this batch. Raw Exception:{}'.format(e))
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                # ema update and overwrite the source params
                if model.use_queue:
                    model.ema.update()
                    model.ema.apply_shadow()
                D = state.shape[-1]
                if model.use_queue:
                    model._dequeue_and_enqueue(state[:,1:,:].reshape(-1, D).detach())  # must put it after forward because it has been used in the forward
            with torch.inference_mode():
                rot_err, tsl_err = se3_err(best_Tcl, gt_extran)  # (B, 3), (B, 3)
                se3_loss = se3_reduce(rot_err, tsl_err)
                log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())  #  average through batch
                log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())
                log_tracker.update('se3', se3_loss.mean().item())
            if (bi + 1) % run_argv['log_per_iter'] == 0:
                logger.info("\tBatch {}|{}: {}".format(bi+1, len(progress), log_tracker.result()))
                # logger.debug("\tBatch {}|{}: {}".format(bi+1, len(progress), traj_tracker.result()))
        scheduler.step()
        logger.info("Epoch {}|{}: {}".format(epoch, run_argv['n_epoch'], log_tracker.result()))
        if epoch % run_argv['val_per_epoch'] == 0:
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
    parser.add_argument("--dataset_config",type=str,default="cfg/dataset/kitti.yml")
    parser.add_argument("--model_config",type=str,default="cfg/model/ebm_nq.yml")
    parser.add_argument("--base_dir",type=str,default="pool_3c_t")
    parser.add_argument("--task_name",type=str,default="kitti_rot_only")
    parser.add_argument("--resume",type=str,default=None)
    parser.add_argument("--tr_num", type=int, default=256)
    parser.add_argument("--val_repeat", type=int, default=8)
    parser.add_argument("--val_num", type=int, default=512)
    parser.add_argument("--gt_rot_perturb",action='store_true', default=False)
    parser.add_argument("--gt_tsl_perturb",action='store_true', default=False)
    args = parser.parse_args()
    assert not (args.gt_rot_perturb and args.gt_tsl_perturb), "cannot add both rotation and translation perturbations to gt"
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
    config['run']['val_num_perturbations'] = args.val_num
    config['dataset']['env']['gt_rot_perturb'] = args.gt_rot_perturb
    config['dataset']['env']['gt_tsl_perturb'] = args.gt_tsl_perturb
    main(config, args.model_config)
