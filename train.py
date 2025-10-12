import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse

# from models.embedding import PredictMode
from typing import Literal, Generator, Tuple, Dict, Union, Iterable, Callable
from dataset import  __classdict__ as DatasetDict, DATASET_TYPE, PerturbDataset, SeqBatchSampler, BatchedPerturbDatasetOutput
from torch.utils.data import DataLoader
import yaml
import yaml_include
yaml.add_constructor("!inc", yaml_include.Constructor(base_dir='.'), yaml.SafeLoader)
from logging import Logger
from pathlib import Path
import logging
# import torch.nn as nn
from accelerate import Accelerator
from core.tools import load_checkpoint, save_checkpoint
from core.logger import LogTracker, fmt_time, print_warning
from models.lr_scheduler import get_lr_scheduler, get_optimizer
from models.loss import se3_err, se3_reduce, se3_rmse, get_loss
from models.util import se3
from models.model import ProjFusion, ProjDualFusion

def unique_append(s:str, arr:list):
    if not s in arr:
        arr.append(s)

def get_dataloader(dataset_type: Literal['kitti','nuscenes'],
        train_base_dataset_argv: Dict, train_dataset_argv: Dict,
        val_base_dataset_argv: Dict, val_dataset_argv: Dict,
        train_dataloader_argv: Dict, val_dataloader_argv: Dict) -> Tuple[Generator[BatchedPerturbDatasetOutput, None, None], Generator[BatchedPerturbDatasetOutput, None, None]]:
    dataset_class = DatasetDict[dataset_type]
    train_base_dataset:DATASET_TYPE = dataset_class(**train_base_dataset_argv)
    val_base_dataset:DATASET_TYPE = dataset_class(**val_base_dataset_argv)
    train_dataset = PerturbDataset(train_base_dataset, **train_dataset_argv)
    val_dataset = PerturbDataset(val_base_dataset, **val_dataset_argv)
    if 'batch_sampler' in train_dataloader_argv:
        train_dataloader_argv['batch_sampler'] = SeqBatchSampler(*train_base_dataset.get_seq_params(), **train_dataloader_argv['batch_sampler'])
    if 'batch_sampler' in val_dataloader_argv:
        val_dataloader_argv['batch_sampler'] = SeqBatchSampler(*val_base_dataset.get_seq_params(), **val_dataloader_argv['batch_sampler'])
    if hasattr(train_dataset, 'collate_fn'):
        train_dataloader_argv['collate_fn'] = getattr(train_dataset, 'collate_fn')
    if hasattr(val_dataset, 'collate_fn'):
        val_dataloader_argv['collate_fn'] = getattr(val_dataset, 'collate_fn')
    train_dataloader: Generator[BatchedPerturbDatasetOutput, None, None] = DataLoader(train_dataset, **train_dataloader_argv)
    val_dataloader: Generator[BatchedPerturbDatasetOutput, None, None] = DataLoader(val_dataset, **val_dataloader_argv)
    return train_dataloader, val_dataloader

@torch.inference_mode()
def val_epoch(model: ProjFusion, logger: Logger, loader: Generator[BatchedPerturbDatasetOutput, None, None], 
        log_per_iter: int, loss_fn: Callable[..., torch.Tensor], device: torch.device) -> float:
    model.eval()
    progress = tqdm(loader, total=len(loader))
    log_tracker = LogTracker('loss','rot_err','tsl_err','se3_err', phase='val')
    num_batches = len(loader)
    for i, data in enumerate(progress):
        img = data['img'].to(device)
        pcd = data['pcd'].to(device)
        init_extran = data['init_extran'].to(device)
        gt_extran = data['pose_target'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
        gt_log = se3.log(gt_extran)
        camera_info = data['camera_info']
        rot_log, tsl_log = model(img, pcd, init_extran, camera_info)
        pred_log = torch.cat([rot_log, tsl_log], dim=-1)
        pred_extran = se3.exp(pred_log)
        loss = loss_fn(pred_log, gt_log)
        rot_err, tsl_err = se3_err(pred_extran, gt_extran)  # set the Tcl with the largest logits as the predicted one
        se3_loss = se3_reduce(rot_err, tsl_err)
        log_tracker.update('loss', loss.item())
        log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())
        log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())
        log_tracker.update('se3_err', se3_loss.mean().item())
        if (i+1) % log_per_iter == 0 or i+1 == len(loader):
            logger.info("\tBatch {}|{} {}".format(i+1, num_batches, log_tracker.result()))
    return log_tracker.avg('se3_err')

def main(config:Dict, config_path:Union[str, Iterable[str]]):
    run_argv: Dict = config['run']
    path_argv: Dict = config['path']
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True, parents=True)
    experiment_dir = experiment_dir.joinpath(path_argv['name'])
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath(path_argv['checkpoint'])
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    yaml.safe_dump(config, open(str(log_dir.joinpath(os.path.basename(config_path))),'w'))
    np.random.seed(config['run']['seed'])
    torch.manual_seed(config['run']['seed'])
    device = config['run']['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    dataset_argv = config['dataset']['train']  # train and val
    train_dataloader, val_dataloader = get_dataloader(
        config['dataset']['type'], 
        dataset_argv['dataset']['train']['base'], dataset_argv['dataset']['train']['main'], 
        dataset_argv['dataset']['val']['base'], dataset_argv['dataset']['val']['main'],
        dataset_argv['dataloader']['args'], dataset_argv['dataloader']['val_args'])
    loss_fn = get_loss(**config['loss'])
    if config['model']['type'] == 'ProjDualFusion':
        MODEL_CLASS = ProjDualFusion
    elif config['model']['type'] == 'ProjFusion':
        MODEL_CLASS = ProjFusion
    else:
        raise NotImplementedError(f"Unknown model type: {config['model']['type']}")
    model = MODEL_CLASS(**config['model']['args']).to(device)
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), config['optimizer']['type'], **config['optimizer']['args'])
    clip_grad = config['optimizer']['max_grad']
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
    # accelerator
    accelerator = Accelerator(
                        mixed_precision=run_argv.get('precision', 'bf16'),
                        gradient_accumulation_steps=run_argv.get('train_gradient_acc_steps', 1),
                    )
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
            val_epoch(model, logger, val_dataloader, run_argv['log_per_iter'], loss_fn, device)
    else:
        start_epoch = 1
        best_loss = float('inf')
        logger.info("Start from scratch")
    # 初始化 GradScaler
    if accelerator is not None:
        model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, val_dataloader
        )
    else:
        model = model.to(device)
    optimizer.zero_grad()
    for epoch in range(start_epoch, run_argv['n_epoch'] + 1):  # start epoch start from 1
        log_tracker_keys = ['loss','rot_err','tsl_err','se3_err']
        log_tracker = LogTracker(*log_tracker_keys, phase='train')
        # -- train
        model.train()
        progress: Generator[BatchedPerturbDatasetOutput, None, None] = tqdm(train_dataloader, total=len(train_dataloader))  # dataloader with tqdm
        for bi, data in enumerate(progress):
            with accelerator.accumulate(model):
                img = data['img'].to(device)
                pcd = data['pcd'].to(device)
                init_extran = data['init_extran'].to(device)
                gt_extran = data['pose_target'].to(device)  # transform uncalibrated_pcd to calibrated_pcd
                gt_log = se3.log(gt_extran)
                camera_info = data['camera_info']
                rot_log, tsl_log = model(img, pcd, init_extran, camera_info)
                pred_log = torch.cat([rot_log, tsl_log], dim=-1)
                loss = loss_fn(pred_log, gt_log)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(
                        model.parameters(), 
                        clip_grad, 
                        norm_type=2
                    )
                optimizer.step()
                optimizer.zero_grad()
                with torch.inference_mode():
                    pred_extran = se3.exp(pred_log)
                    rot_err, tsl_err = se3_err(pred_extran, gt_extran)  # set the Tcl with the largest logits as the predicted one
                    se3_loss = se3_reduce(rot_err, tsl_err)
                log_tracker.update('loss', loss.item())
                log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())
                log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())
                log_tracker.update('se3_err', se3_loss.mean().item())
                # 更新参数
                if (bi + 1) % run_argv['log_per_iter'] == 0:
                    logger.info("\tBatch {}|{}: {}".format(bi+1, len(progress), log_tracker.result()))
                    # logger.debug("\tBatch {}|{}: {}".format(bi+1, len(progress), traj_tracker.result()))
        scheduler.step()
        logger.info("Epoch {}|{}: {}".format(epoch, run_argv['n_epoch'], log_tracker.result()))
        if epoch % run_argv['val_per_epoch'] == 0:
            val_loss = val_epoch(model, logger, val_dataloader, run_argv['log_per_iter'], loss_fn, device)
            if val_loss < best_loss:
                logger.info("Find Best Model at Epoch {} prev | curr best loss: {} | {}".format(epoch, best_loss, val_loss))
                best_loss = val_loss
                save_checkpoint(checkpoints_dir.joinpath('best_model.pth'), epoch, best_loss, model, optimizer, scheduler)
        save_checkpoint(checkpoints_dir.joinpath('last_model.pth'), epoch, best_loss, model, optimizer, scheduler)
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
    parser.add_argument("--model_config",type=str,default="cfg/model/projdualfusion.yml")
    parser.add_argument("--base_dir",type=str,default="kitti")
    parser.add_argument("--task_name",type=str,default="debug")
    parser.add_argument("--resume",type=str,default=None)
    parser.add_argument("--resume_eval_first", action="store_true")
    parser.add_argument("--tr_acc_num",type=int, default=1)
    args = parser.parse_args()
    dataset_config:Dict = yaml.load(open(args.dataset_config,'r'), yaml.SafeLoader)
    config: Dict  = yaml.load(open(args.model_config,'r'), yaml.SafeLoader)
    config.update(dataset_config)
    config['path']['name'] = args.task_name
    config['path']['base_dir'] = config['path']['base_dir'].format(base_dir=args.base_dir)
    config['path']['pretrain'] = config['path']['pretrain'].format(base_dir=args.base_dir, task_name=args.task_name)
    if args.resume:
        config['path']['resume'] = args.resume
        config['path']['resume_eval_first'] = args.resume_eval_first
    config['run']['train_gradient_acc_steps'] = args.tr_acc_num
    # args.rot_only == True and args.tsl_only == True, there will be two gt_mat in one sampling (B, 2, 4, 4) + (B, G-2, 4, 4), rotation gt first
    main(config, args.model_config)
