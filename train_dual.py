import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse
from typing import Literal, Generator, Tuple, Dict, Union, Iterable
from dataset import BaseKITTIDataset, NuSceneDataset, SeqBatchSampler, BatchedBaseDatasetOutput, BatchedCameraInfoDict
from torch.utils.data import DataLoader
import yaml

from logging import Logger
from pathlib import Path
import logging
import torch.nn as nn
from einops import rearrange, repeat
from accelerate import Accelerator
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
        gt_extran = data['extran'].to(device=device)  # (B, 4, 4)
        camera_info = data['camera_info']
        global_best_Tcl = None
        best_logits = None
        query_gt_extran = None
        last_err = None
        curr_extran = env.perturb(gt_extran, 1, disentangled=True, concat_input=False).squeeze(1)  # init_extran (B, 4, 4)，不包含GT矩阵
        with model.state_emb.model_buffer_manager(img, pcd):  # store and clear buffer automatically
            for _ in range(repeat_perturb):
                query_rot_curr_extran = env.perturb(curr_extran, num_perturbations, disentangled=True, concat_input=True, add_rot_perturb=True, add_tsl_perturb=False)  # (B, G, 4, 4)
                _, _, best_rot_Tcl = model(img, pcd, query_rot_curr_extran, camera_info, predict_mode=PredictMode.RotOnly, mode=EBMMode.LOGIT_IDX)  # (B, G), (B,), (B, 4, 4)
                query_curr_extran = env.perturb(best_rot_Tcl, num_perturbations, disentangled=False, concat_input=True, add_rot_perturb=False, add_tsl_perturb=True)  # (B, G, 4, 4) 
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
    run_argv:Dict = config['run']
    print("Run args:")
    for key, value in run_argv.items():
        print(f"{key}: {value}")
    path_argv:Dict = config['path']
    n_acc_steps = run_argv.get('train_gradient_acc_steps', 4)
    base_num_perturb = run_argv.get('train_num_perturbations', 256)  # 每次 perturb 的数量（显存可承受）
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
    for key, value in config['scheduler']['args'].items():
        if key.endswith('steps'):
            config['scheduler']['args'][key] = int(len(train_dataloader) * value)   # epochs -> steps
    print("scheduler args:")
    for key, value in config['scheduler']['args'].items():
        print(f"{key}: {value}")
    scheduler = get_lr_scheduler(optimizer, config['scheduler']['type'], **config['scheduler']['args'])
    accelerator = Accelerator(
                        mixed_precision=run_argv.get('precision', 'bf16'),
                        gradient_accumulation_steps=n_acc_steps,
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
            with env.set_mag_attr_temp(**config['dataset']['env']['val']):
                val_epoch(env, model, logger, val_dataloader, run_argv['log_per_iter'], run_argv['val_repeat_perturb'], run_argv['val_num_perturbations'], device)
    else:
        start_epoch = 1
        best_loss = float('inf')
        logger.info("Start from scratch")
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )

    seperate_optimize = run_argv.get('seperate_optimize', True)
    for epoch in range(start_epoch, run_argv['n_epoch'] + 1):  # start epoch start from 1
        log_tracker_keys = ['rot_ce', 'rot_err', 'tsl_ce', 'tsl_err']
        log_tracker = LogTracker(*log_tracker_keys, phase='train')

        # -- train
        model.train()
        progress = tqdm(train_dataloader, total=len(train_dataloader))
        with progress:
            for bi, data in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    img: torch.Tensor = data['img'].to(device)
                    pcd: torch.Tensor = data['pcd'].to(device)
                    camera_info: BatchedCameraInfoDict = data['camera_info']
                    gt_extran: torch.Tensor = data['extran'].to(device)

                    total_rot_loss = 0.0
                    total_tsl_loss = 0.0
                    count = 0
                    
                    with model.state_emb.model_buffer_manager(img, pcd):
                        if not seperate_optimize:  # 同时优化旋转和平移
                            with torch.no_grad():
                                query_extran = env.perturb(
                                    gt_extran, base_num_perturb, disentangled=True, concat_input=True,
                                    add_rot_perturb_to_gt=True, add_tsl_perturb_to_gt=True
                                )  # (B, G, 4, 4)

                            logits, labels, _, best_Tcl, _ = model(
                                img, pcd, query_extran, camera_info,
                                predict_mode=PredictMode.Both,
                                mode=EBMMode.LOGIT_LABEL_IDX_STATE
                            )
                            (rot_logits, rot_labels, rot_best_Tcl), (tsl_logits, tsl_labels, tsl_best_Tcl) = \
                                zip(logits, labels, best_Tcl)
                            loss_rot = F.cross_entropy(rot_logits, rot_labels, reduction='mean')
                            loss_tsl = F.cross_entropy(tsl_logits, tsl_labels, reduction='mean')
                            loss = loss_rot + loss_tsl

                            # 推理阶段计算误差（不参与梯度）
                            with torch.inference_mode():
                                rot_err, _ = se3_err(rot_best_Tcl, gt_extran)
                                _, tsl_err = se3_err(tsl_best_Tcl, gt_extran)
                                log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())
                                log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())

                            # 正常反向传播（不再用 scaler.scale）
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(
                                    model.parameters(), 
                                    clip_grad, 
                                    norm_type=2
                                )
                        else:  # 分别优化旋转和平移
                            with torch.no_grad():
                                init_rot_extran = env.perturb(gt_extran, 1, disentangled=False, concat_input=False, add_rot_perturb=False, add_tsl_perturb=True).squeeze(1)  # 初始矩阵，可能存在平移误差
                                query_rot_extran = env.perturb(
                                    init_rot_extran, base_num_perturb, disentangled=True, concat_input=True, add_rot_perturb=True, add_tsl_perturb=False
                                )  # (B, G, 4, 4) 只含有旋转扰动的矩阵
                            rot_logits, rot_labels, _, rot_best_Tcl, _ = model(
                                img, pcd, query_rot_extran.detach(), camera_info,
                                predict_mode=PredictMode.RotOnly,
                                mode=EBMMode.LOGIT_LABEL_IDX_STATE
                            )
                            loss_rot = F.cross_entropy(rot_logits, rot_labels, reduction='mean')
                            accelerator.backward(loss_rot)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(
                                    model.parameters(), 
                                    clip_grad, 
                                    norm_type=2
                                )
                            with torch.no_grad(), env.set_mag_attr_temp(**config['dataset']['env']['small_rot']):  # 只拷贝真值的平移部分，继承旋转部分的初值
                                init_tsl_extran = env.perturb(gt_extran, 1, disentangled=True, concat_input=False, add_rot_perturb=True, add_tsl_perturb=False).squeeze(1)  # 初始矩阵，可能存在平移误差
                                query_tsl_extran = env.perturb(
                                    init_tsl_extran.detach(), base_num_perturb, disentangled=False, concat_input=True, add_rot_perturb=False, add_tsl_perturb=True
                                )  # (B, G, 4, 4) 只含有旋转扰动的矩阵
                            tsl_logits, tsl_labels, _, tsl_best_Tcl, _ = model(
                                img, pcd, query_tsl_extran, camera_info,
                                predict_mode=PredictMode.TslOnly,
                                mode=EBMMode.LOGIT_LABEL_IDX_STATE
                            )
                            loss_tsl = F.cross_entropy(tsl_logits, tsl_labels, reduction='mean')
                            accelerator.backward(loss_tsl)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(
                                    model.parameters(), 
                                    clip_grad, 
                                    norm_type=2
                                )
                            # 推理阶段计算误差（不参与梯度）
                            with torch.inference_mode():
                                rot_err, _ = se3_err(rot_best_Tcl, gt_extran)
                                _, tsl_err = se3_err(tsl_best_Tcl, gt_extran)
                                log_tracker.update('rot_err', se3_rmse(rot_err).mean().item())
                                log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item())
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        total_rot_loss += loss_rot.item()
                        total_tsl_loss += loss_tsl.item()
                        count += 1

                        progress.set_description(
                            desc="B: {}|{}, loss_rot: {:.3f}, loss_tsl: {:.3f}".format(
                                bi + 1, len(train_dataloader), loss_rot, loss_tsl
                            )
                        )
                        progress.update(1)
                        # 记录平均 loss
                        avg_rot_loss = total_rot_loss / count
                        avg_tsl_loss = total_tsl_loss / count
                        log_tracker.update('rot_ce', avg_rot_loss)
                        log_tracker.update('tsl_ce', avg_tsl_loss)

                        if (bi + 1) % run_argv['log_per_iter'] == 0:
                            logger.info("\tBatch {}|{}: {}".format(bi + 1, len(progress), log_tracker.result()))
        logger.info("Epoch {}|{}: {}".format(epoch, run_argv['n_epoch'], log_tracker.result()))
        if epoch % run_argv['val_per_epoch'] == 0:
            with env.set_mag_attr_temp(**config['dataset']['env']['val']):
                val_loss = val_epoch(env, model, logger, val_dataloader, run_argv['log_per_iter'], run_argv['val_repeat_perturb'], run_argv['val_num_perturbations'], device)
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
    parser.add_argument("--model_config",type=str,default="cfg/model/ebm_dual_attn_fusion_split.yml")
    parser.add_argument("--base_dir",type=str,default="attn_t")
    parser.add_argument("--task_name",type=str,default="kitti_rot_tsl_debug")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_eval_first", action="store_true")
    parser.add_argument("--tr_num", type=int, default=256)
    parser.add_argument("--tr_acc_num",type=int, default=1)
    parser.add_argument("--val_repeat", type=int, default=4)
    parser.add_argument("--val_num", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    
    if args.config is None:  # 采用分离式加载config，model_config采用分级式加载
        import yaml_include
        yaml.add_constructor("!inc", yaml_include.Constructor(base_dir='.'), yaml.SafeLoader)
        dataset_config: Dict = yaml.safe_load(open(args.dataset_config,'r'))
        config: Dict = yaml.safe_load(open(args.model_config,'r'))
        base_config: Dict  = yaml.safe_load(open(config['base'],'r'))
        config.update(base_config)
        config.update(dataset_config)
    else:
        config: Dict = yaml.safe_load(open(args.config, 'r'))
        config_path = args.config
    config['path']['name'] = args.task_name
    config['path']['base_dir'] = config['path']['base_dir'].format(base_dir=args.base_dir)
    config['path']['pretrain'] = config['path']['pretrain'].format(base_dir=args.base_dir, task_name=args.task_name)
    if args.resume:
        config['path']['resume'] = args.resume
        print(f"\033[33;1mResume from {args.resume}\033[0m")
        config['path']['resume_eval_first'] = args.resume_eval_first
    config['run']['train_num_perturbations'] = args.tr_num
    config['run']['train_gradient_acc_steps'] = args.tr_acc_num
    config['run']['val_repeat_perturb'] = args.val_repeat
    config['run']['val_num_perturbations'] = args.val_num
    if args.lr is not None:
        config['optimizer']['args']['lr'] = args.lr
    config_path = args.model_config
    PREDICT_MODE = PredictMode.Both
    # args.rot_only == True and args.tsl_only == True, there will be two gt_mat in one sampling (B, 2, 4, 4) + (B, G-2, 4, 4), rotation gt first
    main(config, config_path)
