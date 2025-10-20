from tqdm import tqdm
import argparse
from typing import List, Union, Generator, Tuple, Dict, Iterable, Callable
from collections import defaultdict
from itertools import islice
from dataset import __classdict__ as DatasetDict, DATASET_TYPE, PerturbDataset, BatchedPerturbDatasetOutput
import yaml
import yaml_include  # pip install pyyaml-include
yaml.add_constructor("!inc", yaml_include.Constructor(base_dir='.'), yaml.SafeLoader)
from logging import Logger
from pathlib import Path
import shutil
import logging
from copy import deepcopy
# import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from core.logger import LogTracker, fmt_time
from core.tools import load_checkpoint_model_only
from core.tools import Timer
from models.loss import se3_err, se3_reduce, se3_rmse, get_loss
from models.util import se3
from models.model import ProjFusion, ProjDualFusion

def get_dataloader(test_dataset_argv:Iterable[Dict],
        test_dataloader_argv:Dict, dataset_type:str) -> Tuple[List[str], List[Generator[BatchedPerturbDatasetOutput, None, None]]]:
    name_list = []
    dataloader_list = []
    data_class: DATASET_TYPE = DatasetDict[dataset_type]
    if isinstance(test_dataset_argv, list):
        for dataset_argv in test_dataset_argv:
            name_list.append(dataset_argv['name'])
            base_dataset = data_class(**dataset_argv['base'])
            dataset = PerturbDataset(base_dataset, **dataset_argv['main'])
            if hasattr(dataset, 'collate_fn'):
                test_dataloader_argv['collate_fn'] = getattr(dataset, 'collate_fn')
            dataloader = DataLoader(dataset, **test_dataloader_argv)
            dataloader_list.append(dataloader)
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
    return name_list, dataloader_list

def to_npy(x0:torch.Tensor) -> np.ndarray:
    return x0.detach().cpu().numpy()

@torch.inference_mode()
def test_seq(model: Union[ProjFusion, ProjDualFusion], logger: Logger, loader: Generator[BatchedPerturbDatasetOutput, None, None],
             seq_name: str, res_dir: Path, log_per_iter: int, run_iter: int, device: torch.device, cnt: int=0):
    model.eval()
    progress = tqdm(loader, total=len(loader), desc=seq_name)
    log_tracker = LogTracker('rot_err','tsl_err','se3_err','time', phase='test')
    num_batches = len(loader)
    if isinstance(res_dir, Path):
        res_dir = res_dir
    else:
        res_dir = Path(res_dir)
    for i, data in enumerate(islice(progress, cnt, None)):
        img = data['img'].to(device)
        pcd = data['pcd'].to(device)
        batch_n = len(img)
        init_extran = data['init_extran'].to(device)
        gt_extran = data['gt_extran'].to(device)
        gt_delta_extran = data['pose_target'].to(device)
        gt_log = se3.log(gt_delta_extran)
        camera_info = data['camera_info']
        se3_xlist = [to_npy(se3.log(init_extran))]
        pred_extran = init_extran
        with Timer() as timer, model.cache_manager(img, pcd):  # use cache to accelerate iteration
            for _ in range(run_iter):
                rot_log, tsl_log = model(img, pcd, pred_extran, camera_info)
                pred_log = torch.cat([rot_log, tsl_log], dim=-1)
                pred_extran = se3.exp(pred_log) @ pred_extran
                se3_xlist.append(to_npy(se3.log(pred_extran)))
        log_tracker.update('time', timer.elapsed_time, batch_n)
        rot_err, tsl_err = se3_err(pred_extran, gt_extran)  # set the Tcl with the largest logits as the predicted one
        se3_loss = se3_reduce(rot_err, tsl_err)
        batched_x0_list = np.stack(se3_xlist, axis=1)  # (B, K, 6)
        for x0 in batched_x0_list:
            np.savetxt(res_dir.joinpath("%06d.txt"%cnt), x0)
            cnt += 1
        se3_loss = se3_reduce(rot_err, tsl_err)
        log_tracker.update('rot_err', se3_rmse(rot_err).mean().item(), batch_n)
        log_tracker.update('tsl_err', se3_rmse(tsl_err).mean().item(), batch_n)
        log_tracker.update('se3_err', se3_loss.mean().item(), batch_n)
        if (i+1) % log_per_iter == 0 or i+1 == len(loader):
            logger.info("\tBatch {}|{} {}".format(i+1, num_batches, log_tracker.result()))
    return log_tracker.result()

def main(config:Dict, resume:bool):
    run_argv: Dict = config['run']
    path_argv: Dict = config['path']
    np.random.seed(run_argv['seed'])
    torch.manual_seed(run_argv['seed'])
    device = run_argv['device']
    # torch.backends.cudnn.benchmark=True
    # torch.backends.cudnn.enabled = False
    dataset_argv = config['dataset']['test']
    dataset_type = config['dataset']['type']
    name_list, dataloader_list = get_dataloader(dataset_argv['dataset'], dataset_argv['dataloader'], dataset_type)
    if config['model']['type'] == 'ProjDualFusion':
        MODEL_CLASS = ProjDualFusion
    elif config['model']['type'] == 'ProjFusion':
        MODEL_CLASS = ProjFusion
    else:
        raise NotImplementedError(f"Unknown model type: {config['model']['type']}")
    model = MODEL_CLASS(**config['model']['args']).to(device)
    # loss_fn = get_loss(**config['loss'])
    # summary(model)
    # exit(0)
    experiment_dir = Path(path_argv['base_dir'])
    experiment_dir.mkdir(exist_ok=True, parents=True)
    experiment_dir = experiment_dir.joinpath(path_argv['name'])
    experiment_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath(path_argv['log'])
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(path_argv['log'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_mode = 'a' if path_argv['resume'] is not None else 'w'
    file_handler = logging.FileHandler(str(log_dir) + '/test_{}_{}.log'.format(path_argv['name'], fmt_time()), mode=logger_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('args: ')
    logger.info(args)
    if path_argv['pretrain'] is not None:
        load_checkpoint_model_only(path_argv['pretrain'], model)
        logger.info("Loaded checkpoint from {}".format(path_argv['pretrain']))
    else:
        raise FileNotFoundError("'pretrain' cannot be set to 'None' during test-time")
    record_list: List[Tuple[str, Dict[str, float]]] = []
    res_dir = experiment_dir.joinpath(path_argv['results']).joinpath(path_argv['name'])
    res_dir.mkdir(exist_ok=True, parents=True)
    for name, dataloader in zip(name_list, dataloader_list):
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
        record = test_seq(model, logger, dataloader, name, sub_res_dir, 
            run_argv['log_per_iter'], run_argv['run_iter'], device, cnt)
        logger.info("{}: {}".format(name, record))
        record_list.append([name, record])
    total_record = defaultdict(float)
    logger.info("Summary:")  # view in the bottom
    for name, record in record_list:
        logger.info("{}: {}".format(name, record))
        for key, value in record.items():
            total_record[key] += value
    for key, value in total_record.items():
        total_record[key] = value / len(record_list)
    logger.info("total: {}".format(total_record))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/kitti/projdualfusion_harmonic/log/projdualfusion_harmonic.yml")
    parser.add_argument("--run_iter", type=int, default=1)
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config: Dict  = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    config['run']['run_iter'] = args.run_iter
    if args.name:
        config['path']['name'] = args.name
    main(config, args.resume)
