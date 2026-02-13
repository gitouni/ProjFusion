import os
import numpy as np
from models.util.nptrans import toMatw
from scipy.spatial.transform import Rotation
import argparse
from typing import Tuple
from models.util.transform import inv_pose_np
# from pprint import pprint
from collections import defaultdict
import json
from pathlib import Path

def se3_err(pred_se3:np.ndarray, gt_se3:np.ndarray) -> Tuple[np.ndarray,np.ndarray, np.ndarray]:
    delta_se3 = pred_se3 @ inv_pose_np(gt_se3)
    delta_euler = np.abs(Rotation.from_matrix(delta_se3[...,:3,:3]).as_euler(seq='XYZ',degrees=True))  # (B, 3)
    geodesic_distance = np.rad2deg(np.arccos(np.clip((np.trace(delta_se3[...,:3,:3]) - 1) / 2, -1.0, 1.0)))
    delta_tsl = np.abs(delta_se3[...,:3,3])  # (B, 3)
    return delta_euler, delta_tsl, geodesic_distance   # (B, 3), (B, 3)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir_root",type=str,default="experiments/kitti/projdualfusion_rope_r10_t0.5/results/projdualfusion_rope_r10_t0.5")
    parser.add_argument("--gt_dir",type=str,default="cache/kitti_gt")
    parser.add_argument("--log_file",type=str,default="log/ablation/projdualfusion_rope_r10_t0.5.json")
    parser.add_argument("--sample_num", type=int, default=500, help="number of samples for evaluation")
    parser.add_argument("--L1", type=float, default=[1.0, 2.5], help="threshold of L1 metric (deg, cm)")
    parser.add_argument("--L2", type=float, default=[2.0, 5.0], help="threshold of L2 metric (deg, cm)")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    gt_files = sorted(os.listdir(args.gt_dir))
    pred_dirs = sorted(os.listdir(args.pred_dir_root))
    assert len(gt_files) == len(pred_dirs), "number of gt files ({}) != number of pred subdirs ({})".format(len(gt_files), len(pred_dirs))
    names = pred_dirs
    for pred_subdir in pred_dirs:
        pred_dir = os.path.join(args.pred_dir_root, pred_subdir)
        pred_files = sorted(os.listdir(pred_dir))
        sample_num = min(args.sample_num, len(pred_files))
    print("Compute metrics on {} with sample_num={}".format(names, sample_num))
    all_metrics = defaultdict(list)
    meta = defaultdict(lambda: defaultdict(float))
    log_path = os.path.dirname(args.log_file)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    for name, gt_file, pred_subdir in zip(names, gt_files, pred_dirs):
        gt_se3 = np.loadtxt(os.path.join(args.gt_dir, gt_file))
        pred_dir = os.path.join(args.pred_dir_root, pred_subdir)
        pred_files = sorted(os.listdir(pred_dir))
        if len(pred_files) > sample_num:
            index = np.unique(np.linspace(0, len(pred_files)-1, sample_num, endpoint=True, dtype=int))
            pred_files = [pred_files[i] for i in index]
        for i, pred_file in enumerate(pred_files):
            pred_se3_i = np.loadtxt(os.path.join(pred_dir, pred_file))
            if np.ndim(pred_se3_i) == 2:
                pred_se3_i = pred_se3_i[-1]  # sequences of prediction
            pred_se3 = toMatw(pred_se3_i)
            R_err_i, t_err_i, geodesic_distance_i = se3_err(pred_se3, gt_se3)
            RRMSE = np.linalg.norm(R_err_i)
            tRMSE = np.linalg.norm(t_err_i)
            all_metrics['Rx'].append(R_err_i[0])
            all_metrics['Ry'].append(R_err_i[1])
            all_metrics['Rz'].append(R_err_i[2])
            all_metrics['tx'].append(t_err_i[0])
            all_metrics['ty'].append(t_err_i[1])
            all_metrics['tz'].append(t_err_i[2])
            all_metrics['RRMSE'].append(RRMSE)
            all_metrics['tRMSE'].append(tRMSE)
            all_metrics['geodesic_distance'].append(geodesic_distance_i)
            all_metrics['RMAE'].append(np.mean(np.abs(R_err_i)))
            all_metrics['tMAE'].append(np.mean(np.abs(t_err_i)))
    # average
    for metric in all_metrics.keys():
        meta[metric]['mean'] = np.mean(all_metrics[metric]).item()
        meta[metric]['std'] = np.std(all_metrics[metric]).item()
    meta['success_rate']['L1'] = np.mean((np.array(all_metrics['RRMSE']) < args.L1[0]) &\
                                         (np.array(all_metrics['tRMSE']) < args.L1[1] / 100)).item()
    meta['success_rate']['L2'] = np.mean((np.array(all_metrics['RRMSE']) < args.L2[0]) &\
                                         (np.array(all_metrics['tRMSE']) < args.L2[1] / 100)).item()
    json.dump(meta, open(args.log_file,'w'),indent=2)
    print('log file saved to {}'.format(args.log_file))