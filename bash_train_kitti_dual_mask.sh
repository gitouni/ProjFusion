#!/bin/bash

PYTHON_PATH="python"
SCRIPT_PATH="train_dual.py"
DATASET_CONFIG="cfg/dataset/kitti_fast.yml"
MODEL_CONFIG="cfg/model/ebm_dual_attn_fusion_mask.yml"
BASE_DIR='attn_t'
TASK_NAME=kitti_rot_tsl_sep_mlp_mask
TR_NUM=512
TR_ACC_NUM=1
VAL_NUM=512
VAL_REPEAT=4
echo "Run Exp: ${MODEL_CONFIG} ${DATASET_CONFIG} ${BASE_DIR} ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} '--tr_num' ${TR_NUM} '--tr_acc_num' ${TR_ACC_NUM} '--val_num' ${VAL_NUM} '--val_repeat' ${VAL_REPEAT}
