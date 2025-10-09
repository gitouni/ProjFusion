#!/bin/bash

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti.yml"
MODEL_CONFIG="cfg/model/ebm_nq_attn.yml"
BASE_DIR='attn_t'
TASK_NAME='kitti_nq'
TR_NUM=384
TR_ACC_NUM=1
VAL_NUM=384
VAL_REPEAT=4
echo "Run Exp: ${MODEL_CONFIG} ${DATASET_CONFIG}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} '--tr_num' ${TR_NUM} '--tr_acc_num' ${TR_ACC_NUM} '--val_num' ${VAL_NUM} '--val_repeat' ${VAL_REPEAT}

