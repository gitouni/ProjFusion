#!/bin/bash

PYTHON_PATH="python"
SCRIPT_PATH="train_dual_bk.py"
DATASET_CONFIG="cfg/dataset/kitti.yml"
MODEL_CONFIG="cfg/model/ebm_nq_dual_pool.yml"
BASE_DIR='pool_t'
TASK_NAME='kitti_rot_tsl'
TR_NUM="256"
VAL_NUM="512"
VAL_REPEAT="4"
echo "Run Exp: ${MODEL_CONFIG} ${DATASET_CONFIG} ${BASE_DIR} ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} '--tr_num' ${TR_NUM} '--val_num' ${VAL_NUM} '--val_repeat' ${VAL_REPEAT}
