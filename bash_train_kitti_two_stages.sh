#!/bin/bash

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti.yml"
MODEL_CONFIG="cfg/model/ebm_nq.yml"
BASE_DIR='pool_t'
TASK_NAME='kitti_rot_only'
echo "Run Exp: ${MODEL_CONFIG} ${DATASET_CONFIG} ${BASE_DIR} ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} '--gt_tsl_perturb'

TASK_NAME='kitti_tsl_only'
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} '--gt_rot_perturb'