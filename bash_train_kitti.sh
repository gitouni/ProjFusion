#!/bin/bash

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti_large.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_lerr'

${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 

# PYTHON_PATH="python"
# SCRIPT_PATH="train.py"
# DATASET_CONFIG="cfg/dataset/nusc.yml"
# MODEL_CONFIG="cfg/model/projdualfusion_harmonic_attn.yml"
# BASE_DIR='nusc'
# TASK_NAME='projdualfusion_harmonic_attn'

# ${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 