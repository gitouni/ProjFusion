#!/bin/bash

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic'

${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 

MODEL_CONFIG="cfg/model/projdualfusion.yml"
TASK_NAME='projdualfusion'
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 