#!/bin/bash

# KITTI Odometry

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti_r15_t0.15.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_r15_t0.15'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti_r10_t0.25.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_r10_t0.25'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti_r10_t0.5.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_r10_t0.5'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}

# nuScenes

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/nusc_r15_t0.15.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='nusc'
TASK_NAME='projdualfusion_harmonic_r15_t0.15'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/nusc_r10_t0.25.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='nusc'
TASK_NAME='projdualfusion_harmonic_r10_t0.25'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 

PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/nusc_r10_t0.5.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='nusc'
TASK_NAME='projdualfusion_harmonic_r10_t0.5'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 

DATASET_CONFIG="cfg/dataset/kitti_r10_t0.5.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic_resnet.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_resnet_r10_t0.5'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 

DATASET_CONFIG="cfg/dataset/kitti_r10_t0.5.yml"
MODEL_CONFIG="cfg/model/projdualfusion.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_r10_t0.5'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 

DATASET_CONFIG="cfg/dataset/kitti_r10_t0.5.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic_f2.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_f2_r10_t0.5'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 

DATASET_CONFIG="cfg/dataset/kitti_r10_t0.5.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic_f10.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_f10_r10_t0.5'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 
