<h1 align="center">🚀 NativeCross</h1>
<p align="center">
  <b>Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations</b>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/gitouni/ProjFusion?style=social" alt="GitHub Repo stars"/>
  <img src="https://img.shields.io/github/forks/gitouni/ProjFusion?style=social" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/last-commit/gitouni/ProjFusion" alt="GitHub last commit"/>
  <a href="https://ieeexplore.ieee.org/document/11480778">
    <img src="https://img.shields.io/badge/IEEE-Published-blue.svg" alt="IEEE"/>
  </a>
  <a href="https://arxiv.org/abs/2603.29414">
    <img src="https://img.shields.io/badge/arXiv-2603.29414-b31b1b.svg" alt="arXiv"/>
  </a>
</p>


This paper address the problem of camera-LiDAR extrinsic calibration when the initial pose estimate is significantly misaligned. Most existing methods rely on depth map projections to extract point cloud features, but large extrinsic perturbations cause most LiDAR points to project outside the image boundary, degrading feature quality and calibration accuracy.

<div align="center">
  <img src="assets/miscalibrated-depth.jpg" alt="miscalibrated_depth_map" width="60%">
</div>



We propose a native-domain cross-attention mechanism that directly aligns camera and LiDAR features without depth map projection, maintaining geometric consistency and achieving robust calibration even under large initial misalignments (see Abstract for details).
<div align="center">
  <img src="assets/abstract.jpg" alt="cross_attention" width="60%">
</div>

## Abstract
Accurate camera-LiDAR fusion relies on precise extrinsic calibration, which fundamentally depends on establishing reliable cross-modal correspondences under potentially large misalignments. Existing learning-based methods typically project LiDAR points into depth maps for feature fusion, which distorts 3D geometry and degrades performance when the extrinsic initialization is far from the ground truth. To address this issue, we propose an extrinsic-aware cross-attention framework that directly aligns image patches and LiDAR point groups in their native domains. The proposed attention mechanism explicitly injects extrinsic parameter hypotheses into the correspondence modeling process, enabling geometry-consistent cross-modal interaction without relying on projected 2D depth maps. Extensive experiments on the KITTI and nuScenes benchmarks demonstrate that our method consistently outperforms state-of-the-art approaches in both accuracy and robustness. Under large extrinsic perturbations, our approach achieves accurate calibration in 88% of KITTI cases and 99% of nuScenes cases, substantially surpassing the second-best baseline. We have open sourced our code on this https URL to benefit the community.

## Pipeline
![pipeline](assets/framework.jpg)

# Dependencies
|Pytorch|CUDA|Python|
|---|---|---|
|2.6.0|11.8|3.10.18|
# Build Packages
* Check TORCH_CUDA_ARCH_LIST:
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
```
The output can be `(8, 6)` to indicate 8.6.
* Set `TORCH_CUDA_ARCH_LIST` to simplify compilation: (for example, 8.6)
```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```
* Build csrc package for our method
```bash
cd models/tools/csrc/
python setup.py build_ext --inplace
```
* Build correlation_cuda package for LCCNet
```bash
cd models/lccnet/correlation_package/
python setup.py build_ext --inplace
```
* Install pointnet2_ops
```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
* Install GPU KNN
```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
* Third-party Libraries
```bash
pip install -r requirements.txt
```
<details>
  <summary>Troubleshooting</summary>
  The `correlation_cuda` package may be incompatible with CUDA >= 12.0. The failure of building this package only affects implementation of our baseline, LCCNet. If you have CUDA >= 12.0 and still want to implement LCCNET, it would be easy to use correlation pacakge in csrc to re-implement it. To try our best to reproduce LCCNet's performance, we utilize their own correlation package.
</details>

# Link KITTI Dataset to the root
* download KITTI dataset from [https://www.cvlibs.net/datasets/kitti/eval_odometry.php](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). (RGB, Veloydne and Calib data are all required)
* link the `dataset` filefolder as follows:
```bash
mkdir data
cd data
ln -s /path/to/kitti/ kitti
cd ..
```
# Link Nuscenes Dataset to the root
* download nuscenes dataset from [https://www.nuscenes.org/nuscenes#download](https://www.nuscenes.org/nuscenes#download). (v1.0-full, download keyframes of the trainval part and the test part)
* The files you download are:
```                  
v1.0-test_blobs.tgz            v1.0-trainval06_keyframes.tgz
v1.0-test_meta.tgz             v1.0-trainval07_keyframes.tgz
v1.0-trainval01_keyframes.tgz  v1.0-trainval08_keyframes.tgz
v1.0-trainval02_keyframes.tgz  v1.0-trainval09_keyframes.tgz
v1.0-trainval03_keyframes.tgz  v1.0-trainval10_keyframes.tgz
v1.0-trainval04_keyframes.tgz  v1.0-trainval_meta.tgz
v1.0-trainval05_keyframes.tgz
```
* unzip files naemd `v1-*.tgz` to the same directory as follows:
```bash
tar -xzvf v1.0-test_blobs.tgz  -C /path/to/nuscenes
tar -xzvf v1.0-test_meta.tgz   -C /path/to/nuscenes
...
```
After that, your dir-tree will be:
```
/path/to/nuscenes
- LICENSE
- maps
- samples
- sweeps
- v1.0-trainval
- v1.0-test
```
Finally, link your path `/path/to/nuscenes` to the data dir:

```bash
cd data
ln -s /path/to/nuscenes nuscenes
cd ..
```

# Expected output
After you set all the dataset, run `dataset.py`. The following content is expected to be printed in the terminal:
```
Ground truth poses are not avaialble for sequence 16.
Ground truth poses are not avaialble for sequence 17.
Ground truth poses are not avaialble for sequence 18.
dataset len:4023
img: torch.Size([3, 376, 1241])
pcd: torch.Size([3, 8192])
gt: torch.Size([4, 4])
extran: torch.Size([4, 4])
camera_info: {'fx': 718.856, 'fy': 718.856, 'cx': 607.1928, 'cy': 185.2157, 'sensor_h': 376, 'sensor_w': 1241, 'projection_mode': 'perspective'}
group_idx: ()
sub_idx: 0
======
Loading NuScenes tables for version v1.0-trainval...
23 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
4 map,
Done loading in 4.865 seconds.
======
Reverse indexing ...
Done reverse indexing in 2.2 seconds.
======
dataset len:30162
img: torch.Size([3, 900, 1600])
pcd: torch.Size([3, 8192])
camera_info: {'fx': 1266.417203046554, 'fy': 1266.417203046554, 'cx': 816.2670197447984, 'cy': 491.50706579294757, 'sensor_h': 900, 'sensor_w': 1600, 'projection_mode': 'perspective'}
extran: torch.Size([4, 4])
group_idx: 0
sub_idx: 0
```
Please note that the NuScenes dataset class has been optimized for faster loading.