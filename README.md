# Official Implementation of Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations (RA-L 2026)
## Paper Link
[Arxiv](https://arxiv.org/abs/2603.29414) | IEEE

## Abstract
Accurate camera-LiDAR fusion relies on precise extrinsic calibration, which fundamentally depends on establishing reliable cross-modal correspondences under potentially large misalignments. Existing learning-based methods typically project LiDAR points into depth maps for feature fusion, which distorts 3D geometry and degrades performance when the extrinsic initialization is far from the ground truth. To address this issue, we propose an extrinsic-aware cross-attention framework that directly aligns image patches and LiDAR point groups in their native domains. The proposed attention mechanism explicitly injects extrinsic parameter hypotheses into the correspondence modeling process, enabling geometry-consistent cross-modal interaction without relying on projected 2D depth maps. Extensive experiments on the KITTI and nuScenes benchmarks demonstrate that our method consistently outperforms state-of-the-art approaches in both accuracy and robustness. Under large extrinsic perturbations, our approach achieves accurate calibration in 88% of KITTI cases and 99% of nuScenes cases, substantially surpassing the second-best baseline. We have open sourced our code on this https URL to benefit the community.

## Installation
Guidlines awaits to be added.