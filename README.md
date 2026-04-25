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

# Calibration Results on KITTI

- Best: bold
- Second: underline

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Range</th>
      <th>Method</th>
      <th>RRMSE (°)</th>
      <th>RMAE (°)</th>
      <th>tRMSE (cm)</th>
      <th>tMAE (cm)</th>
      <th>L1 (%)</th>
      <th>L2 (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">4.61 +- 3.07</td>
      <td style="background:#000000; color:#FFFFFF;">2.07 +- 1.23</td>
      <td style="background:#000000; color:#FFFFFF;">135 +- 75.1</td>
      <td style="background:#000000; color:#FFFFFF;">62.6 +- 32.6</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">13.1 +- 23.6</td>
      <td style="background:#000000; color:#FFFFFF;">6.31 +- 11.4</td>
      <td style="background:#000000; color:#FFFFFF;">195 +- 1967</td>
      <td style="background:#000000; color:#FFFFFF;">98.4 +- 1099</td>
      <td style="background:#000000; color:#FFFFFF;">0.3%</td>
      <td style="background:#000000; color:#FFFFFF;">1.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">18.3 +- 14.8</td>
      <td style="background:#000000; color:#FFFFFF;">9.44 +- 7.89</td>
      <td style="background:#000000; color:#FFFFFF;">27.3 +- 15.1</td>
      <td style="background:#000000; color:#FFFFFF;">13.8 +- 7.74</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">1.9%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.02 +- 2.10</td>
      <td style="background:#000000; color:#FFFFFF;">0.76 +- 0.73</td>
      <td style="background:#000000; color:#FFFFFF;">5.80 +- 3.60</td>
      <td style="background:#000000; color:#FFFFFF;">2.84 +- 1.78</td>
      <td style="background:#000000; color:#FFFFFF;">8.0%</td>
      <td style="background:#000000; color:#FFFFFF;">32.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">3.88 +- 3.38</td>
      <td style="background:#000000; color:#FFFFFF;">1.42 +- 1.20</td>
      <td style="background:#000000; color:#FFFFFF;">6.07 +- 4.04</td>
      <td style="background:#000000; color:#FFFFFF;">2.97 +- 2.02</td>
      <td style="background:#000000; color:#FFFFFF;">5.4%</td>
      <td style="background:#000000; color:#FFFFFF;">18.6%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.10 +- 2.21</td>
      <td style="background:#000000; color:#FFFFFF;">0.80 +- 0.78</td>
      <td style="background:#000000; color:#FFFFFF;">6.12 +- 4.08</td>
      <td style="background:#000000; color:#FFFFFF;">3.01 +- 2.05</td>
      <td style="background:#000000; color:#FFFFFF;">9.2%</td>
      <td style="background:#000000; color:#FFFFFF;">31.7%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.53 +- 0.78</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.21 +- 0.27</b></td>
      <td style="background:#000000; color:#FFFFFF;">6.03 +- 3.55</td>
      <td style="background:#000000; color:#FFFFFF;">2.90 +- 1.70</td>
      <td style="background:#000000; color:#FFFFFF;">11.2%</td>
      <td style="background:#000000; color:#FFFFFF;">44.1%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;">1.06 +- 1.20</td>
      <td style="background:#000000; color:#FFFFFF;">0.42 +- 0.41</td>
      <td style="background:#000000; color:#FFFFFF;"><u>4.57 +- 2.80</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>2.23 +- 1.34</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>17.2%</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>56.9%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.43 +- 1.04</b></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.21 +- 0.50</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>2.20 +- 1.82</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>1.09 +- 0.90</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>54.6%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>96.6%</b></td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.94 +- 2.13</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.29 +- 0.84</td>
      <td style="background:#3A3939; color:#FFFFFF;">60.7 +- 33.3</td>
      <td style="background:#3A3939; color:#FFFFFF;">28.1 +- 14.6</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.0%</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.1%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#3A3939; color:#FFFFFF;">13.1 +- 26.3</td>
      <td style="background:#3A3939; color:#FFFFFF;">6.46 +- 13.2</td>
      <td style="background:#3A3939; color:#FFFFFF;">147 +- 401</td>
      <td style="background:#3A3939; color:#FFFFFF;">69.9 +- 191</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.4%</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.8%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#3A3939; color:#FFFFFF;">5.32 +- 8.95</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.60 +- 4.53</td>
      <td style="background:#3A3939; color:#FFFFFF;">28.2 +- 25.5</td>
      <td style="background:#3A3939; color:#FFFFFF;">14.2 +- 13.1</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.0%</td>
      <td style="background:#3A3939; color:#FFFFFF;">12.4%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">CalibNet</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.28 +- 2.38</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.89 +- 0.91</td>
      <td style="background:#3A3939; color:#FFFFFF;">6.47 +- 3.75</td>
      <td style="background:#3A3939; color:#FFFFFF;">3.15 +- 1.84</td>
      <td style="background:#3A3939; color:#FFFFFF;">4.2%</td>
      <td style="background:#3A3939; color:#FFFFFF;">26.6%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">RGGNet</td>
      <td style="background:#3A3939; color:#FFFFFF;">3.99 +- 3.49</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.52 +- 1.39</td>
      <td style="background:#3A3939; color:#FFFFFF;">6.23 +- 4.09</td>
      <td style="background:#3A3939; color:#FFFFFF;">3.05 +- 2.03</td>
      <td style="background:#3A3939; color:#FFFFFF;">4.9%</td>
      <td style="background:#3A3939; color:#FFFFFF;">17.8%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">LCCNet</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.50 +- 2.53</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.96 +- 0.95</td>
      <td style="background:#3A3939; color:#FFFFFF;">6.08 +- 3.87</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.98 +- 1.93</td>
      <td style="background:#3A3939; color:#FFFFFF;">7.2%</td>
      <td style="background:#3A3939; color:#FFFFFF;">27.8%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>0.59 +- 0.78</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>0.23 +- 0.27</b></td>
      <td style="background:#3A3939; color:#FFFFFF;">6.27 +- 3.91</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.95 +- 1.79</td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>11.2%</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>42.4%</u></td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.99 +- 2.48</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.74 +- 0.89</td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>5.44 +- 3.39</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>2.60 +- 1.62</u></td>
      <td style="background:#3A3939; color:#FFFFFF;">9.4%</td>
      <td style="background:#3A3939; color:#FFFFFF;">39.2%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">KITTI</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">Ours</td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>0.65 +- 1.43</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>0.32 +- 0.65</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>2.59 +- 1.75</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>1.29 +- 0.89</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>48.8%</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>92.6%</b></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">2.90 +- 2.18</td>
      <td style="background:#000000; color:#FFFFFF;">1.26 +- 0.86</td>
      <td style="background:#000000; color:#FFFFFF;">87.0 +- 38.0</td>
      <td style="background:#000000; color:#FFFFFF;">38.3 +- 15.8</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">12.7 +- 24.1</td>
      <td style="background:#000000; color:#FFFFFF;">6.22 +- 11.7</td>
      <td style="background:#000000; color:#FFFFFF;">223 +- 1394</td>
      <td style="background:#000000; color:#FFFFFF;">110 +- 729</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.8%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">6.02 +- 9.58</td>
      <td style="background:#000000; color:#FFFFFF;">2.90 +- 4.74</td>
      <td style="background:#000000; color:#FFFFFF;">49.9 +- 48.2</td>
      <td style="background:#000000; color:#FFFFFF;">24.9 +- 24.4</td>
      <td style="background:#000000; color:#FFFFFF;">1.0%</td>
      <td style="background:#000000; color:#FFFFFF;">8.6%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.34 +- 2.39</td>
      <td style="background:#000000; color:#FFFFFF;">0.92 +- 0.91</td>
      <td style="background:#000000; color:#FFFFFF;">8.30 +- 4.91</td>
      <td style="background:#000000; color:#FFFFFF;">4.03 +- 2.39</td>
      <td style="background:#000000; color:#FFFFFF;">2.0%</td>
      <td style="background:#000000; color:#FFFFFF;">17.4%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">4.03 +- 3.53</td>
      <td style="background:#000000; color:#FFFFFF;">1.57 +- 1.44</td>
      <td style="background:#000000; color:#FFFFFF;">6.51 +- 4.07</td>
      <td style="background:#000000; color:#FFFFFF;">3.18 +- 2.02</td>
      <td style="background:#000000; color:#FFFFFF;">4.1%</td>
      <td style="background:#000000; color:#FFFFFF;">16.4%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.55 +- 2.55</td>
      <td style="background:#000000; color:#FFFFFF;">0.99 +- 0.96</td>
      <td style="background:#000000; color:#FFFFFF;">6.72 +- 4.55</td>
      <td style="background:#000000; color:#FFFFFF;">3.29 +- 2.25</td>
      <td style="background:#000000; color:#FFFFFF;">6.0%</td>
      <td style="background:#000000; color:#FFFFFF;">25.6%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.95 +- 1.12</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.35 +- 0.39</b></td>
      <td style="background:#000000; color:#FFFFFF;">6.48 +- 4.20</td>
      <td style="background:#000000; color:#FFFFFF;">3.08 +- 2.07</td>
      <td style="background:#000000; color:#FFFFFF;"><u>9.2%</u></td>
      <td style="background:#000000; color:#FFFFFF;">39.1%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;">1.77 +- 2.14</td>
      <td style="background:#000000; color:#FFFFFF;">0.67 +- 0.74</td>
      <td style="background:#000000; color:#FFFFFF;"><u>5.28 +- 3.20</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>2.56 +- 1.52</u></td>
      <td style="background:#000000; color:#FFFFFF;">8.7%</td>
      <td style="background:#000000; color:#FFFFFF;"><u>41.8%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.76 +- 0.91</b></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.37 +- 0.44</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>2.75 +- 1.43</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>1.36 +- 0.71</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>41.0%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>87.7%</b></td>
    </tr>
  </tbody>
</table>


# Calibration Results on nuScenes

- Best: bold
- Second: underline

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Range</th>
      <th>Method</th>
      <th>RRMSE (°)</th>
      <th>RMAE (°)</th>
      <th>tRMSE (cm)</th>
      <th>tMAE (cm)</th>
      <th>L1 (%)</th>
      <th>L2 (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">5.09 +- 4.31</td>
      <td style="background:#000000; color:#FFFFFF;">2.50 +- 1.96</td>
      <td style="background:#000000; color:#FFFFFF;">179 +- 97.3</td>
      <td style="background:#000000; color:#FFFFFF;">81.7 +- 46.6</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">14.9 +- 22.0</td>
      <td style="background:#000000; color:#FFFFFF;">7.18 +- 10.3</td>
      <td style="background:#000000; color:#FFFFFF;">452 +- 1633</td>
      <td style="background:#000000; color:#FFFFFF;">212 +- 774</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.2%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">7.51 +- 4.56</td>
      <td style="background:#000000; color:#FFFFFF;">3.90 +- 2.48</td>
      <td style="background:#000000; color:#FFFFFF;">7.24 +- 4.98</td>
      <td style="background:#000000; color:#FFFFFF;">3.77 +- 2.71</td>
      <td style="background:#000000; color:#FFFFFF;">0.5%</td>
      <td style="background:#000000; color:#FFFFFF;">3.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.12 +- 2.00</td>
      <td style="background:#000000; color:#FFFFFF;">0.90 +- 0.84</td>
      <td style="background:#000000; color:#FFFFFF;">6.34 +- 3.83</td>
      <td style="background:#000000; color:#FFFFFF;">2.90 +- 1.72</td>
      <td style="background:#000000; color:#FFFFFF;">8.2%</td>
      <td style="background:#000000; color:#FFFFFF;">35.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">4.20 +- 3.50</td>
      <td style="background:#000000; color:#FFFFFF;">1.76 +- 1.50</td>
      <td style="background:#000000; color:#FFFFFF;">6.06 +- 4.09</td>
      <td style="background:#000000; color:#FFFFFF;">2.88 +- 1.95</td>
      <td style="background:#000000; color:#FFFFFF;">4.6%</td>
      <td style="background:#000000; color:#FFFFFF;">17.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.34 +- 2.47</td>
      <td style="background:#000000; color:#FFFFFF;">1.00 +- 1.10</td>
      <td style="background:#000000; color:#FFFFFF;">5.59 +- 4.49</td>
      <td style="background:#000000; color:#FFFFFF;">2.64 +- 2.21</td>
      <td style="background:#000000; color:#FFFFFF;">13.7%</td>
      <td style="background:#000000; color:#FFFFFF;">41.9%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;">0.71 +- 1.65</td>
      <td style="background:#000000; color:#FFFFFF;">0.28 +- 0.57</td>
      <td style="background:#000000; color:#FFFFFF;">5.57 +- 4.48</td>
      <td style="background:#000000; color:#FFFFFF;">2.29 +- 1.70</td>
      <td style="background:#000000; color:#FFFFFF;">27.1%</td>
      <td style="background:#000000; color:#FFFFFF;">57.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.30 +- 0.20</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.15 +- 0.10</b></td>
      <td style="background:#000000; color:#FFFFFF;"><u>3.33 +- 2.64</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>1.38 +- 0.96</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>48.9%</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>79.1%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">15° / 15cm</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.37 +- 0.23</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.19 +- 0.12</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.51 +- 0.28</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.25 +- 0.13</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>97.9%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>99.9%</b></td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#3A3939; color:#FFFFFF;">3.84 +- 2.15</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.86 +- 1.03</td>
      <td style="background:#3A3939; color:#FFFFFF;">105 +- 79.4</td>
      <td style="background:#3A3939; color:#FFFFFF;">49.8 +- 37.5</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.0%</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#3A3939; color:#FFFFFF;">13.3 +- 22.3</td>
      <td style="background:#3A3939; color:#FFFFFF;">6.22 +- 10.3</td>
      <td style="background:#3A3939; color:#FFFFFF;">268 +- 1005</td>
      <td style="background:#3A3939; color:#FFFFFF;">123 +- 463</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.2%</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.2%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#3A3939; color:#FFFFFF;">4.73 +- 3.17</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.48 +- 1.72</td>
      <td style="background:#3A3939; color:#FFFFFF;">11.9 +- 8.32</td>
      <td style="background:#3A3939; color:#FFFFFF;">6.25 +- 4.50</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.7%</td>
      <td style="background:#3A3939; color:#FFFFFF;">5.7%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">CalibNet</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.10 +- 1.98</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.89 +- 0.84</td>
      <td style="background:#3A3939; color:#FFFFFF;">6.34 +- 3.84</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.90 +- 1.72</td>
      <td style="background:#3A3939; color:#FFFFFF;">8.4%</td>
      <td style="background:#3A3939; color:#FFFFFF;">35.9%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">RGGNet</td>
      <td style="background:#3A3939; color:#FFFFFF;">3.95 +- 3.33</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.56 +- 1.30</td>
      <td style="background:#3A3939; color:#FFFFFF;">6.03 +- 4.09</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.85 +- 1.94</td>
      <td style="background:#3A3939; color:#FFFFFF;">5.2%</td>
      <td style="background:#3A3939; color:#FFFFFF;">18.3%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">LCCNet</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.41 +- 2.56</td>
      <td style="background:#3A3939; color:#FFFFFF;">1.04 +- 1.14</td>
      <td style="background:#3A3939; color:#FFFFFF;">5.86 +- 5.72</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.78 +- 2.87</td>
      <td style="background:#3A3939; color:#FFFFFF;">13.6%</td>
      <td style="background:#3A3939; color:#FFFFFF;">41.5%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.64 +- 1.21</td>
      <td style="background:#3A3939; color:#FFFFFF;">0.25 +- 0.42</td>
      <td style="background:#3A3939; color:#FFFFFF;">5.63 +- 4.28</td>
      <td style="background:#3A3939; color:#FFFFFF;">2.31 +- 1.60</td>
      <td style="background:#3A3939; color:#FFFFFF;">24.8%</td>
      <td style="background:#3A3939; color:#FFFFFF;">55.0%</td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>0.39 +- 0.25</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>0.20 +- 0.13</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>3.78 +- 2.84</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>1.58 +- 1.04</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>41.3%</u></td>
      <td style="background:#3A3939; color:#FFFFFF;"><u>74.1%</u></td>
    </tr>
    <tr>
      <td style="background:#3A3939; color:#FFFFFF;">nuScenes</td>
      <td style="background:#3A3939; color:#FFFFFF;">10° / 25cm</td>
      <td style="background:#3A3939; color:#FFFFFF;">Ours</td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>0.39 +- 0.28</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>0.19 +- 0.15</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>0.53 +- 0.33</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>0.26 +- 0.16</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>97.2%</b></td>
      <td style="background:#3A3939; color:#FFFFFF;"><b>99.7%</b></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">3.74 +- 2.33</td>
      <td style="background:#000000; color:#FFFFFF;">1.77 +- 1.05</td>
      <td style="background:#000000; color:#FFFFFF;">54.2 +- 33.9</td>
      <td style="background:#000000; color:#FFFFFF;">26.3 +- 16.9</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">12.7 +- 22.0</td>
      <td style="background:#000000; color:#FFFFFF;">6.21 +- 10.7</td>
      <td style="background:#000000; color:#FFFFFF;">359 +- 1156</td>
      <td style="background:#000000; color:#FFFFFF;">169 +- 545</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">4.73 +- 3.17</td>
      <td style="background:#000000; color:#FFFFFF;">2.48 +- 1.72</td>
      <td style="background:#000000; color:#FFFFFF;">23.9 +- 16.6</td>
      <td style="background:#000000; color:#FFFFFF;">12.5 +- 9.01</td>
      <td style="background:#000000; color:#FFFFFF;">0.5%</td>
      <td style="background:#000000; color:#FFFFFF;">3.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.47 +- 2.28</td>
      <td style="background:#000000; color:#FFFFFF;">1.02 +- 0.91</td>
      <td style="background:#000000; color:#FFFFFF;">8.78 +- 5.67</td>
      <td style="background:#000000; color:#FFFFFF;">3.99 +- 2.56</td>
      <td style="background:#000000; color:#FFFFFF;">4.0%</td>
      <td style="background:#000000; color:#FFFFFF;">20.9%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">5.83 +- 4.18</td>
      <td style="background:#000000; color:#FFFFFF;">2.71 +- 2.03</td>
      <td style="background:#000000; color:#FFFFFF;">7.25 +- 4.83</td>
      <td style="background:#000000; color:#FFFFFF;">3.77 +- 2.62</td>
      <td style="background:#000000; color:#FFFFFF;">2.3%</td>
      <td style="background:#000000; color:#FFFFFF;">8.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.83 +- 3.15</td>
      <td style="background:#000000; color:#FFFFFF;">1.21 +- 1.47</td>
      <td style="background:#000000; color:#FFFFFF;">7.56 +- 12.1</td>
      <td style="background:#000000; color:#FFFFFF;">3.60 +- 6.16</td>
      <td style="background:#000000; color:#FFFFFF;">7.0%</td>
      <td style="background:#000000; color:#FFFFFF;">29.4%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;">0.94 +- 1.54</td>
      <td style="background:#000000; color:#FFFFFF;">0.37 +- 0.53</td>
      <td style="background:#000000; color:#FFFFFF;">7.41 +- 5.33</td>
      <td style="background:#000000; color:#FFFFFF;">3.20 +- 2.22</td>
      <td style="background:#000000; color:#FFFFFF;">13.7%</td>
      <td style="background:#000000; color:#FFFFFF;">40.1%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.36 +- 0.24</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.18 +- 0.12</b></td>
      <td style="background:#000000; color:#FFFFFF;"><u>5.71 +- 4.59</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>2.22 +- 1.61</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>27.8%</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>54.4%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">10° / 50cm</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.60 +- 0.36</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.30 +- 0.19</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.77 +- 0.46</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.38 +- 0.22</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>89.8%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>99.2%</b></td>
    </tr>
  </tbody>
</table>




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

# Train Calibration Models
To train our method on KITTI or nuScenes dataset with different settings, please copy related commands and run them in the terminal. The training logs and checkpoints will be saved in `./experiments/{BASE_DIR}/{TASK_NAME}/`:
```bash
# KITTI Odometry

# 15° / 15cm
PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti_r15_t0.15.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_r15_t0.15'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 
# 10° / 25cm
PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti_r10_t0.25.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_r10_t0.25'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}
# 10° / 50cm
PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/kitti_r10_t0.5.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='kitti'
TASK_NAME='projdualfusion_harmonic_r10_t0.5'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}

# nuScenes
# 15° / 15cm
PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/nusc_r15_t0.15.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='nusc'
TASK_NAME='projdualfusion_harmonic_r15_t0.15'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}
# 10° / 25cm
PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/nusc_r10_t0.25.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='nusc'
TASK_NAME='projdualfusion_harmonic_r10_t0.25'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 
# 10° / 50cm
PYTHON_PATH="python"
SCRIPT_PATH="train.py"
DATASET_CONFIG="cfg/dataset/nusc_r10_t0.5.yml"
MODEL_CONFIG="cfg/model/projdualfusion_harmonic.yml"
BASE_DIR='nusc'
TASK_NAME='projdualfusion_harmonic_r10_t0.5'
echo "${PYTHON_PATH} ${SCRIPT_PATH} '--dataset_config' ${DATASET_CONFIG} '--model_config' ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME}"
${PYTHON_PATH} ${SCRIPT_PATH} "--dataset_config" ${DATASET_CONFIG} "--model_config" ${MODEL_CONFIG} '--base_dir' ${BASE_DIR} '--task_name' ${TASK_NAME} 
```