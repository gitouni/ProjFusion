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
