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
