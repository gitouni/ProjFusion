## Calibration Results on KITTI
### Range: 15° / 15cm

<table>
  <thead>
    <tr>
      <th style="background:#000000; color:#FFFFFF;">Dataset</th>
      <th style="background:#000000; color:#FFFFFF;">Method</th>
      <th style="background:#000000; color:#FFFFFF;">RRMSE (°)</th>
      <th style="background:#000000; color:#FFFFFF;">RMAE (°)</th>
      <th style="background:#000000; color:#FFFFFF;">tRMSE (cm)</th>
      <th style="background:#000000; color:#FFFFFF;">tMAE (cm)</th>
      <th style="background:#000000; color:#FFFFFF;">L1 (%)</th>
      <th style="background:#000000; color:#FFFFFF;">L2 (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">4.61</td>
      <td style="background:#000000; color:#FFFFFF;">2.07</td>
      <td style="background:#000000; color:#FFFFFF;">135</td>
      <td style="background:#000000; color:#FFFFFF;">62.6</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">13.1</td>
      <td style="background:#000000; color:#FFFFFF;">6.31</td>
      <td style="background:#000000; color:#FFFFFF;">195</td>
      <td style="background:#000000; color:#FFFFFF;">98.4</td>
      <td style="background:#000000; color:#FFFFFF;">0.3%</td>
      <td style="background:#000000; color:#FFFFFF;">1.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">18.3</td>
      <td style="background:#000000; color:#FFFFFF;">9.44</td>
      <td style="background:#000000; color:#FFFFFF;">27.3</td>
      <td style="background:#000000; color:#FFFFFF;">13.8</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">1.9%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.02</td>
      <td style="background:#000000; color:#FFFFFF;">0.76</td>
      <td style="background:#000000; color:#FFFFFF;">5.80</td>
      <td style="background:#000000; color:#FFFFFF;">2.84</td>
      <td style="background:#000000; color:#FFFFFF;">8.0%</td>
      <td style="background:#000000; color:#FFFFFF;">32.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">3.88</td>
      <td style="background:#000000; color:#FFFFFF;">1.42</td>
      <td style="background:#000000; color:#FFFFFF;">6.07</td>
      <td style="background:#000000; color:#FFFFFF;">2.97</td>
      <td style="background:#000000; color:#FFFFFF;">5.4%</td>
      <td style="background:#000000; color:#FFFFFF;">18.6%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.10</td>
      <td style="background:#000000; color:#FFFFFF;">0.80</td>
      <td style="background:#000000; color:#FFFFFF;">6.12</td>
      <td style="background:#000000; color:#FFFFFF;">3.01</td>
      <td style="background:#000000; color:#FFFFFF;">9.2%</td>
      <td style="background:#000000; color:#FFFFFF;">31.7%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.53</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.21</b></td>
      <td style="background:#000000; color:#FFFFFF;">6.03</td>
      <td style="background:#000000; color:#FFFFFF;">2.90</td>
      <td style="background:#000000; color:#FFFFFF;">11.2%</td>
      <td style="background:#000000; color:#FFFFFF;">44.1%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;">1.06</td>
      <td style="background:#000000; color:#FFFFFF;">0.42</td>
      <td style="background:#000000; color:#FFFFFF;"><u>4.57</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>2.23</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>17.2%</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>56.9%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.43</b></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.21</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>2.20</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>1.09</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>54.6%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>96.6%</b></td>
    </tr>
  </tbody>
</table>

### Range: 10° / 25cm

<table>
  <thead>
    <tr>
      <th style="background:#000000; color:#FFFFFF;">Dataset</th>
      <th style="background:#000000; color:#FFFFFF;">Method</th>
      <th style="background:#000000; color:#FFFFFF;">RRMSE (°)</th>
      <th style="background:#000000; color:#FFFFFF;">RMAE (°)</th>
      <th style="background:#000000; color:#FFFFFF;">tRMSE (cm)</th>
      <th style="background:#000000; color:#FFFFFF;">tMAE (cm)</th>
      <th style="background:#000000; color:#FFFFFF;">L1 (%)</th>
      <th style="background:#000000; color:#FFFFFF;">L2 (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">2.94</td>
      <td style="background:#000000; color:#FFFFFF;">1.29</td>
      <td style="background:#000000; color:#FFFFFF;">60.7</td>
      <td style="background:#000000; color:#FFFFFF;">28.1</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.1%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">13.1</td>
      <td style="background:#000000; color:#FFFFFF;">6.46</td>
      <td style="background:#000000; color:#FFFFFF;">147</td>
      <td style="background:#000000; color:#FFFFFF;">69.9</td>
      <td style="background:#000000; color:#FFFFFF;">0.4%</td>
      <td style="background:#000000; color:#FFFFFF;">1.8%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">5.32</td>
      <td style="background:#000000; color:#FFFFFF;">2.60</td>
      <td style="background:#000000; color:#FFFFFF;">28.2</td>
      <td style="background:#000000; color:#FFFFFF;">14.2</td>
      <td style="background:#000000; color:#FFFFFF;">1.0%</td>
      <td style="background:#000000; color:#FFFFFF;">12.4%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.28</td>
      <td style="background:#000000; color:#FFFFFF;">0.89</td>
      <td style="background:#000000; color:#FFFFFF;">6.47</td>
      <td style="background:#000000; color:#FFFFFF;">3.15</td>
      <td style="background:#000000; color:#FFFFFF;">4.2%</td>
      <td style="background:#000000; color:#FFFFFF;">26.6%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">3.99</td>
      <td style="background:#000000; color:#FFFFFF;">1.52</td>
      <td style="background:#000000; color:#FFFFFF;">6.23</td>
      <td style="background:#000000; color:#FFFFFF;">3.05</td>
      <td style="background:#000000; color:#FFFFFF;">4.9%</td>
      <td style="background:#000000; color:#FFFFFF;">17.8%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.50</td>
      <td style="background:#000000; color:#FFFFFF;">0.96</td>
      <td style="background:#000000; color:#FFFFFF;">6.08</td>
      <td style="background:#000000; color:#FFFFFF;">2.98</td>
      <td style="background:#000000; color:#FFFFFF;">7.2%</td>
      <td style="background:#000000; color:#FFFFFF;">27.8%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.59</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.23</b></td>
      <td style="background:#000000; color:#FFFFFF;">6.27</td>
      <td style="background:#000000; color:#FFFFFF;">2.95</td>
      <td style="background:#000000; color:#FFFFFF;"><u>11.2%</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>42.4%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;">1.99</td>
      <td style="background:#000000; color:#FFFFFF;">0.74</td>
      <td style="background:#000000; color:#FFFFFF;"><u>5.44</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>2.60</u></td>
      <td style="background:#000000; color:#FFFFFF;">9.4%</td>
      <td style="background:#000000; color:#FFFFFF;">39.2%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.65</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.32</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>2.59</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>1.29</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>48.8%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>92.6%</b></td>
    </tr>
  </tbody>
</table>

### Range: 10° / 50cm

<table>
  <thead>
    <tr>
      <th style="background:#000000; color:#FFFFFF;">Dataset</th>
      <th style="background:#000000; color:#FFFFFF;">Method</th>
      <th style="background:#000000; color:#FFFFFF;">RRMSE (°)</th>
      <th style="background:#000000; color:#FFFFFF;">RMAE (°)</th>
      <th style="background:#000000; color:#FFFFFF;">tRMSE (cm)</th>
      <th style="background:#000000; color:#FFFFFF;">tMAE (cm)</th>
      <th style="background:#000000; color:#FFFFFF;">L1 (%)</th>
      <th style="background:#000000; color:#FFFFFF;">L2 (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">2.90</td>
      <td style="background:#000000; color:#FFFFFF;">1.26</td>
      <td style="background:#000000; color:#FFFFFF;">87.0</td>
      <td style="background:#000000; color:#FFFFFF;">38.3</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">12.7</td>
      <td style="background:#000000; color:#FFFFFF;">6.22</td>
      <td style="background:#000000; color:#FFFFFF;">223</td>
      <td style="background:#000000; color:#FFFFFF;">110</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.8%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">6.02</td>
      <td style="background:#000000; color:#FFFFFF;">2.90</td>
      <td style="background:#000000; color:#FFFFFF;">49.9</td>
      <td style="background:#000000; color:#FFFFFF;">24.9</td>
      <td style="background:#000000; color:#FFFFFF;">1.0%</td>
      <td style="background:#000000; color:#FFFFFF;">8.6%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.34</td>
      <td style="background:#000000; color:#FFFFFF;">0.92</td>
      <td style="background:#000000; color:#FFFFFF;">8.30</td>
      <td style="background:#000000; color:#FFFFFF;">4.03</td>
      <td style="background:#000000; color:#FFFFFF;">2.0%</td>
      <td style="background:#000000; color:#FFFFFF;">17.4%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">4.03</td>
      <td style="background:#000000; color:#FFFFFF;">1.57</td>
      <td style="background:#000000; color:#FFFFFF;">6.51</td>
      <td style="background:#000000; color:#FFFFFF;">3.18</td>
      <td style="background:#000000; color:#FFFFFF;">4.1%</td>
      <td style="background:#000000; color:#FFFFFF;">16.4%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.55</td>
      <td style="background:#000000; color:#FFFFFF;">0.99</td>
      <td style="background:#000000; color:#FFFFFF;">6.72</td>
      <td style="background:#000000; color:#FFFFFF;">3.29</td>
      <td style="background:#000000; color:#FFFFFF;">6.0%</td>
      <td style="background:#000000; color:#FFFFFF;">25.6%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.95</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.35</b></td>
      <td style="background:#000000; color:#FFFFFF;">6.48</td>
      <td style="background:#000000; color:#FFFFFF;">3.08</td>
      <td style="background:#000000; color:#FFFFFF;"><u>9.2%</u></td>
      <td style="background:#000000; color:#FFFFFF;">39.1%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;">1.77</td>
      <td style="background:#000000; color:#FFFFFF;">0.67</td>
      <td style="background:#000000; color:#FFFFFF;"><u>5.28</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>2.56</u></td>
      <td style="background:#000000; color:#FFFFFF;">8.7%</td>
      <td style="background:#000000; color:#FFFFFF;"><u>41.8%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">KITTI</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.76</b></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.37</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>2.75</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>1.36</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>41.0%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>87.7%</b></td>
    </tr>
  </tbody>
</table>
