## Calibration Results on nuScenes
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
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">5.09</td>
      <td style="background:#000000; color:#FFFFFF;">2.50</td>
      <td style="background:#000000; color:#FFFFFF;">179</td>
      <td style="background:#000000; color:#FFFFFF;">81.7</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">14.9</td>
      <td style="background:#000000; color:#FFFFFF;">7.18</td>
      <td style="background:#000000; color:#FFFFFF;">452</td>
      <td style="background:#000000; color:#FFFFFF;">212</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.2%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">7.51</td>
      <td style="background:#000000; color:#FFFFFF;">3.90</td>
      <td style="background:#000000; color:#FFFFFF;">7.24</td>
      <td style="background:#000000; color:#FFFFFF;">3.77</td>
      <td style="background:#000000; color:#FFFFFF;">0.5%</td>
      <td style="background:#000000; color:#FFFFFF;">3.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.12</td>
      <td style="background:#000000; color:#FFFFFF;">0.90</td>
      <td style="background:#000000; color:#FFFFFF;">6.34</td>
      <td style="background:#000000; color:#FFFFFF;">2.90</td>
      <td style="background:#000000; color:#FFFFFF;">8.2%</td>
      <td style="background:#000000; color:#FFFFFF;">35.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">4.20</td>
      <td style="background:#000000; color:#FFFFFF;">1.76</td>
      <td style="background:#000000; color:#FFFFFF;">6.06</td>
      <td style="background:#000000; color:#FFFFFF;">2.88</td>
      <td style="background:#000000; color:#FFFFFF;">4.6%</td>
      <td style="background:#000000; color:#FFFFFF;">17.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.34</td>
      <td style="background:#000000; color:#FFFFFF;">1.00</td>
      <td style="background:#000000; color:#FFFFFF;">5.59</td>
      <td style="background:#000000; color:#FFFFFF;">2.64</td>
      <td style="background:#000000; color:#FFFFFF;">13.7%</td>
      <td style="background:#000000; color:#FFFFFF;">41.9%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;">0.71</td>
      <td style="background:#000000; color:#FFFFFF;">0.28</td>
      <td style="background:#000000; color:#FFFFFF;">5.57</td>
      <td style="background:#000000; color:#FFFFFF;">2.29</td>
      <td style="background:#000000; color:#FFFFFF;">27.1%</td>
      <td style="background:#000000; color:#FFFFFF;">57.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.30</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.15</b></td>
      <td style="background:#000000; color:#FFFFFF;"><u>3.33</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>1.38</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>48.9%</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>79.1%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.37</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.19</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.51</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.25</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>97.9%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>99.9%</b></td>
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
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">3.84</td>
      <td style="background:#000000; color:#FFFFFF;">1.86</td>
      <td style="background:#000000; color:#FFFFFF;">105</td>
      <td style="background:#000000; color:#FFFFFF;">49.8</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">13.3</td>
      <td style="background:#000000; color:#FFFFFF;">6.22</td>
      <td style="background:#000000; color:#FFFFFF;">268</td>
      <td style="background:#000000; color:#FFFFFF;">123</td>
      <td style="background:#000000; color:#FFFFFF;">0.2%</td>
      <td style="background:#000000; color:#FFFFFF;">0.2%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">4.73</td>
      <td style="background:#000000; color:#FFFFFF;">2.48</td>
      <td style="background:#000000; color:#FFFFFF;">11.9</td>
      <td style="background:#000000; color:#FFFFFF;">6.25</td>
      <td style="background:#000000; color:#FFFFFF;">1.7%</td>
      <td style="background:#000000; color:#FFFFFF;">5.7%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.10</td>
      <td style="background:#000000; color:#FFFFFF;">0.89</td>
      <td style="background:#000000; color:#FFFFFF;">6.34</td>
      <td style="background:#000000; color:#FFFFFF;">2.90</td>
      <td style="background:#000000; color:#FFFFFF;">8.4%</td>
      <td style="background:#000000; color:#FFFFFF;">35.9%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">3.95</td>
      <td style="background:#000000; color:#FFFFFF;">1.56</td>
      <td style="background:#000000; color:#FFFFFF;">6.03</td>
      <td style="background:#000000; color:#FFFFFF;">2.85</td>
      <td style="background:#000000; color:#FFFFFF;">5.2%</td>
      <td style="background:#000000; color:#FFFFFF;">18.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.41</td>
      <td style="background:#000000; color:#FFFFFF;">1.04</td>
      <td style="background:#000000; color:#FFFFFF;">5.86</td>
      <td style="background:#000000; color:#FFFFFF;">2.78</td>
      <td style="background:#000000; color:#FFFFFF;">13.6%</td>
      <td style="background:#000000; color:#FFFFFF;">41.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;">0.64</td>
      <td style="background:#000000; color:#FFFFFF;">0.25</td>
      <td style="background:#000000; color:#FFFFFF;">5.63</td>
      <td style="background:#000000; color:#FFFFFF;">2.31</td>
      <td style="background:#000000; color:#FFFFFF;">24.8%</td>
      <td style="background:#000000; color:#FFFFFF;">55.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.39</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.20</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>3.78</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>1.58</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>41.3%</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>74.1%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.39</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.19</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.53</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.26</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>97.2%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>99.7%</b></td>
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
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CoFiI2P</td>
      <td style="background:#000000; color:#FFFFFF;">3.74</td>
      <td style="background:#000000; color:#FFFFFF;">1.77</td>
      <td style="background:#000000; color:#FFFFFF;">54.2</td>
      <td style="background:#000000; color:#FFFFFF;">26.3</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">DirectCalib</td>
      <td style="background:#000000; color:#FFFFFF;">12.7</td>
      <td style="background:#000000; color:#FFFFFF;">6.21</td>
      <td style="background:#000000; color:#FFFFFF;">359</td>
      <td style="background:#000000; color:#FFFFFF;">169</td>
      <td style="background:#000000; color:#FFFFFF;">0.0%</td>
      <td style="background:#000000; color:#FFFFFF;">0.3%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibAnything</td>
      <td style="background:#000000; color:#FFFFFF;">4.73</td>
      <td style="background:#000000; color:#FFFFFF;">2.48</td>
      <td style="background:#000000; color:#FFFFFF;">23.9</td>
      <td style="background:#000000; color:#FFFFFF;">12.5</td>
      <td style="background:#000000; color:#FFFFFF;">0.5%</td>
      <td style="background:#000000; color:#FFFFFF;">3.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.47</td>
      <td style="background:#000000; color:#FFFFFF;">1.02</td>
      <td style="background:#000000; color:#FFFFFF;">8.78</td>
      <td style="background:#000000; color:#FFFFFF;">3.99</td>
      <td style="background:#000000; color:#FFFFFF;">4.0%</td>
      <td style="background:#000000; color:#FFFFFF;">20.9%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">RGGNet</td>
      <td style="background:#000000; color:#FFFFFF;">5.83</td>
      <td style="background:#000000; color:#FFFFFF;">2.71</td>
      <td style="background:#000000; color:#FFFFFF;">7.25</td>
      <td style="background:#000000; color:#FFFFFF;">3.77</td>
      <td style="background:#000000; color:#FFFFFF;">2.3%</td>
      <td style="background:#000000; color:#FFFFFF;">8.5%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">LCCNet</td>
      <td style="background:#000000; color:#FFFFFF;">2.83</td>
      <td style="background:#000000; color:#FFFFFF;">1.21</td>
      <td style="background:#000000; color:#FFFFFF;">7.56</td>
      <td style="background:#000000; color:#FFFFFF;">3.60</td>
      <td style="background:#000000; color:#FFFFFF;">7.0%</td>
      <td style="background:#000000; color:#FFFFFF;">29.4%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">LCCRAFT</td>
      <td style="background:#000000; color:#FFFFFF;">0.94</td>
      <td style="background:#000000; color:#FFFFFF;">0.37</td>
      <td style="background:#000000; color:#FFFFFF;">7.41</td>
      <td style="background:#000000; color:#FFFFFF;">3.20</td>
      <td style="background:#000000; color:#FFFFFF;">13.7%</td>
      <td style="background:#000000; color:#FFFFFF;">40.1%</td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">CalibDepth</td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.36</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.18</b></td>
      <td style="background:#000000; color:#FFFFFF;"><u>5.71</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>2.22</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>27.8%</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>54.4%</u></td>
    </tr>
    <tr>
      <td style="background:#000000; color:#FFFFFF;">nuScenes</td>
      <td style="background:#000000; color:#FFFFFF;">Ours</td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.60</u></td>
      <td style="background:#000000; color:#FFFFFF;"><u>0.30</u></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.77</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>0.38</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>89.8%</b></td>
      <td style="background:#000000; color:#FFFFFF;"><b>99.2%</b></td>
    </tr>
  </tbody>
</table>
