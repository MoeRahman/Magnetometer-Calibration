# Magnetometer-Calibration

This repository allows developers to calibrate raw magnitometer sensor data into a unit sphere.
The calibrated data can then be used for orientation estimation for UAV or heading estimation fro ground vehicles.

### Raw Uncalibarated Data:

<img src="plots/raw_data_plot.png"/>

<i> Figure 1: 3-D scatter plot of uncalibrated sensor data </i>

<img src=plots/xy.png width=32%/> <img src=plots/xz.png width=32%/> <img src=plots/yz.png width=32%/>

<i> Figure 2: Orthogonal projections & histogram of 3-D scatter plot </i>

### Calibration Process:

<img src=plots/raw_scatter_proj.png />

<i> Figure 3: Uncalibrated Orthogonal projections </i>

<img src=plots/hard_iron_proj.png />

<i> Figure 4: Hard_Iron_Bias Orthogonal projections </i>

<img src=plots/soft_iron_proj.png />

<i> Figure 5: Soft_Iron_Bias Orthogonal projections </i>

<img src=plots/cal_scatter.png />
