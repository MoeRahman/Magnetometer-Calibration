# Magnetometer-Calibration

This repository allows developers to calibrate raw magnitometer sensor data into a unit sphere.
The calibrated data can then be used for orientation estimation for UAV or heading estimation fro ground vehicles.

### Raw Uncalibarated Data:

<img src=plots/xy.png width=33%/><img src=plots/xz.png width=33%/><img src=plots/yz.png width=33%/>

<i> Figure 2: Orthogonal projections & histogram of 3-D scatter plot </i>

### Calibration Process:
First we take the uncalibrated data points and calculate the mean of the data to find the centroid of a set of data points $ S $.

$S = {( (x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n) }$

The centroid is calculated by:

$(\bar{x}, \bar{y}) = \left( \frac{1}{n} \sum_{i=0}^{n} x_i, \frac{1}{n} \sum_{i=0}^{n} y_i \right)$

We can now subtract the mean from the dataset to recenter the points about the origin.

<img src=plots/raw_scatter_proj.png />

<i> Figure 3: Uncalibrated Orthogonal projections </i>

This resolves the issue of hard-iron bias and results in the following scatter plots:

<img src=plots/hard_iron_proj.png />

<i> Figure 4: Hard_Iron_Bias Orthogonal projections </i>

Using the method of least-squares we can calculate the optimal $ 3x3 $ affine transformation matrix that can transform the points into a set of points around the unit sphere centered about the origin.

<img src=plots/soft_iron_proj.png />

<i> Figure 5: Soft_Iron_Bias Orthogonal projections </i>

Both transformations can be viewed in the graphs below.

<img src=plots/cal_scatter.png width = 100%/>
