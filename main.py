from magnetometer_calibration.generator import GenerateSensorData
from magnetometer_calibration.calibration import CalibrateData

import matplotlib.pyplot as plt
import numpy as np



def plot_3d_scatter(title, uncalibrated, centered, scaled):
    uncalibrated_x, uncalibrated_y, uncalibrated_z = uncalibrated
    centered_x, centered_y, centered_z = centered
    scaled_x, scaled_y, scaled_z = scaled

    #Orthogonal Projection & 3-D Scatter Plots
    fig = plt.figure(figsize=(6,6), layout='constrained')
    fig.suptitle(title)

    ax = fig.add_subplot(projection='3d')
    ax.scatter(uncalibrated_x, uncalibrated_y, uncalibrated_z, s = 2, c = "red", alpha = 1, label = "uncalibrated")
    ax.scatter(centered_x, centered_y, centered_z, s = 2, c = "blue", alpha = 1, label = "centered")
    ax.scatter(scaled_x, scaled_y, scaled_z, s = 25, c = "green", alpha = 0.75, label = "scaled")

    axis = np.linspace(0, 1, 100)
    ax.plot(axis, ys = 0, zs = 0, c = "red") #x-axis
    ax.plot(0, axis, zs = 0, c = "blue")     #y-axis
    ax.plot(0, 0, axis, c = "green")         #z-axis

    plots = (uncalibrated, centered, scaled)
    max_x, max_y, max_z = (0, 0, 0)
    min_x, min_y, min_z = (9999, 9999, 9999)

    max_axis, min_axis = (0, 0)

    for plot in plots:
        mx, my, mz = plot.max(axis = 1)
        max_x, max_y, max_z = (max(max_x, mx), max(max_y, my), max(max_z, mz))
        max_axis = max(max_x, max_y, max_z)

        xm, ym, zm = plot.min(axis = 1)
        min_x, min_y, min_z = (min(min_x, xm), min(min_y, ym), min(min_z, zm))
        min_axis = min(min_x, min_y, min_z)

    ax.set_xlim(min_axis, max_axis)
    ax.set_ylim(min_axis, max_axis)
    ax.set_zlim(min_axis, max_axis)
    fig.legend()

    return

def plot_orthogonal_histogram(title, uncalibrated, centered, scaled):

    return


def main():
    NUMBER_OF_POINTS = 250

    magnetometer_data = GenerateSensorData(NUMBER_OF_POINTS)
    uncalibrated_data = magnetometer_data.generate_uniform_points()
    uncentered_data = CalibrateData(uncalibrated_data)

    hard_iron_bias = uncentered_data.hard_iron_bias()
    centered_data = uncentered_data.sensor_data - hard_iron_bias.reshape(3,1)

    soft_iron_bias = CalibrateData(centered_data).soft_iron_bias()
    homogeneous_points = np.vstack((centered_data, np.ones((1, centered_data.shape[1]))))
    calibrated_data = (homogeneous_points.transpose()@soft_iron_bias).transpose()[:3]

    plot_3d_scatter("Data Calibration", uncalibrated_data, centered_data, calibrated_data)

    plt.show()
    return


if __name__ == "__main__":
    main()