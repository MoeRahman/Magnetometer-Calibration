from magnetometer_calibration.generator import GenerateSensorData
from magnetometer_calibration.calibration import CalibrateData

import matplotlib.pyplot as plt
import numpy as np


def plot_3d_scatter(title, coordinates):
    X, Y, Z = coordinates

    #Orthogonal Projection & Histogram & 3-D Scatter Plots
    fig, axs = plt.subplots(2, 2, figsize=(6,6), layout='constrained')

    fig.suptitle(title)

    Axis = [
        (axs[0,0], X, Y),
        (axs[0,1], X, Z),
        (axs[1,0], Y, Z)
    ]

    for ax, data_x, data_y in Axis:
        ax.scatter(data_x, data_y, s = 1, c = "black")
        ax.grid()

    axs[1,1] = fig.add_subplot(2, 2, 4, projection='3d')
    axs[1,1].scatter(X, Y, Z, s = 1, c = "red")
    return


def main():
    NUMBER_OF_POINTS = 500

    magnetometer_sensor_data = GenerateSensorData(NUMBER_OF_POINTS)
    coordinates = magnetometer_sensor_data.generate_points()

    plot_3d_scatter("Uncalibrated Data", coordinates)

    calibrate = CalibrateData(coordinates)

    hard_iron_bias_estimate = calibrate.hard_iron_bias()
    calibrate.sensor_data -= hard_iron_bias_estimate.reshape(3,1)
    plot_3d_scatter("Recentered Data", calibrate.sensor_data[:3])

    soft_iron_bias_estimate = calibrate.soft_iron_bias()
    homogeneous_points = np.vstack((calibrate.sensor_data, np.ones((1, calibrate.sensor_data.shape[1]))))
    calibrated_data = homogeneous_points.transpose()@soft_iron_bias_estimate

    plot_3d_scatter("Calibrated Data", calibrated_data.transpose()[:3])

    plt.show()

    return


if __name__ == "__main__":
    main()