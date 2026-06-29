from magnetometer_calibration.generator import GenerateSensorData
from magnetometer_calibration.calibration import CalibrateData

import matplotlib.pyplot as plt
import numpy as np

def plot_3d_scatter(coordinates):
    X, Y, Z = coordinates

    #Orthogonal Projection & Histogram & 3-D Scatter Plots
    fig, axs = plt.subplots(2, 2, figsize=(6,6), layout='constrained')

    fig.suptitle("Uncalibrated Data")

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

    plt.show()

def main():
    NUMBER_OF_POINTS = 500

    magnetometer_sensor_data = GenerateSensorData(NUMBER_OF_POINTS)
    coordinates = magnetometer_sensor_data.generate_points()

    plot_3d_scatter(coordinates)

    calibrate = CalibrateData(coordinates)
    calibrated_coordinates = calibrate.ellipsoidal_fit()[:3]

    plot_3d_scatter(calibrated_coordinates)

    return

if __name__ == "__main__":
    main()