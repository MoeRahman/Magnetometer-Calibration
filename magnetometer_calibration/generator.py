import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

NUMBER_OF_POINTS = 500

class GenerateSensorData():
    def __init__(self, number_of_points) -> None:
        self.number_of_points = number_of_points

    def generate_points(self) -> np.ndarray:

        phi = np.random.uniform(0, np.pi*2, self.number_of_points)
        theta = np.random.uniform(0, np.pi*2, self.number_of_points)

        mag_x = np.sin( theta ) * np.cos( phi )
        mag_y = np.sin( theta ) * np.sin( phi )
        mag_z = np.cos( theta )

        true_data = np.array([mag_x, mag_y, mag_z])

        soft_iron_bias = np.array([[1.0, 0.1, 0.0], [0.0, 10.0, 0.0], [0.2, 0.0, 1.0]])
        hard_iron_bias = np.random.uniform(0, 5, (3,1))

        noise = np.random.normal(0, 0.1, (3, self.number_of_points))
        bias_sensor_data = soft_iron_bias@true_data + hard_iron_bias + noise

        return bias_sensor_data


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.scatter(x, y, s = 0.25)

    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax.grid()


def main() -> None:
    magnetometer_sensor_data = GenerateSensorData(NUMBER_OF_POINTS)
    X, Y, Z = magnetometer_sensor_data.generate_points()[:3]

    plots = [
        ("X vs Y", X, Y),
        ("X vs Z", X, Z),
        ("Y vs Z", Y, Z)
    ]

    #Orthogonal Projection & Histogram
    for title, data_x, data_y in plots:
        fig, axs = plt.subplot_mosaic([['histx', '.'], ['scatter','histy']],
                                    figsize=(6,6), width_ratios=(4, 1), height_ratios=(1, 4),
                                    layout='constrained')
        fig.suptitle(title)
        scatter_hist(data_x, data_y, axs['scatter'], axs['histx'], axs['histy'])

    #3-D Scatter Plots
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z, s = 1, c = "red")

    plt.show()

    return


if __name__ == "__main__":
    main()