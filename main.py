from magnetometer_calibration.generator import GenerateSensorData

import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_POINTS = 500

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