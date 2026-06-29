import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

NUMBER_OF_POINTS = 500
SOFT_IRON_MAGNITUDE = 0.25

class GenerateSensorData():
    def __init__(self, number_of_points) -> None:
        self.number_of_points = number_of_points

    def generate_points(self) -> np.ndarray:

        #Generate random phi and theta angles for sphere
        phi = np.random.uniform(0, np.pi*2, self.number_of_points)
        theta = np.random.uniform(0, np.pi*2, self.number_of_points)

        mag_x = np.sin( theta ) * np.cos( phi )
        mag_y = np.sin( theta ) * np.sin( phi )
        mag_z = np.cos( theta )

        #Generate points on unit sphere
        true_data = np.array([mag_x, mag_y, mag_z])

        #Random Magnetic Bias
        soft_iron_bias = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])
        
        hard_iron_bias = np.random.uniform(0, 10, (3,1))

        #Random Noise Matrix
        noise = np.random.normal(0, 0.1, (3, self.number_of_points))

        bias_sensor_data = soft_iron_bias@true_data + hard_iron_bias

        return bias_sensor_data


def main() -> None:
    magnetometer_sensor_data = GenerateSensorData(NUMBER_OF_POINTS)
    X, Y, Z = magnetometer_sensor_data.generate_points()[:3]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', aspect = 'auto')
    ax.scatter(X, Y, Z, s = 1, c = "red")

    plt.show()

    return


if __name__ == "__main__":
    main()