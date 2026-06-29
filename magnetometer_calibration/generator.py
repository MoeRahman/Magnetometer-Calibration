import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

NUMBER_OF_POINTS = 500
class GenerateSensorData():
    def __init__(self, number_of_points) -> None:
        self.number_of_points = number_of_points

    def generate_points(self) -> np.ndarray:
        #Random Covariance
        noise_x = np.random.uniform(0, 2, self.number_of_points)
        noise_y = np.random.uniform(0, 2, self.number_of_points)
        noise_z = np.random.uniform(0, 2, self.number_of_points)

        #Generate random phi and theta angles for sphere
        phi = np.random.uniform(0, np.pi*2, self.number_of_points)
        theta = np.random.uniform(0, np.pi*2, self.number_of_points)

        #Generate points on unit sphere
        points = np.array([noise_x*(np.sin( theta ) * np.cos( phi )), # X
                           noise_y*(np.sin( theta ) * np.sin( phi )), # Y
                           noise_z*(np.cos( theta ))])                # Z

        return points

def main() -> None:
    magnetometer_sensor_data = GenerateSensorData(NUMBER_OF_POINTS)
    X, Y, Z = magnetometer_sensor_data.generate_points()[:3]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z)

    plt.show()

    return

if __name__ == "__main__":
    main()