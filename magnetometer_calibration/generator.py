import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

class GenerateSensorData():
    def __init__(self, number_of_points) -> None:
        self.number_of_points = number_of_points

    def generate_points(self) -> np.ndarray:
        #Generating random points on unit sphere
        phi = np.random.uniform(0, np.pi*2, self.number_of_points)
        theta = np.random.uniform(0, np.pi*2, self.number_of_points)
        points = np.array([np.sin( theta ) * np.cos( phi ),
                 np.sin( theta ) * np.sin( phi ),
                 np.cos( theta ),
                 np.ones(self.number_of_points)])

        return points

def main() -> None:
    magnetometer_sensor_data = GenerateSensorData(500)
    points = magnetometer_sensor_data.generate_points()

    return

if __name__ == "__main__":
    main()