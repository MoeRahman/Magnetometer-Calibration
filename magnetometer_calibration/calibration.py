import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

class CalibrateData():
    def __init__(self, sensor_data) -> None:
        self.sensor_data = sensor_data
        return

    def hard_iron_bias(self) -> np.ndarray:
        centroid = self.sensor_data.mean(axis = 1)
        return centroid

    def soft_iron_bias(self) -> np.ndarray:
        points = self.sensor_data
        number_of_points = points.shape[1]
        homogeneous_points = np.vstack((points, np.ones((1, number_of_points))))

        def transform(points, transform_matrix):
            transformed_points = homogeneous_points.transpose() @ transform_matrix
            return transformed_points[:, :3]

        def loss_function(matrix_params, points):
            transform_matrix = matrix_params.reshape((4,4))
            transformed_points = transform(points, transform_matrix)
            distances = np.linalg.norm(transformed_points, axis=1) - 1
            return np.sum(distances**2)

        initial_transform = np.eye(4).flatten()
        result = sp.optimize.least_squares(loss_function, initial_transform, args=(points,))
        optimized_transform_matrix = result.x.reshape((4,4))
        return optimized_transform_matrix


def main():
    NUMBER_OF_POINTS = 500

    magnetometer_sensor_data = GenerateSensorData(NUMBER_OF_POINTS)
    coordinates = magnetometer_sensor_data.generate_uniform_points()

    calibrate = CalibrateData(coordinates)
    hard_iron_bias_estimate = calibrate.hard_iron_bias()

    calibrate.sensor_data -= hard_iron_bias_estimate

    transformation_matrix = calibrate.soft_iron_bias()

    homogeneous_points = np.vstack((coordinates, np.ones((1, coordinates.shape[1]))))
    print(homogeneous_points.transpose()@transformation_matrix)
    return


if __name__ == "__main__":
    from generator import GenerateSensorData
    main()