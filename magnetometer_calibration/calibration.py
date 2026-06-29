import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

class CalibrateData():
    def __init__(self, sensor_data) -> None:
        self.sensor_data = sensor_data
        return

    def ellipsoidal_fit(self) -> np.ndarray:
        points = self.sensor_data

        def transform(points, transform_matrix):
            number_of_points = points.shape[1]
            homogeneous_points = np.vstack((points, np.ones((1,number_of_points))))
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
    print("Hello, World")
    return


if __name__ == "__main__":
    main()