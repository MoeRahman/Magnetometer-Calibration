import matplotlib.pyplot as plt
import numpy as np

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

        soft_iron_bias = np.array([[1.0, 0.5, 0.0], [0.0, 10.0, 0.0], [0.6, 0.0, 1.0]])
        hard_iron_bias = np.random.uniform(0, 5, (3,1))

        noise = np.random.normal(0, 0.1, (3, self.number_of_points))
        bias_sensor_data = soft_iron_bias@true_data + hard_iron_bias + noise
        return bias_sensor_data


def main() -> None:
    NUMBER_OF_POINTS = 500

    magnetometer_sensor_data = GenerateSensorData(NUMBER_OF_POINTS)
    X, Y, Z = magnetometer_sensor_data.generate_points()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z, s = 1, c = "red")

    plt.show()
    return


if __name__ == "__main__":
    main()