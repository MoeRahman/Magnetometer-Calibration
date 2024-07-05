import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize 

def generate_uncalibrated_data() -> np.ndarray:
    
    #Generate a set of random unit vectors
    phi = np.random.uniform(0, np.pi*2, 500)
    theta = np.random.uniform(0, np.pi*2, 500)
    points = np.array([np.sin( theta ) * np.cos( phi ),
              np.sin( theta ) * np.sin( phi ),
              np.cos( theta ), 
              np.ones(500)])
    
    randScaleX, randScaleY, randScaleZ = [np.random.uniform(0, 10),
                                          np.random.uniform(0, 10),
                                          np.random.uniform(0, 10)]
    
    offsetX, offsetY, offsetZ = [np.random.uniform(0, 2),
                                 np.random.uniform(0, 2),
                                 np.random.uniform(0, 2)]
    
    
    affine_transform = np.array( [[randScaleX, 0.0, 0.0, offsetX],
                                  [0.0, randScaleY, 0.0, offsetY],
                                  [0.0, 0.0, randScaleZ, offsetZ],
                                  [0.0, 0.0, 0.0, 1]])
    
    biased_data = np.matmul(affine_transform, points)

    return points, biased_data

def three_dim_scatter(cal_points:np.ndarray, 
                      ideal_sphere:np.ndarray,
                      uncal_points:np.ndarray) -> None:
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    Ux, Uy, Uz = uncal_points[:3]
    Cx, Cy, Cz = cal_points[:3]
    Ix, Iy, Iz = ideal_sphere[:3]

    ax.scatter(Ux, Uy, Uz, marker = 'o', 
               label="Uncalibrated Data") #Uncalibrated data
    ax.scatter(Cx, Cy, Cz, color = 'red', marker = 'o', 
               label="Calibrated Data") #Calibrated
    ax.scatter(Ix, Iy, Iz, color = 'green', marker = '+', 
               label="Ideal Data") #Ideal
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect("equal")

    error = np.linalg.norm(cal_points - ideal_sphere, axis=1)
    rms_error = np.sqrt(np.mean(error**2))

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='w', 
                              label=f'RMS Error: {rms_error:.4f}'))
    ax.legend(handles=handles, loc='upper left')

    plt.show()

    return

def unit_circle_fit(points:np.ndarray) -> np.ndarray:

    def apply_affine_transformation(points, affine_transformation):
        n_points = points.shape[0]
        homogenous_points = np.hstack((points, np.ones((n_points, 1))))
        transformed_points = homogenous_points @ affine_transformation.T
        return transformed_points[:, :3]

    def loss_function(affine_params, points):

        # Reshape affine parameters to 4x4 matrix
        affine_transform = affine_params.reshape((4, 4))
        transformed_points = apply_affine_transformation(points, 
                                                         affine_transform)
        
        distances = np.linalg.norm(transformed_points, axis=1) - 1
        return np.sum(distances ** 2)

    
    # Initialize with the identity affine transformation
    initial_affine_transformation = np.eye(4).flatten()
    result = minimize(loss_function, initial_affine_transformation, 
                      args=(points,), method='L-BFGS-B')
    optimized_affine_transformation = result.x.reshape((4, 4))
    
    return optimized_affine_transformation


def main() -> None:
    """
        Generate random Magnetometer data.
        In reality you would grab real sensor
        data and import the data into python but the 
        calibration works similarly
    """

    unit_sphere, uncalibrated = generate_uncalibrated_data()

    calibrated_transform = unit_circle_fit(np.transpose(uncalibrated[:3]))
    calibrated = np.matmul(calibrated_transform, uncalibrated)

    three_dim_scatter(calibrated[:3] ,unit_sphere[:3], uncalibrated[:3])
    

    return




if (__name__ == "__main__"):
    main()

