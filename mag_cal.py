import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

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

def three_dim_scatter(cal_points:np.ndarray, uncal_points:np.ndarray) -> None:
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    Ux, Uy, Uz = uncal_points[:3]
    Cx, Cy, Cz = cal_points[:3]

    ax.scatter(Ux, Uy, Uz, marker = 'o', label="Uncalibrated Data") #Uncalibrated data
    ax.scatter(Cx, Cy, Cz, color = 'red', marker = 'o', label="Calibrated Data") #Calibrated
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect("equal")

    ax.legend()
    plt.show()

    return

def unit_circle_fit():
    return


def main() -> None:
    """
        Generate random Magnetometer data.
        In reality you would grab real sensor
        data and import the data into python but the 
        calibration works similarly
    """

    unit_sphere, uncalibrated = generate_uncalibrated_data()
    three_dim_scatter(unit_sphere[:3], uncalibrated[:3],)
    

    return




if (__name__ == "__main__"):
    main()

