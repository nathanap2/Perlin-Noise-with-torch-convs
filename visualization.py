from matplotlib import pyplot as plt
import numpy as np

def visualize_vector_field_quiver(field: np.ndarray):
    _,H,W = field.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    dx, dy = field
    plt.figure()
    plt.quiver(x_coords,y_coords,dx,dy,scale_units='xy',color='blue')
    plt.show() 
