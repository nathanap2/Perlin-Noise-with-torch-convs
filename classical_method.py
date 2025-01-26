import numpy as np

"""
Implementation of the standard algorithm in its parallelized version.
This is essentially the same algo as, for instance, 
https://github.com/pvigier/perlin-numpy/tree/master 
by Pierre Vigier & Laurence Warne, except some objects 
may be pre-computed for faster downstream processing
(moreover, there seem to be a slight perf improvement when
we avoid manipulating huge grids of repeated gradients)
"""


def lerp(a, b, w):
    return a + w * (b - a)

def create_cell_template(perlin_cell_width):
    frac_1D_values = np.linspace(0, 1, perlin_cell_width, endpoint = False)
    frac_2D_values = np.meshgrid(frac_1D_values,frac_1D_values, indexing='ij')
    frac_2D_values_one_cell = np.stack(frac_2D_values,
                                       axis = -1)
    return frac_2D_values_one_cell
    
def perlin_classical(gradients, offset_cell, cell_weights):
    """
    Here the idea is to use something quite similar to Kronecker product
    except that we have matrix which coefs are vectors instead of scalar
    and we multiply them with dot product

    Parameters
    ----------
    gradients : np.ndarray of shape (n_x+1, n_y+1, 2)
        The Perlin gradient vectors at each node. 
        For a grid of n_x by n_y cells, we have (n_x+1)*(n_y+1) gradient nodes.

    offset_cell : np.ndarray of shape (cw, cw, 2)
        The (dx, dy) offsets for a single cell, typically in the range [0,1].

    cell_weights : np.ndarray of shape (cw, cw, 2)
        The (fade_x, fade_y) interpolation weights for each (i, j) within a cell.
        Often this is just the classical Perlin fade function applied to dx, dy.

    Returns
    -------
    noise : np.ndarray of shape (n_x * cw, n_y * cw)
        The reconstructed Perlin noise over the entire grid.
        A vertical flip is applied at the end to match the original orientation.
    """

    n_x = gradients.shape[0] - 1
    n_y = gradients.shape[1] - 1  # same as n_x for now
    cw  = offset_cell.shape[0]    # cell width

    # Extract the 4 corner gradients grids
    #    shapes: (n_x, n_y, 2)
    g_tl = gradients[:-1, :-1]  # top-left
    g_tr = gradients[1:,  :-1]  # top-right
    g_bl = gradients[:-1, 1:]   # bottom-left
    g_br = gradients[1:,  1:]   # bottom-right

    # Utility to shift offsets by (shift_x, shift_y),
    # e.g., subtract (1,0) if referencing the right corner, etc.
    def shifted_cell(shift_x, shift_y):
        return offset_cell - np.array([shift_x, shift_y])

    # Use Einstein summation to compute dot-products between all 
    # gradients and in-cell offsets combinations
    dot_tl = np.einsum('xyc, ijc -> xyij', g_tl, offset_cell)
    dot_tr = np.einsum('xyc, ijc -> xyij', g_tr, shifted_cell(1, 0))
    dot_bl = np.einsum('xyc, ijc -> xyij', g_bl, shifted_cell(0, 1))
    dot_br = np.einsum('xyc, ijc -> xyij', g_br, shifted_cell(1, 1))

    # reshape cell_weights to match dot_** shapes (could be done outside the fn
    # before calling it)
    fade_x = cell_weights[..., 0]  # shape (cw, cw)
    fade_y = cell_weights[..., 1]  # shape (cw, cw)
    fade_x_4d = fade_x[None, None, :, :]  # shape (1,1,cw,cw)
    fade_y_4d = fade_y[None, None, :, :]

    # blend
    dot_top    = lerp(dot_tl, dot_tr, fade_x_4d)  # (n_x, n_y, cw, cw)
    dot_bottom = lerp(dot_bl, dot_br, fade_x_4d)
    blended    = lerp(dot_top, dot_bottom, fade_y_4d)

    # collapse on 2D block matrix
    return blended.transpose(0, 2, 1, 3).reshape(n_x * cw, n_y * cw)


