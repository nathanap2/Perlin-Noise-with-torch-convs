import os
import numpy as np
import torch
from smoothstep_functions import fade

"""
core code for the torch conv method
It's fondamentally the same as the "Kronecker-like" method in numpy
but it uses convTranspose2D, which is simple and improves readability
+ might open some learning possibilities
"""


def build_centered_smoothstep_kernel(cell_size, smoothstep_fn = fade):
    """
    Build 2D kernel of shape (2*cell_size+1) x (2*cell_size+1) 
    """
    size = 2*cell_size + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    
    for i in range(size):
        for j in range(size):
            # distance to center 
            dx = abs(i - cell_size)/cell_size
            dy = abs(j - cell_size)/cell_size
            # actually we want 1 at the center and then a decrease
            t_x = 1.0 - dx
            t_y = 1.0 - dy
            
            val_x = smoothstep_fn(t_x)
            val_y = smoothstep_fn(t_y)
            
            kernel[i, j] = val_x * val_y
    
    return kernel


def build_offset_field(cell_size: int) -> np.ndarray:
    """
    2D field of shape (2, 2*cs+1, 2*cs+1)
    """
    size = 2 * cell_size + 1
    offset_field = np.zeros((2, size, size), dtype=np.float32)
    
    for i in range(size):
        dx = i - cell_size  # in [-cell_size, +cell_size]
        for j in range(size):
            dy = j - cell_size
            offset_field[0, i, j] = dx/cell_size
            offset_field[1, i, j] = dy/cell_size
    return offset_field


def build_perlin_kernel(cell_size: int) -> torch.Tensor:
    dxdy = build_offset_field(cell_size) #(Δx, Δy), shape=(2, H, W)
    fade_map = build_centered_smoothstep_kernel(cell_size) #shape=(H, W)
    kernel = dxdy * fade_map
    return kernel[None,...]
    
    
def build_perlin_transpose_conv(cell_size: int) -> torch.nn.ConvTranspose2d:
    kernel_size = (2*cell_size+1)
    convT = torch.nn.ConvTranspose2d(
        in_channels=2,
        out_channels=1,
        kernel_size=kernel_size,
        stride=cell_size,
        padding=cell_size,
        bias=False
    )
    kernel = torch.Tensor(build_perlin_kernel(cell_size))
    kernel = kernel.permute(1,0,2,3)
    with torch.no_grad():
        convT.weight.copy_(kernel)
    return convT
