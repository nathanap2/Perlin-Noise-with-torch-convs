import torch


def build_random_gradient_grid(height=4, width=4, normalize=True):
    """
    Return tensor of shape [1, 2, H, W] which contains (g_x, g_y) for each grid node
    """
    g = torch.randn((1, 2, height, width))  # random normal
    if normalize:
        norms = g.norm(dim=1, keepdim=True)  # shape (1, 1, H, W)
        norms = torch.clamp(norms, min=1e-8)
        g = g / norms
    return g
