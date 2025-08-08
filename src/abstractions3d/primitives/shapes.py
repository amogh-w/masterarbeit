from torch import Tensor


class Box3D:
    """3D box defined by center (x, y, z) and scale (width, height, depth)."""

    def __init__(self, center: Tensor, scale: Tensor):
        self.center = center
        self.scale = scale
