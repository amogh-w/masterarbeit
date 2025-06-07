"""
primitives.py

This module defines the core geometric primitive used for abstraction: the Box.
It also includes utility functions for visualizing collections of boxes using matplotlib.
"""
import matplotlib.pyplot as plt
from torch import Tensor


class Box:
    """
    Represents a rectangular box in 2D space using a center point and a scale (width and height).

    Attributes:
        center (Tensor): A 2D tensor representing the center coordinates (x, y).
        scale (Tensor): A 2D tensor representing the width and height of the box.
    """
    def __init__(self, center: Tensor, scale: Tensor):
        """
        Initializes a Box with a given center and scale.

        Args:
            center (Tensor): The center coordinates of the box.
            scale (Tensor): The size of the box along the x and y axes.
        """
        self.center = center
        self.scale = scale


def show_boxes(boxes: list[Box], limits=(-1, 1)):
    """
    Visualizes a list of Box objects on a 2D plot using matplotlib.

    Args:
        boxes (list[Box]): A list of Box instances to be displayed.
        limits (tuple, optional): The (min, max) bounds for both x and y axes. Default is (-1, 1).
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)

    for i, box in enumerate(boxes):
        rect = plt.Rectangle(
            (box.center[0].item() - box.scale[0].item() / 2, box.center[1].item() - box.scale[1].item() / 2),
            box.scale[0].item(),
            box.scale[1].item(),
            rotation_point="center",
            color=f"C{i}",
            fill=False
        )
        ax.add_patch(rect)

    plt.grid()
    plt.show()