"""
visualization.py

Provides functions to visualize 2D geometric shapes (Box and Circle) using matplotlib.
"""

import matplotlib.pyplot as plt
from abstractions.primitives.shapes import Box, Circle


def show_boxes(boxes: list[Box], limits=(-1, 1)):
    """
    Visualizes a list of Box objects on a 2D plot using matplotlib.

    Args:
        boxes (list[Box]): A list of Box instances to be displayed.
        limits (tuple, optional): The (min, max) bounds for both x and y axes. Default is (-1, 1).
    """
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)

    for i, box in enumerate(boxes):
        rect = plt.Rectangle(
            (
                box.center[0].item() - box.scale[0].item() / 2,
                box.center[1].item() - box.scale[1].item() / 2,
            ),
            box.scale[0].item(),
            box.scale[1].item(),
            rotation_point="center",
            color=f"C{i}",
            fill=False,
        )
        ax.add_patch(rect)

    plt.grid()
    plt.show()


def show_circles(circles: list[Circle], limits=(-1, 1)):
    """
    Visualizes a list of Circle objects on a 2D plot using matplotlib.

    Args:
        circles (list[Circle]): A list of Circle instances to be displayed.
        limits (tuple, optional): The (min, max) bounds for both x and y axes. Default is (-1, 1).
    """
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)

    for i, circle in enumerate(circles):
        circ = plt.Circle(
            (circle.center[0].item(), circle.center[1].item()),
            circle.radius.item(),
            color=f"C{i}",
            fill=False,
        )
        ax.add_patch(circ)

    plt.grid()
    plt.show()
