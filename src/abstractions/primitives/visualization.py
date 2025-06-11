"""
visualization.py

Provides functions to visualize 2D geometric shapes (Box and Circle) using k3d for interactive viewing.
"""

# import matplotlib.pyplot as plt
import k3d
import numpy as np
from abstractions.primitives.shapes import Box, Circle


# def show_boxes(boxes: list[Box], limits=(-1, 1)):
#     """
#     Visualizes a list of Box objects on a 2D plot using matplotlib.

#     Args:
#         boxes (list[Box]): A list of Box instances to be displayed.
#         limits (tuple, optional): The (min, max) bounds for both x and y axes. Default is (-1, 1).
#     """
#     fig, ax = plt.subplots()
#     ax.set_aspect("equal")
#     ax.set_xlim(*limits)
#     ax.set_ylim(*limits)

#     for i, box in enumerate(boxes):
#         rect = plt.Rectangle(
#             (
#                 box.center[0].item() - box.scale[0].item() / 2,
#                 box.center[1].item() - box.scale[1].item() / 2,
#             ),
#             box.scale[0].item(),
#             box.scale[1].item(),
#             rotation_point="center",
#             color=f"C{i}",
#             fill=False,
#         )
#         ax.add_patch(rect)

#     plt.grid()
#     plt.show()


# def show_circles(circles: list[Circle], limits=(-1, 1)):
#     """
#     Visualizes a list of Circle objects on a 2D plot using matplotlib.

#     Args:
#         circles (list[Circle]): A list of Circle instances to be displayed.
#         limits (tuple, optional): The (min, max) bounds for both x and y axes. Default is (-1, 1).
#     """
#     fig, ax = plt.subplots()
#     ax.set_aspect("equal")
#     ax.set_xlim(*limits)
#     ax.set_ylim(*limits)

#     for i, circle in enumerate(circles):
#         circ = plt.Circle(
#             (circle.center[0].item(), circle.center[1].item()),
#             circle.radius.item(),
#             color=f"C{i}",
#             fill=False,
#         )
#         ax.add_patch(circ)

#     plt.grid()
#     plt.show()


def show_boxes(boxes: list[Box], limits=(-1, 1), lock_2d_view=True):
    """
    Visualizes a list of Box objects in a 3D interactive k3d plot.

    Args:
        boxes (list[Box]): A list of Box instances to be displayed.
        limits (tuple, optional): The (min, max) bounds for both x and y axes. Default is (-1, 1).
    """
    plot = k3d.plot()
    for i, box in enumerate(boxes):
        cx, cy = box.center.tolist()
        sx, sy = box.scale.tolist()
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z = 0.0
        lines = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]]
        plot += k3d.line(np.array(lines, dtype=np.float32), color=0x0055FF, width=0.01)
    if lock_2d_view:
        plot.camera_auto_fit = False
        plot.camera_no_rotate = True
        plot.camera = [0, 0, 3, 0, 0, 0, 0, 1, 0]
    plot.display()


def show_circles(circles: list[Circle], limits=(-1, 1), lock_2d_view=True):
    """
    Visualizes a list of Circle objects in a 3D interactive k3d plot.

    Args:
        circles (list[Circle]): A list of Circle instances to be displayed.
        limits (tuple, optional): The (min, max) bounds for both x and y axes. Default is (-1, 1).
    """
    plot = k3d.plot()
    theta = np.linspace(0, 2 * np.pi, 100)
    for i, circle in enumerate(circles):
        cx, cy = circle.center.tolist()
        r = circle.radius.item()
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        z = np.zeros_like(x)
        coords = np.vstack([x, y, z]).T.astype(np.float32)
        plot += k3d.line(coords, color=0xFF0000, width=0.01)
    if lock_2d_view:
        plot.camera_auto_fit = False
        plot.camera_no_rotate = True
        plot.camera = [0, 0, 3, 0, 0, 0, 0, 1, 0]
    plot.display()
