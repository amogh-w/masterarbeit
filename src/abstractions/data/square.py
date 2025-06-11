from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect, SymTrans


def square(center_x, center_y, width, height, thickness):
    """
    Constructs a square-like shape using two symmetric bars along X and Y axes.

    Args:
        center_x (float): X coordinate of the square center.
        center_y (float): Y coordinate of the square center.
        width (float): Width of the square.
        height (float): Height of the square.
        thickness (float): Thickness of the bars.

    Returns:
        Shape: A symmetric square-shaped composite.
    """
    x_part = SymTrans(
        Move(
            Rect(width, thickness), center_x, center_y - 0.5 * height + 0.5 * thickness
        ),
        axis="y",
        dist=height - thickness,
        degree=2,
    )

    y_part = SymTrans(
        Move(
            Rect(thickness, height), center_x - 0.5 * width + 0.5 * thickness, center_y
        ),
        axis="x",
        dist=width - thickness,
        degree=2,
    )

    return Union(x_part, y_part)


def random_squares(num_shapes: int):
    """
    Generates a list of square-shaped compositions with random parameters.

    Args:
        num_shapes (int): Number of square shapes to generate.

    Returns:
        list[Shape]: List of square-shaped structures.
    """
    return [
        square(
            center_x=random_quantized_uniform(-2.0, 2.0, 40),
            center_y=random_quantized_uniform(-2.0, 2.0, 40),
            width=random_quantized_uniform(2.0, 4.0, 20),
            height=random_quantized_uniform(2.0, 4.0, 20),
            thickness=random_quantized_uniform(0.1, 1.0, 9),
        )
        for _ in range(num_shapes)
    ]
