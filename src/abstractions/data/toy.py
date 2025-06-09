"""
toy_data.py

This module provides functions to generate toy shapes using a domain-specific language (DSL)
for compositional geometry. It includes basic shape primitives like squares, chairs, and tables,
as well as random generators for testing abstraction discovery algorithms.
"""

import random

from abstractions.dsl import Union, SymRef, Move, Rect, SymTrans, Shape


def random_quantized_uniform(low: float, high: float, steps: int) -> float:
    """
    Generates a random float value between `low` and `high`, quantized into a number of steps.

    Args:
        low (float): Lower bound of the range.
        high (float): Upper bound of the range.
        steps (int): Number of discrete steps between low and high.

    Returns:
        float: A quantized random float within the specified range.
    """
    step = random.randint(0, steps)
    return low + (high - low) * step / steps


def random_shape(max_nodes: int = 32):
    """
    Recursively generates a random shape composed of primitive and composite operations.

    Args:
        max_nodes (int): Maximum number of nodes in the shape expression tree.

    Returns:
        Shape: A randomly composed shape.
    """
    op_index = random.randint(0, 4)

    if max_nodes <= 1 or op_index == 0:
        # Rect
        return Rect(
            random_quantized_uniform(0.5, 2.0, 15),
            random_quantized_uniform(0.5, 2.0, 15),
        )
    elif op_index == 1:
        # Move
        return Move(
            random_shape(max_nodes - 1),
            random_quantized_uniform(0.5, 2.0, 40),
            random_quantized_uniform(0.5, 2.0, 40),
        )
    elif op_index == 2:
        # Union
        return Union(random_shape(max_nodes // 2), random_shape((max_nodes - 1) // 2))
    elif op_index == 3:
        # SymTrans
        return SymTrans(
            random_shape(max_nodes - 1),
            random.choice(["x", "y"]),
            random_quantized_uniform(-2.0, 2.0, 40),
            random.randint(2, 5),
        )
    elif op_index == 4:
        # SymRef
        return SymRef(random_shape(max_nodes - 1), random.choice(["x", "y"]))
    else:
        raise ValueError("Invalid operation index")


def random_shapes(num_shapes: int, max_nodes: int = 32) -> list[Shape]:
    """
    Generates a list of random shapes.

    Args:
        num_shapes (int): Number of shapes to generate.
        max_nodes (int): Maximum nodes per shape.

    Returns:
        list[Shape]: A list of randomly composed shapes.
    """
    return [random_shape(max_nodes) for _ in range(num_shapes)]


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


def chair_1(
    width, leg_height, leg_thickness, back_height, back_thickness, seat_thickness
):
    """
    Constructs a simple chair shape design with two vertical legs, a seat, and a back.

    Returns:
        Shape: A composite shape representing a chair.
    """
    legs = SymRef(
        Move(
            Rect(leg_thickness, leg_height),
            width / 2 - leg_thickness / 2,
            -leg_height / 2 - seat_thickness / 2,
        ),
        "x",
    )

    back = Move(
        Rect(back_thickness, back_height),
        -width / 2 + back_thickness / 2,
        back_height / 2 + seat_thickness / 2,
    )

    seat = Rect(width, seat_thickness)

    return Union(Union(back, seat), legs)


def random_chairs_1(num_shapes: int):
    """
    Generates a list of `chair_1` designs with random dimensions.

    Args:
        num_shapes (int): Number of chair shapes to generate.

    Returns:
        list[Shape]: List of chair shapes.
    """
    return [
        chair_1(
            width=random_quantized_uniform(0.5, 1.0, 20),
            leg_height=random_quantized_uniform(0.5, 1.0, 20),
            leg_thickness=random_quantized_uniform(0.05, 0.2, 6),
            back_height=random_quantized_uniform(0.2, 1.0, 32),
            back_thickness=random_quantized_uniform(0.05, 0.2, 6),
            seat_thickness=random_quantized_uniform(0.05, 0.2, 6),
        )
        for _ in range(num_shapes)
    ]


def chair_2(
    width, leg_height, leg_thickness, back_height, back_thickness, seat_thickness
):
    """
    Constructs an alternative chair design with vertical and horizontal legs.

    Returns:
        Shape: A composite chair structure.
    """
    legs = Union(
        Move(
            Rect(leg_thickness, leg_height), 0.0, -leg_height / 2 - seat_thickness / 2
        ),
        Move(
            Rect(width, leg_thickness),
            0.0,
            -leg_height - leg_thickness / 2 - seat_thickness / 2,
        ),
    )

    back = Move(
        Rect(back_thickness, back_height),
        -width / 2 + back_thickness / 2,
        back_height / 2 + seat_thickness / 2,
    )

    seat = Rect(width, seat_thickness)

    return Union(Union(back, seat), legs)


def random_chairs_2(num_shapes: int):
    """
    Generates a list of `chair_2` shapes with random parameters.

    Args:
        num_shapes (int): Number of shapes to generate.

    Returns:
        list[Shape]: List of chair designs.
    """
    return [
        chair_2(
            width=random_quantized_uniform(0.5, 1.0, 20),
            leg_height=random_quantized_uniform(0.5, 1.0, 20),
            leg_thickness=random_quantized_uniform(0.05, 0.2, 6),
            back_height=random_quantized_uniform(0.2, 1.0, 32),
            back_thickness=random_quantized_uniform(0.05, 0.2, 6),
            seat_thickness=random_quantized_uniform(0.05, 0.2, 6),
        )
        for _ in range(num_shapes)
    ]


def table_1(width, leg_height, leg_thickness, top_thickness):
    """
    Constructs a simple symmetric table using a rectangular top and legs.

    Returns:
        Shape: A composite table shape.
    """
    legs = SymRef(
        Move(
            Rect(leg_thickness, leg_height),
            width / 2 - leg_thickness / 2,
            -leg_height / 2 - top_thickness / 2,
        ),
        "x",
    )

    top = Rect(width, top_thickness)

    return Union(top, legs)


def random_tables_1(num_shapes: int):
    """
    Generates a list of `table_1` structures with randomized dimensions.

    Args:
        num_shapes (int): Number of table shapes to generate.

    Returns:
        list[Shape]: List of table-shaped structures.
    """
    return [
        table_1(
            width=random_quantized_uniform(0.5, 1.0, 20),
            leg_height=random_quantized_uniform(0.2, 0.7, 20),
            leg_thickness=random_quantized_uniform(0.05, 0.2, 6),
            top_thickness=random_quantized_uniform(0.05, 0.2, 6),
        )
        for _ in range(num_shapes)
    ]
