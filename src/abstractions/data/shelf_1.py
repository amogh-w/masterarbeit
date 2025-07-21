from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect, SymTrans


def shelf_1(
    shelf_width,
    shelf_thickness,
    support_height,
    support_thickness,
    support_offset_x,  # How far from the center the supports are placed
):
    """
    Constructs a simple shelf shape design with a horizontal board and two vertical supports.

    Returns:
        Shape: A composite shape representing a shelf.
    """
    # The main horizontal shelf board
    board = Rect(shelf_width, shelf_thickness)

    # A single support (rectangular)
    # We'll use SymRef to create the second one symmetrically
    single_support = Move(
        Rect(support_thickness, support_height),
        support_offset_x,  # Position for the first support
        -shelf_thickness / 2 - support_height / 2,  # Place below the board
    )

    # Use SymRef to create the mirrored support
    supports = SymRef(single_support, "x")  # Symmetrical reflection across the Y-axis

    # Union the board and the supports
    return Union(board, supports)


def random_shelves_1(num_shapes: int):
    """
    Generates a list of `shelf_1` designs with random dimensions.

    Args:
        num_shapes (int): Number of shelf shapes to generate.

    Returns:
        list[Shape]: List of shelf shapes.
    """
    return [
        shelf_1(
            shelf_width=random_quantized_uniform(
                0.8, 2.5, 20
            ),  # Width of the shelf board
            shelf_thickness=random_quantized_uniform(
                0.05, 0.15, 6
            ),  # Thickness of the shelf board
            support_height=random_quantized_uniform(
                0.2, 0.5, 10
            ),  # Height of the supports
            support_thickness=random_quantized_uniform(
                0.03, 0.1, 5
            ),  # Thickness of the supports
            support_offset_x=random_quantized_uniform(
                0.2, 0.8, 15
            ),  # How far supports are from center
        )
        for _ in range(num_shapes)
    ]
