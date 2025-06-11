from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect, SymTrans


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
