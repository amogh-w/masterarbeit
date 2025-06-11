from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect, SymTrans


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
