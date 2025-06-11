from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect, SymTrans


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
