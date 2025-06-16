from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect


def chair_3(
    width,
    leg_height,
    leg_thickness,
    back_height,
    back_thickness,
    seat_thickness,
    armrest_thickness,
):
    """
    Constructs a chair with legs, a seat, backrest, and armrests.

    Parameters
    ----------
    width : float
        Total width of the chair.
    leg_height : float
        Height of the legs.
    leg_thickness : float
        Thickness of the legs.
    back_height : float
        Height of the backrest.
    back_thickness : float
        Thickness of the backrest.
    seat_thickness : float
        Thickness of the seat.
    armrest_thickness : float
        Thickness of the armrests.

    Returns
    -------
    Shape
        Composite chair shape.
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

    armrests = Union(
        Move(
            Rect(armrest_thickness, seat_thickness),
            -width / 2 - armrest_thickness / 2,
            seat_thickness / 2 + 0.05,
        ),
        Move(
            Rect(armrest_thickness, seat_thickness),
            width / 2 + armrest_thickness / 2,
            seat_thickness / 2 + 0.05,
        ),
    )

    return Union(Union(back, Union(seat, armrests)), legs)


def random_chairs_3(num_shapes: int):
    """
    Generates a list of `chair_3` shapes with random parameters.

    Parameters
    ----------
    num_shapes : int
        Number of chairs to generate.

    Returns
    -------
    list[Shape]
        List of chair shapes.
    """
    return [
        chair_3(
            width=random_quantized_uniform(0.5, 1.0, 20),
            leg_height=random_quantized_uniform(0.5, 1.0, 20),
            leg_thickness=random_quantized_uniform(0.05, 0.2, 6),
            back_height=random_quantized_uniform(0.2, 1.0, 32),
            back_thickness=random_quantized_uniform(0.05, 0.2, 6),
            seat_thickness=random_quantized_uniform(0.05, 0.2, 6),
            armrest_thickness=random_quantized_uniform(0.05, 0.15, 5),
        )
        for _ in range(num_shapes)
    ]
