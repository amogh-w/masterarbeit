from functools import reduce
from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.nodes import Union, Move, Rect


def nest_union(shapes):
    return reduce(lambda a, b: Union(a, b), shapes)


def lamp_1(
    base_width,
    base_thickness,
    stand_height,
    stand_thickness,
    shade_width,
    shade_height,
    shade_thickness,
):
    """
    Generates a simple lamp shape with base, stand, and shade.

    Parameters
    ----------
    base_width : float
        Width of the lamp base.
    base_thickness : float
        Thickness (height) of the lamp base.
    stand_height : float
        Height of the lamp stand.
    stand_thickness : float
        Thickness (width) of the lamp stand.
    shade_width : float
        Width of the lamp shade.
    shade_height : float
        Height of the lamp shade.
    shade_thickness : float
        Thickness (depth) of the lamp shade.

    Returns
    -------
    Shape
        Composite lamp shape.
    """
    base = Move(
        Rect(base_width, base_thickness), 0, -stand_height / 2 - base_thickness / 2
    )
    stand = Move(Rect(stand_thickness, stand_height), 0, 0)
    shade = Move(
        Rect(shade_width, shade_height), 0, stand_height / 2 + shade_height / 2
    )

    all_parts = nest_union([base, stand, shade])
    return all_parts


def random_lamps_1(num_shapes: int):
    """
    Generates a list of random lamp shapes with varying parameters.

    Parameters
    ----------
    num_shapes : int
        Number of lamp shapes to generate.

    Returns
    -------
    list[Shape]
        List of generated lamp shapes.
    """
    return [
        lamp_1(
            base_width=random_quantized_uniform(0.3, 1.0, 15),
            base_thickness=random_quantized_uniform(0.05, 0.2, 6),
            stand_height=random_quantized_uniform(0.5, 1.5, 20),
            stand_thickness=random_quantized_uniform(0.05, 0.2, 6),
            shade_width=random_quantized_uniform(0.4, 1.0, 15),
            shade_height=random_quantized_uniform(0.3, 1.0, 15),
            shade_thickness=random_quantized_uniform(0.05, 0.2, 6),
        )
        for _ in range(num_shapes)
    ]
