from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect, SymTrans


def lamp_1(
    base_width,
    base_height,
    stem_height,
    stem_thickness,
    lampshade_width,
    lampshade_height,
    # Removed lampshade_offset_y from here to ensure direct stacking by default.
    # If a gap is needed, it can be introduced by explicitly adding a small offset to a parameter.
):
    """
    Constructs a simple lamp shape design with a base, stem, and a rectangular lampshade.
    Ensures that the components are always touching (stacked on top of each other).
    Uses only: Union, SymRef, Move, Rect, SymTrans.

    Returns:
        Shape: A composite shape representing a lamp.
    """
    # Base (Rectangular, centered at Y=0, so its bottom is at -base_height/2)
    base = Rect(base_width, base_height)

    # Stem (a thin rectangle moved on top of the base)
    # The bottom of the stem needs to align with the top of the base.
    # Top of base is at base_height / 2
    # Center of stem needs to be at (base_height / 2) + (stem_height / 2)
    stem_y_pos = base_height / 2 + stem_height / 2
    stem = Move(
        Rect(stem_thickness, stem_height),
        0,  # X-coordinate: centered
        stem_y_pos,
    )

    # Lampshade (a wider rectangular prism moved on top of the stem)
    # The bottom of the lampshade needs to align with the top of the stem.
    # Top of stem is at stem_y_pos + (stem_height / 2)
    # Center of lampshade needs to be at (top of stem) + (lampshade_height / 2)
    lampshade_y_pos = stem_y_pos + stem_height / 2 + lampshade_height / 2
    lampshade = Move(
        Rect(lampshade_width, lampshade_height),
        0,  # X-coordinate: centered
        lampshade_y_pos,
    )

    # Union all parts together
    return Union(Union(base, stem), lampshade)


def random_lamps_1(num_shapes: int):
    """
    Generates a list of `lamp_1` designs with random dimensions.
    Ensures consistent touching between components.
    Uses only: Union, SymRef, Move, Rect, SymTrans.

    Args:
        num_shapes (int): Number of lamp shapes to generate.

    Returns:
        list[Shape]: List of lamp shapes.
    """
    return [
        lamp_1(
            base_width=random_quantized_uniform(0.3, 0.8, 10),
            base_height=random_quantized_uniform(0.05, 0.15, 6),
            stem_height=random_quantized_uniform(0.5, 1.5, 20),
            stem_thickness=random_quantized_uniform(0.02, 0.08, 5),
            lampshade_width=random_quantized_uniform(0.4, 1.2, 15),
            lampshade_height=random_quantized_uniform(0.2, 0.6, 8),
            # Removed lampshade_offset_y from the random generation,
            # as the base math ensures touching.
        )
        for _ in range(num_shapes)
    ]
