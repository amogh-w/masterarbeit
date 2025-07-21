from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect, SymTrans


def window_1(
    outer_width,
    outer_height,
    frame_thickness,
    mullion_thickness=None,  # Optional: set to None if no mullions desired
    num_horizontal_mullions=0,
    num_vertical_mullions=0,
):
    """
    Constructs a simple rectangular window with an outer frame and optional internal mullions.

    Args:
        outer_width (float): The total width of the window.
        outer_height (float): The total height of the window.
        frame_thickness (float): The thickness of the outer frame.
        mullion_thickness (float, optional): The thickness of the internal mullions.
                                             If None, no mullions are generated.
        num_horizontal_mullions (int): Number of horizontal mullions (dividers).
        num_vertical_mullions (int): Number of vertical mullions (dividers).

    Returns:
        Shape: A composite shape representing a window.
    """
    # 1. Outer Frame (four thin rectangles forming a frame)
    # This creates a hollow frame implicitly by unioning 4 pieces.

    # Top and Bottom parts of the frame
    horizontal_frame_part = Rect(outer_width, frame_thickness)

    # Move top part up
    top_frame = Move(horizontal_frame_part, 0, outer_height / 2 - frame_thickness / 2)
    # Move bottom part down
    bottom_frame = Move(
        horizontal_frame_part, 0, -outer_height / 2 + frame_thickness / 2
    )

    # Left and Right parts of the frame (adjusted height to fit between top/bottom frames)
    vertical_frame_height = outer_height - 2 * frame_thickness
    vertical_frame_part = Rect(frame_thickness, vertical_frame_height)

    # Move left part left
    left_frame = Move(vertical_frame_part, -outer_width / 2 + frame_thickness / 2, 0)
    # Move right part right
    right_frame = Move(vertical_frame_part, outer_width / 2 - frame_thickness / 2, 0)

    # Combine all frame parts
    frame = Union(top_frame, Union(bottom_frame, Union(left_frame, right_frame)))

    # 2. Internal Mullions
    mullions = []
    if mullion_thickness is not None and (
        num_horizontal_mullions > 0 or num_vertical_mullions > 0
    ):
        # Calculate the inner dimensions where mullions will reside
        inner_width = outer_width - 2 * frame_thickness
        inner_height = outer_height - 2 * frame_thickness

        # Horizontal Mullions
        if num_horizontal_mullions > 0:
            mullion_spacing_h = inner_height / (num_horizontal_mullions + 1)
            for i in range(num_horizontal_mullions):
                # Y position relative to inner window center
                y_pos = (i + 1) * mullion_spacing_h - inner_height / 2
                mullions.append(Move(Rect(inner_width, mullion_thickness), 0, y_pos))

        # Vertical Mullions
        if num_vertical_mullions > 0:
            mullion_spacing_v = inner_width / (num_vertical_mullions + 1)
            for i in range(num_vertical_mullions):
                # X position relative to inner window center
                x_pos = (i + 1) * mullion_spacing_v - inner_width / 2
                mullions.append(Move(Rect(mullion_thickness, inner_height), x_pos, 0))

    # Combine all mullions into a single Union (if any exist)
    all_mullions = None
    if mullions:
        all_mullions = mullions[0]
        for m in mullions[1:]:
            all_mullions = Union(all_mullions, m)

    # Final Union: frame + mullions (if any)
    if all_mullions:
        return Union(frame, all_mullions)
    else:
        return frame


def random_windows_1(num_shapes: int):
    """
    Generates a list of `window_1` designs with random dimensions and mullion configurations.

    Args:
        num_shapes (int): Number of window shapes to generate.

    Returns:
        list[Shape]: List of window shapes.
    """
    return [
        window_1(
            outer_width=random_quantized_uniform(0.8, 1.8, 15),
            outer_height=random_quantized_uniform(0.8, 1.8, 15),
            frame_thickness=random_quantized_uniform(0.05, 0.1, 5),
            mullion_thickness=(
                random_quantized_uniform(0.02, 0.05, 3)
                if random_quantized_uniform(0, 1, 2) > 0.5
                else None
            ),  # 50% chance of having mullions
            num_horizontal_mullions=int(
                random_quantized_uniform(0, 2, 3)
            ),  # 0, 1, or 2 horizontal mullions
            num_vertical_mullions=int(
                random_quantized_uniform(0, 2, 3)
            ),  # 0, 1, or 2 vertical mullions
        )
        for _ in range(num_shapes)
    ]
