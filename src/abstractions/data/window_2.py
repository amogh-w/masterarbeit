from abstractions.dsl.nodes import Rect, Move, Union
from abstractions.dsl.core import Shape
from abstractions.data.utils import random_quantized_uniform


def window_2(width, height, frame_thickness, spacing):
    """
    Constructs a window with a rectangular frame and 2x2 grid of panes.

    Parameters
    ----------
    width : float
        Total width of the window.
    height : float
        Total height of the window.
    frame_thickness : float
        Thickness of the window frame.
    spacing : float
        Spacing between panes.

    Returns
    -------
    Shape
        A DSL shape representing a framed window with 2x2 grid panes.
    """
    outer_frame = Rect(width, height)

    # Compute pane dimensions
    inner_width = width - 2 * frame_thickness
    inner_height = height - 2 * frame_thickness

    pane_width = (inner_width - spacing) / 2
    pane_height = (inner_height - spacing) / 2

    # Top-left pane
    pane1 = Move(
        Rect(pane_width, pane_height),
        -spacing / 2 - pane_width / 2,
        spacing / 2 + pane_height / 2,
    )

    # Top-right pane
    pane2 = Move(
        Rect(pane_width, pane_height),
        spacing / 2 + pane_width / 2,
        spacing / 2 + pane_height / 2,
    )

    # Bottom-left pane
    pane3 = Move(
        Rect(pane_width, pane_height),
        -spacing / 2 - pane_width / 2,
        -(spacing / 2 + pane_height / 2),
    )

    # Bottom-right pane
    pane4 = Move(
        Rect(pane_width, pane_height),
        spacing / 2 + pane_width / 2,
        -(spacing / 2 + pane_height / 2),
    )

    panes = Union(Union(pane1, pane2), Union(pane3, pane4))
    return Union(outer_frame, panes)


def random_windows_2(n):
    return [
        window_2(
            width=random_quantized_uniform(0.8, 1.5, 15),
            height=random_quantized_uniform(1.0, 2.0, 15),
            frame_thickness=random_quantized_uniform(0.05, 0.1, 5),
            spacing=random_quantized_uniform(0.05, 0.15, 5),
        )
        for _ in range(n)
    ]
