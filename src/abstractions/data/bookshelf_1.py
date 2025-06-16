from functools import reduce
from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.nodes import Union, Move, Rect


def nest_union(shapes):
    return reduce(lambda a, b: Union(a, b), shapes)


def bookshelf_1(
    width,
    height,
    shelf_thickness,
    num_shelves,
    num_dividers,
    divider_thickness,
):
    """
    Generates a bookshelf shape with shelves and vertical dividers.

    Parameters
    ----------
    width : float
        Total width of the bookshelf.
    height : float
        Total height of the bookshelf.
    shelf_thickness : float
        Thickness of each horizontal shelf.
    num_shelves : int
        Number of horizontal shelves.
    num_dividers : int
        Number of vertical dividers between shelves.
    divider_thickness : float
        Thickness of each vertical divider.

    Returns
    -------
    Shape
        Composite bookshelf shape.
    """
    # Calculate shelf spacing
    shelf_spacing = (height - shelf_thickness) / num_shelves

    # Create shelves
    shelves = []
    for i in range(num_shelves + 1):
        y_pos = -height / 2 + i * shelf_spacing
        shelf = Move(Rect(width, shelf_thickness), 0, y_pos)
        shelves.append(shelf)

    # Create dividers
    dividers = []
    if num_dividers > 0:
        divider_spacing = width / (num_dividers + 1)
        for i in range(num_dividers):
            x_pos = -width / 2 + (i + 1) * divider_spacing
            divider = Move(Rect(divider_thickness, height), x_pos, 0)
            dividers.append(divider)

    # Union all shelves and dividers
    all_parts = nest_union(shelves)
    if dividers:
        all_parts = nest_union([all_parts] + dividers)

    return all_parts


def random_bookshelves_1(num_shapes: int):
    """
    Generates a list of random bookshelf shapes with varying dimensions.

    Parameters
    ----------
    num_shapes : int
        Number of bookshelf shapes to generate.

    Returns
    -------
    list[Shape]
        List of generated bookshelf shapes.
    """
    return [
        bookshelf_1(
            width=random_quantized_uniform(0.5, 1.5, 20),
            height=random_quantized_uniform(1.0, 2.5, 30),
            shelf_thickness=random_quantized_uniform(0.03, 0.1, 5),
            num_shelves=random_quantized_uniform(2, 6, 5),
            num_dividers=random_quantized_uniform(0, 4, 5),
            divider_thickness=random_quantized_uniform(0.02, 0.1, 5),
        )
        for _ in range(num_shapes)
    ]
