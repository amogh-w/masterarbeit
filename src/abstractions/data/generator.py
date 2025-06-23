"""
generator.py

Main entry point to generate synthetic DSL-based shape datasets.
"""

from .bookshelf_1 import random_bookshelves_1
from .chair_1 import random_chairs_1
from .chair_2 import random_chairs_2
from .chair_3 import random_chairs_3
from .lamp_1 import random_lamps_1
from .table_1 import random_tables_1
from .square import random_squares
from .random_tree import random_shapes


def generate_dataset(kind: str, num_shapes: int):
    """
    Dispatches to the appropriate generator based on kind.

    Args:
        kind (str): One of ['chair_1', 'chair_2', 'table_1', 'square', 'random']
        num_shapes (int): Number of shape programs to generate

    Returns:
        list[Shape]: List of shape programs
    """
    match kind:
        case "bookshelf_1":
            return random_bookshelves_1(num_shapes)
        case "chair_1":
            return random_chairs_1(num_shapes)
        case "chair_2":
            return random_chairs_2(num_shapes)
        case "chair_3":
            return random_chairs_3(num_shapes)
        case "lamp_1":
            return random_lamps_1(num_shapes)
        case "table_1":
            return random_tables_1(num_shapes)
        case "square":
            return random_squares(num_shapes)
        case "random":
            return random_shapes(num_shapes)
        case _:
            raise ValueError(f"Unknown shape kind: {kind}")
