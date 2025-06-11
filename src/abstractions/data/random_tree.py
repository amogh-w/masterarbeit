import random
from abstractions.data.utils import random_quantized_uniform
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union, SymRef, Move, Rect, SymTrans


def random_shape(max_nodes: int = 32):
    """
    Recursively generates a random shape composed of primitive and composite operations.

    Args:
        max_nodes (int): Maximum number of nodes in the shape expression tree.

    Returns:
        Shape: A randomly composed shape.
    """
    op_index = random.randint(0, 4)

    if max_nodes <= 1 or op_index == 0:
        # Rect
        return Rect(
            random_quantized_uniform(0.5, 2.0, 15),
            random_quantized_uniform(0.5, 2.0, 15),
        )
    elif op_index == 1:
        # Move
        return Move(
            random_shape(max_nodes - 1),
            random_quantized_uniform(0.5, 2.0, 40),
            random_quantized_uniform(0.5, 2.0, 40),
        )
    elif op_index == 2:
        # Union
        return Union(random_shape(max_nodes // 2), random_shape((max_nodes - 1) // 2))
    elif op_index == 3:
        # SymTrans
        return SymTrans(
            random_shape(max_nodes - 1),
            random.choice(["x", "y"]),
            random_quantized_uniform(-2.0, 2.0, 40),
            random.randint(2, 5),
        )
    elif op_index == 4:
        # SymRef
        return SymRef(random_shape(max_nodes - 1), random.choice(["x", "y"]))
    else:
        raise ValueError("Invalid operation index")


def random_shapes(num_shapes: int, max_nodes: int = 32) -> list[Shape]:
    """
    Generates a list of random shapes.

    Args:
        num_shapes (int): Number of shapes to generate.
        max_nodes (int): Maximum nodes per shape.

    Returns:
        list[Shape]: A list of randomly composed shapes.
    """
    return [random_shape(max_nodes) for _ in range(num_shapes)]
