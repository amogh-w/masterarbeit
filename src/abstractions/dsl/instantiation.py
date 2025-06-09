"""
Shape instantiation logic for the shape DSL (Domain-Specific Language).

This module defines the `instantiate` function, which takes a structural type description
(`type_list`) and a corresponding set of parameters (`param_list`) to recursively construct
an abstract shape expression tree.

Each entry in `type_list` corresponds to a shape node or primitive type (e.g., `Rect`, `Move`, `float`),
and the recursion pattern ensures correct ordering of arguments during instantiation.

This function is the inverse of `param_tuple()` on shape nodes and enables expression reconstruction.

Example:
    type_list = [Move, Rect, float, float, float, float]
    param_list = [10, 20, 1, 2]
    shape = instantiate(type_list, param_list)  # → Move(Rect(10, 20), 1, 2)
"""

from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Rect, Move, Union, SymTrans, SymRef


def instantiate(type_list: list[type], param_list: list):
    """
    Instantiates a shape expression from a list of types and corresponding parameters.

    Args:
        type_list (list[type]): A list describing the expected structure of the shape.
        param_list (list): A list of parameters used to instantiate the shape.

    Returns:
        Shape: The instantiated shape object.
    """

    def _instantiate(type_list: list[type], param_list: list):
        token = type_list.pop(0)

        if issubclass(token, Rect):
            s_x = _instantiate(type_list, param_list)
            s_y = _instantiate(type_list, param_list)
            return Rect(s_x, s_y)
        elif issubclass(token, Move):
            child = _instantiate(type_list, param_list)
            t_x = _instantiate(type_list, param_list)
            t_y = _instantiate(type_list, param_list)
            return Move(child, t_x, t_y)
        elif issubclass(token, Union):
            child1 = _instantiate(type_list, param_list)
            child2 = _instantiate(type_list, param_list)
            return Union(child1, child2)
        elif issubclass(token, SymTrans):
            child = _instantiate(type_list, param_list)
            axis = _instantiate(type_list, param_list)
            dist = _instantiate(type_list, param_list)
            degree = _instantiate(type_list, param_list)
            return SymTrans(child, axis, dist, degree)
        elif issubclass(token, SymRef):
            child = _instantiate(type_list, param_list)
            axis = _instantiate(type_list, param_list)
            return SymRef(child, axis)
        elif issubclass(token, float):
            return float(param_list.pop(0))
        elif issubclass(token, int):
            return int(param_list.pop(0))
        elif issubclass(token, str):
            return str(param_list.pop(0))
        elif issubclass(token, Shape):
            return param_list.pop(0)
        else:
            raise ValueError(f"Unknown token: {token}")

    return _instantiate(type_list.copy(), param_list.copy())
