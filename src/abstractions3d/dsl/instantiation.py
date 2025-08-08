"""
3D Shape instantiation logic for the shape DSL.

This module defines the `instantiate_3d` function, which takes a structural type description
(`type_list`) and a corresponding set of parameters (`param_list`) to recursively construct
an abstract 3D shape expression tree.

Each entry in `type_list` corresponds to a shape node or primitive type (e.g., `Rect3D`, `Move3D`, `float`),
and the recursion pattern ensures correct ordering of arguments during instantiation.

This function is the inverse of `param_tuple()` on 3D shape nodes and enables expression reconstruction.
"""

from abstractions3d.dsl.core import Shape3D
from abstractions3d.dsl.nodes import Cube, Move3D, Union3D, SymTrans3D, SymRef3D


def instantiate_3d(type_list: list[type], param_list: list):
    """
    Instantiates a 3D shape expression from a list of types and corresponding parameters.

    Args:
        type_list (list[type]): A list describing the expected structure of the shape.
        param_list (list): A list of parameters used to instantiate the shape.

    Returns:
        Shape: The instantiated 3D shape object.
    """

    def _instantiate(type_list: list[type], param_list: list):
        token = type_list.pop(0)

        if issubclass(token, Cube):
            w = _instantiate(type_list, param_list)
            h = _instantiate(type_list, param_list)
            d = _instantiate(type_list, param_list)
            return Cube(w, h, d)
        elif issubclass(token, Move3D):
            child = _instantiate(type_list, param_list)
            t_x = _instantiate(type_list, param_list)
            t_y = _instantiate(type_list, param_list)
            t_z = _instantiate(type_list, param_list)
            return Move3D(child, t_x, t_y, t_z)
        elif issubclass(token, Union3D):
            child1 = _instantiate(type_list, param_list)
            child2 = _instantiate(type_list, param_list)
            return Union3D(child1, child2)
        elif issubclass(token, SymTrans3D):
            child = _instantiate(type_list, param_list)
            axis = _instantiate(type_list, param_list)
            dist = _instantiate(type_list, param_list)
            degree = _instantiate(type_list, param_list)
            return SymTrans3D(child, axis, dist, degree)
        elif issubclass(token, SymRef3D):
            child = _instantiate(type_list, param_list)
            axis = _instantiate(type_list, param_list)
            return SymRef3D(child, axis)
        elif issubclass(token, float):
            return float(param_list.pop(0))
        elif issubclass(token, int):
            return int(param_list.pop(0))
        elif issubclass(token, str):
            return str(param_list.pop(0))
        elif issubclass(token, Shape3D):
            return param_list.pop(0)
        else:
            raise ValueError(f"Unknown token: {token}")

    return _instantiate(type_list.copy(), param_list.copy())