"""
Shape instantiation logic for the shape DSL (Domain-Specific Language).

This module defines the `instantiate` function, which reconstructs an abstract syntax tree
representing a shape program from a linearized type description and parameter list.

The `type_list` contains the sequence of types (shape nodes and primitive types),
and `param_list` provides the corresponding parameter values in order.

The function recursively consumes these lists to build nested Shape instances.

This is effectively the inverse operation of `param_tuple()` for shape nodes,
allowing full round-trip serialization and deserialization of shape expressions.

Example:
    type_list = [Move, Rect, float, float, float, float]
    param_list = [10, 20, 1, 2]
    shape = instantiate(type_list, param_list)  # Returns Move(Rect(10, 20), 1, 2)
"""

from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Rect, Move, Union, SymTrans, SymRef


def instantiate(type_list: list[type], param_list: list):
    """
    Instantiate a shape expression from a list of types and corresponding parameters.

    Parameters
    ----------
    type_list : list[type]
        A list describing the expected structure of the shape expression.
    param_list : list
        A list of parameters corresponding to the shape's nodes.

    Returns
    -------
    Shape
        The fully constructed shape expression tree.
    """

    SHAPE_META = {
        Rect: {"args": [float, float], "constructor": Rect},
        Move: {"args": [Shape, float, float], "constructor": Move},
        Union: {"args": [Shape, Shape], "constructor": Union},
        SymTrans: {"args": [Shape, str, float, int], "constructor": SymTrans},
        SymRef: {"args": [Shape, str], "constructor": SymRef},
    }

    def _instantiate():
        token = type_list.pop(0)
        meta = SHAPE_META.get(token)
        if meta is None:
            # Handle primitives
            if token in (float, int, str):
                return token(param_list.pop(0))
            if issubclass(token, Shape):
                return param_list.pop(0)
            raise ValueError(f"Unknown token: {token}")

        args = []
        for arg_type in meta["args"]:
            if issubclass(arg_type, Shape):
                args.append(_instantiate())
            else:
                args.append(arg_type(param_list.pop(0)))
        return meta["constructor"](*args)

    return _instantiate()
