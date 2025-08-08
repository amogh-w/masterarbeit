"""
Core components for the 3D shape DSL (Domain-Specific Language).

This module defines the base `Shape3D` class, which serves as the abstract superclass for all
symbolic 3D shapes used in the DSL system. Shapes support recursive composition through their
`children` attribute and provide stubs for methods like `get_box3d_list()` and `param_tuple()`.

Also included is the `left_pad` utility function for formatting multiline string representations.

Typical use:
    class Cube(Shape3D):
        def __init__(self, w, h, d):
            super().__init__([])
            self.w = w
            self.h = h
            self.d = d

Dependencies:
- Uses the `Box3D` class for shape evaluation.
"""

from __future__ import annotations

from abstractions3d.primitives.shapes import Box3D


class Shape3D:
    """
    Base class for all 3D shapes in the DSL.

    Attributes:
        children (list[Shape3D]): Child shapes used in composition.
    """

    def __init__(self, children: list[Shape3D]):
        self.children = children

    def __str__(self):
        return "Shape3D"

    def get_box3d_list(self) -> list[Box3D]:
        """
        Returns a list of Box3D instances representing the shape and its children.

        Returns:
            list[Box3D]: Boxes corresponding to this shape and its children.
        """
        pass

    def param_tuple(self):
        """
        Returns a tuple of parameters defining the shape (e.g., width, height, depth).
        """
        pass


def left_pad(string: str, pad: str, n: int) -> str:
    """
    Indents each line of a multiline string by a given number of padding characters.

    Args:
        string (str): The string to be indented.
        pad (str): Padding character(s).
        n (int): Number of times to apply the padding.

    Returns:
        str: Indented multiline string.
    """
    return "\n".join([n * pad + s for s in string.split("\n")])