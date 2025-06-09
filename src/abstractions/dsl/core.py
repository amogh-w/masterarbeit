"""
Core components for the shape DSL (Domain-Specific Language).

This module defines the base `Shape` class, which serves as the abstract superclass for all
symbolic shapes used in the DSL system. Shapes support recursive composition through their
`children` attribute and provide stubs for methods like `get_box_list()` and `param_tuple()`.

Also included is the `left_pad` utility function for formatting multiline string representations.

Typical use:
    class Rect(Shape):
        def __init__(self, w, h):
            super().__init__([])
            self.w = w
            self.h = h

Dependencies:
- Uses the `Box` class for shape evaluation.
"""

from __future__ import annotations

from abstractions.primitives.shapes import Box


class Shape:
    """
    Base class for all shapes in the DSL.

    Attributes:
        children (list[Shape]): Child shapes used in composition.
    """

    def __init__(self, children: list[Shape]):
        self.children = children

    def __str__(self):
        return "Shape"

    def get_box_list(self) -> list[Box]:
        pass

    def param_tuple(self):
        pass


def left_pad(string, pad, n):
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


def left_pad(string, pad, n):
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
