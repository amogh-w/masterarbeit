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
from graphviz import Digraph


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


def _add_graphviz_nodes(dot, shape, parent_id=None, node_id=0):
    if isinstance(shape, Shape):
        label = shape.__class__.__name__
        dot.node(
            str(node_id),
            label,
            shape="box",
            style="filled",
            fillcolor="#e0f7fa",
            fontname="Helvetica",
        )
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id))
        _, args = shape.param_tuple()
        next_id = node_id + 1
        for arg in args:
            next_id = _add_graphviz_nodes(dot, arg, node_id, next_id)
        return next_id
    else:
        dot.node(
            str(node_id),
            repr(shape),
            shape="ellipse",
            style="filled",
            fillcolor="#fff9c4",
            fontname="Helvetica",
        )
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id))
        return node_id + 1


def print_tree(shape):
    """
    Renders a shape DSL tree as a styled Graphviz Digraph for display in Jupyter.

    Args:
        shape (Shape): Root of the DSL tree.

    Returns:
        graphviz.Digraph: A styled graph object that can be rendered inline.
    """
    dot = Digraph(format="svg")
    # dot.attr(
    #     rankdir="TB",
    #     fontname="Helvetica",
    #     fontsize="10",
    #     dpi="150",
    #     nodesep="0.4",
    #     ranksep="0.6",
    #     concentrate="true",
    # )
    # dot.attr(size="")  # allow Graphviz to autosize
    _add_graphviz_nodes(dot, shape)
    return dot
