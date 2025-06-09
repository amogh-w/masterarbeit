"""
Node definitions for the shape DSL (Domain-Specific Language).

This module defines symbolic shape classes such as Rect, Move, Union, SymTrans, and SymRef,
which allow hierarchical composition and transformation of shapes in a structured way.

Each shape subclass of `Shape` implements:
- `__str__`: for readable, indented serialization
- `get_box_list()`: for converting symbolic shapes to concrete `Box` representations
- `param_tuple()`: for reconstruction or serialization of the shape tree

Shapes are composed in a tree structure and can be evaluated into lists of Box objects for
visualization or spatial analysis.

Dependencies:
- Uses PyTorch tensors for geometric computation.
- Depends on the `Box` class for geometric output.
- Uses `left_pad` for formatted string output.
"""

from abstractions.dsl.core import Shape, left_pad
from abstractions.primitives.shapes import Box

import torch


class Rect(Shape):
    """
    A rectangle shape defined by width and height.

    Args:
        s_x (float): Width of the rectangle.
        s_y (float): Height of the rectangle.

    Returns:
        A box centered at origin with the specified scale.
    """

    def __init__(self, s_x: float, s_y: float):
        super().__init__(children=[])
        self.s_x = s_x
        self.s_y = s_y

    def __str__(self):
        return f"Rect(\n    {self.s_x:.3f},\n    {self.s_y:.3f}\n)"

    def get_box_list(self) -> list[Box]:
        return [
            Box(
                center=torch.tensor([0.0, 0.0]),
                scale=torch.tensor([self.s_x, self.s_y]),
            )
        ]

    def param_tuple(self):
        return Rect, (self.s_x, self.s_y)


class Move(Shape):
    """
    A shape representing a translation of a child shape.

    Args:
        child (Shape): The shape to be moved.
        t_x (float): Translation along the x-axis.
        t_y (float): Translation along the y-axis.

    Returns:
        The translated shape's boxes shifted accordingly.
    """

    def __init__(self, child: Shape, t_x: float, t_y: float):
        super().__init__(children=[child])
        self.t_x = t_x
        self.t_y = t_y

    def __str__(self):
        return f"Move(\n    {left_pad(str(self.children[0]), '    ', 1)},\n    {self.t_x:.3f},\n    {self.t_y:.3f}\n)"

    def get_box_list(self) -> list[Box]:
        child_boxes = self.children[0].get_box_list()

        for box in child_boxes:
            box.center[0] += self.t_x
            box.center[1] += self.t_y

        return child_boxes

    def param_tuple(self):
        return Move, (self.children[0], self.t_x, self.t_y)


class Union(Shape):
    """
    A shape representing the union of two child shapes.

    Args:
        child1 (Shape): The first child shape.
        child2 (Shape): The second child shape.

    Returns:
        Combined list of boxes from both child shapes.
    """

    def __init__(self, child1: Shape, child2: Shape):
        super().__init__(children=[child1, child2])

    def __str__(self):
        return f"Union(\n    {left_pad(str(self.children[0]), '    ', 1)},\n    {left_pad(str(self.children[1]), '    ', 1)}\n)"

    def get_box_list(self) -> list[Box]:
        child_boxes1 = self.children[0].get_box_list()
        child_boxes2 = self.children[1].get_box_list()
        return child_boxes1 + child_boxes2

    def param_tuple(self):
        return Union, (self.children[0], self.children[1])


class SymTrans(Shape):
    """
    A shape representing symmetric translation of a child shape along an axis.

    Args:
        child (Shape): The shape to be symmetrically translated.
        axis (str): Axis along which to translate ('x' or 'y').
        dist (float): Total distance of translation.
        degree (int): Number of translated copies including the original.

    Returns:
        List of boxes including the original and translated copies.
    """

    def __init__(self, child: Shape, axis: str, dist: float, degree: int):
        super().__init__(children=[child])
        self.axis = axis
        self.dist = dist
        self.degree = degree

    def __str__(self):
        return f"SymTrans(\n    {left_pad(str(self.children[0]), '    ', 1)},\n    {self.axis},\n    {self.dist:.3f},\n    {self.degree}\n)"

    def get_box_list(self) -> list[Box]:
        child_boxes = self.children[0].get_box_list()
        copies = []
        dt = (self.dist / (self.degree - 1)) * (
            torch.tensor([1.0, 0.0]) if self.axis == "x" else torch.tensor([0.0, 1.0])
        )

        for box in child_boxes:
            for _ in range(self.degree - 1):
                copies.append(Box(center=box.center + dt, scale=box.scale))

        return child_boxes + copies

    def param_tuple(self):
        return SymTrans, (self.children[0], self.axis, self.dist, self.degree)


class SymRef(Shape):
    """
    A shape representing symmetric reflection of a child shape about an axis.

    Args:
        child (Shape): The shape to be reflected.
        axis (str): Axis about which to reflect ('x' or 'y').

    Returns:
        List of boxes including the original and reflected copies.
    """

    def __init__(self, child: Shape, axis: str):
        super().__init__(children=[child])
        self.axis = axis

    def __str__(self):
        return f"SymRef(\n    {left_pad(str(self.children[0]), '    ', 1)},\n    {self.axis}\n)"

    def get_box_list(self) -> list[Box]:
        child_boxes = self.children[0].get_box_list()
        copies = []
        dt = (
            torch.tensor([-1.0, 1.0]) if self.axis == "x" else torch.tensor([1.0, -1.0])
        )

        for box in child_boxes:
            copies.append(Box(center=dt * box.center, scale=box.scale))

        return child_boxes + copies

    def param_tuple(self):
        return SymRef, (self.children[0], self.axis)
