"""
dsl.py

This module defines a domain-specific language (DSL) for compositional geometric abstraction.
It includes shape classes such as Rect, Move, Union, SymTrans, and SymRef, each supporting
recursive composition and evaluation to generate a list of geometric boxes.

Additionally, it supports instantiation from abstract parameter lists and includes an Abstraction
class that can expand itself using a neural model decoder.
"""

from __future__ import annotations

import torch
from torch import nn

from abstractions.primitives import Box

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
        return [Box(center=torch.tensor([0.0, 0.0]), scale=torch.tensor([self.s_x, self.s_y]))]

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
        dt = (self.dist / (self.degree - 1)) * (torch.tensor([1.0, 0.0]) if self.axis == 'x' else torch.tensor([0.0, 1.0]))

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
        dt = (torch.tensor([-1.0, 1.0]) if self.axis == 'x' else torch.tensor([1.0, -1.0]))

        for box in child_boxes:
            copies.append(Box(center=dt * box.center, scale=box.scale))

        return child_boxes + copies

    def param_tuple(self):
        return SymRef, (self.children[0], self.axis)


class Abstraction(Shape):
    """
    A shape abstraction that can expand itself using a neural model decoder.

    Args:
        type_list (list[type]): List of types describing the shape structure.
        float_parameters (list[float]): List of float parameters used for expansion.
        other_parameters (list): List of other parameters including shapes and strings.
        model (nn.Module): Neural model with a decoder to expand float parameters.

    Returns:
        Expanded concrete shape after decoding.
    """
    def __init__(self, type_list: list[type], float_parameters: list[float], other_parameters: list, model: nn.Module):
        super().__init__(children=[child for child in other_parameters if isinstance(child, Shape)])
        self.type_list = type_list
        self.float_parameters = float_parameters
        self.other_parameters = other_parameters
        self.model = model

    def __str__(self):
        output = "Abstraction(\n"

        for child in self.children:
            output += f"    {left_pad(str(child), '    ', 1)},\n"

        for p in self.float_parameters:
            output += f"    {p:.3f},\n"

        for p in self.other_parameters:
            if not isinstance(p, Shape):
                output += f"    {p},\n"

        output += ")"
        return output

    def expand(self) -> Shape:
        """
        Expands the abstract parameters into a concrete shape using the decoder from the neural model.

        Returns:
            Shape: The expanded, instantiated shape.
        """
        self.model.eval()
        decoder = self.model.decoder
        decoder_input = torch.tensor(self.float_parameters, dtype=torch.float32)[None, :]
        decoder_output = decoder(decoder_input)
        expanded_float_parameters = decoder_output[0, :].tolist()
        full_parameter_list = []
        i_floats = 0
        i_others = 0

        for type in self.type_list:
            if type==float:
                full_parameter_list.append(expanded_float_parameters[i_floats])
                i_floats += 1
            elif type==int or type==str or type==Shape:
                full_parameter_list.append(self.other_parameters[i_others])
                i_others += 1

        expanded_shape = instantiate(self.type_list, full_parameter_list)
        return expanded_shape

    def get_box_list(self) -> list[Box]:
        return self.expand().get_box_list()

    def param_tuple(self):
        return Abstraction, (self.type_list, self.float_parameters, self.other_parameters, self.model)

