"""
3D shape DSL nodes: Rect3D, Move3D, Union3D, SymTrans3D, SymRef3D.
"""

from abstractions3d.dsl.core import Shape3D, left_pad
from abstractions3d.primitives.shapes import Box3D
import torch
import textwrap


class Rect3D(Shape3D):
    """Rectangular box shape."""

    def __init__(self, s_x: float, s_y: float, s_z: float):
        super().__init__(children=[])
        self.s_x = s_x
        self.s_y = s_y
        self.s_z = s_z

    def __str__(self):
        params = f"{self.s_x:.3f},\n{self.s_y:.3f},\n{self.s_z:.3f}"
        indented = textwrap.indent(params, "    ")
        return f"Rect3D(\n{indented}\n)"

    def get_box3d_list(self) -> list[Box3D]:
        return [
            Box3D(
                center=torch.tensor([0.0, 0.0, 0.0]),
                scale=torch.tensor([self.s_x, self.s_y, self.s_z]),
            )
        ]

    def param_tuple(self):
        return Rect3D, (self.s_x, self.s_y, self.s_z)


class Move3D(Shape3D):
    """Translate child shape in 3D."""

    def __init__(self, child: Shape3D, t_x: float, t_y: float, t_z: float):
        super().__init__(children=[child])
        self.t_x = t_x
        self.t_y = t_y
        self.t_z = t_z

    def __str__(self):
        child_str = textwrap.indent(str(self.children[0]), "    ")
        params = f"{self.t_x:.3f},\n{self.t_y:.3f},\n{self.t_z:.3f}"
        indented_params = textwrap.indent(params, "    ")
        return f"Move3D(\n{child_str},\n{indented_params}\n)"

    def get_box3d_list(self) -> list[Box3D]:
        child_boxes = self.children[0].get_box3d_list()
        for box in child_boxes:
            box.center += torch.tensor([self.t_x, self.t_y, self.t_z])
        return child_boxes

    def param_tuple(self):
        return Move3D, (self.children[0], self.t_x, self.t_y, self.t_z)


class Union3D(Shape3D):
    """Union of two shapes."""

    def __init__(self, child1: Shape3D, child2: Shape3D):
        if not isinstance(child1, Shape3D) or not isinstance(child2, Shape3D):
            raise TypeError("Union3D expects two Shape3D instances.")
        super().__init__(children=[child1, child2])

    def __str__(self):
        child1_str = textwrap.indent(str(self.children[0]), "    ")
        child2_str = textwrap.indent(str(self.children[1]), "    ")
        return f"Union3D(\n{child1_str},\n{child2_str}\n)"

    def get_box3d_list(self) -> list[Box3D]:
        return self.children[0].get_box3d_list() + self.children[1].get_box3d_list()

    def param_tuple(self):
        return Union3D, (self.children[0], self.children[1])


class SymTrans3D(Shape3D):
    """Symmetric translation along axis."""

    def __init__(self, child: Shape3D, axis: str, dist: float, degree: int):
        super().__init__(children=[child])
        self.axis = axis
        self.dist = dist
        self.degree = degree

    def __str__(self):
        child_str = textwrap.indent(str(self.children[0]), "    ")
        params = f"{self.axis},\n{self.dist:.3f},\n{self.degree}"
        indented_params = textwrap.indent(params, "    ")
        return f"SymTrans3D(\n{child_str},\n{indented_params}\n)"

    def get_box3d_list(self) -> list[Box3D]:
        child_boxes = self.children[0].get_box3d_list()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if self.axis not in axis_map:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        idx = axis_map[self.axis]
        copies = []
        for box in child_boxes:
            for i in range(self.degree - 1):
                offset = torch.zeros(3)
                offset[idx] = self.dist * (i + 1) / (self.degree - 1)
                copies.append(Box3D(center=box.center + offset, scale=box.scale))
        return child_boxes + copies

    def param_tuple(self):
        return SymTrans3D, (self.children[0], self.axis, self.dist, self.degree)


class SymRef3D(Shape3D):
    """Symmetric reflection about axis plane."""

    def __init__(self, child: Shape3D, axis: str):
        super().__init__(children=[child])
        self.axis = axis

    def __str__(self):
        child_str = textwrap.indent(str(self.children[0]), "    ")
        return f"SymRef3D(\n{child_str},\n    {self.axis}\n)"

    def get_box3d_list(self) -> list[Box3D]:
        child_boxes = self.children[0].get_box3d_list()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if self.axis not in axis_map:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        idx = axis_map[self.axis]
        copies = []
        for box in child_boxes:
            reflected_center = box.center.clone()
            reflected_center[idx] *= -1
            copies.append(Box3D(center=reflected_center, scale=box.scale))
        return child_boxes + copies

    def param_tuple(self):
        return SymRef3D, (self.children[0], self.axis)