from __future__ import annotations
from abstractions3d.primitives.shapes import Box3D

class Shape3D:
    """Base class for 3D shapes in the DSL."""

    def __init__(self, children: list[Shape3D]):
        self.children = children

    def __str__(self):
        return "Shape3D"

    def get_box3d_list(self) -> list[Box3D]:
        return []

    def param_tuple(self):
        return ()
    
def left_pad(string: str, pad: str, n: int) -> str:
    """
    Indents each line of a multiline string by a given number of padding characters.
    """
    return "\n".join([n * pad + s for s in string.split("\n")])