import textwrap
import numpy as np


# Helper to format vectors for printing
def _format_vec(vec, precision=3):
    """
    Formats a list of numbers into a clean string, handling both single numbers
    and lists of numbers.
    """
    if isinstance(vec, (list, np.ndarray)):
        return f"[{', '.join(f'{x:.{precision}f}' for x in vec)}]"
    else:
        # Handle single float/int values
        return f"{vec:.{precision}f}"


# --- LEAF NODE (always a unit cube at the origin) ---
class Box:
    def __init__(self, label: int):
        self.label = label

    def __str__(self):
        # The string representation now only includes the label ID.
        return f"Box(label={self.label})"


# --- NEW TRANSFORMATION NODES ---
class Scale:
    def __init__(self, child, lengths):
        self.child = child
        self.lengths = lengths

    def __str__(self):
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Scale(lengths={_format_vec(self.lengths)})\n{child_str}"


class Rotate:
    def __init__(self, child, quaternion):
        self.child = child
        self.quaternion = quaternion

    def __str__(self):
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Rotate(quat={_format_vec(self.quaternion, precision=4)})\n{child_str}"


class Translate:
    def __init__(self, child, center):
        self.child = child
        self.center = center

    def __str__(self):
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Translate(center={_format_vec(self.center)})\n{child_str}"


# --- INTERNAL NODES ---
class Union:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        left_str = textwrap.indent(str(self.left), "    ")
        right_str = textwrap.indent(str(self.right), "    ")
        return f"Union(\n{left_str},\n{right_str}\n)"


class Symmetry:
    def __init__(self, child):
        self.child = child


class SymRef(Symmetry):
    def __init__(self, child, plane_normal, point_on_plane):
        super().__init__(child)
        self.plane = plane_normal
        self.point_on_plane = point_on_plane

    def __str__(self):
        info = (
            f"SymRef(\n"
            f"    plane={_format_vec(self.plane)},\n"
            f"    point_on_plane={_format_vec(self.point_on_plane)}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"


class SymRot(Symmetry):
    def __init__(self, child, axis, center, n_fold: int):
        super().__init__(child)
        self.axis = axis
        self.center = center
        self.n = n_fold

    def __str__(self):
        info = (
            f"SymRot(\n"
            f"    axis={_format_vec(self.axis)},\n"
            f"    center={_format_vec(self.center)},\n"
            f"    n={self.n}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"


class SymTrans(Symmetry):
    def __init__(self, child, end_point, n_fold: int):
        super().__init__(child)
        self.end_point = end_point
        self.n = n_fold

    def __str__(self):
        info = (
            f"SymTrans(\n"
            f"    end_point={_format_vec(self.end_point)},\n"
            f"    n={self.n}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"
