"""dsl_nodes.py

Definitions of DSL node types for 3D shape construction. 
Simplified to match PartNet data: Cuboid, Translate, Rotate, Union.
"""

import textwrap
import numpy as np
from scipy.spatial.transform import Rotation


def _format_vec(vec, precision=3):
    """Format a vector or number as a string."""
    if isinstance(vec, (list, np.ndarray, tuple)):
        return f"[{', '.join(f'{x:.{precision}f}' for x in vec)}]"
    else:
        return f"{vec:.{precision}f}"


class Cuboid:
    """Represents a primitive cuboid with specific dimensions.
    
    This replaces the old 'Box' and 'Scale' nodes.
    """

    def __init__(self, size, label=0):
        """
        Parameters
        ----------
        size : array_like
            (width, height, depth) dimensions.
        label : int
            Category ID (default 0).
        """
        self.size = np.array(size)
        self.label = label

    def __str__(self):
        return f"Cuboid(size={_format_vec(self.size)})"

    def expand(self):
        """Return geometry for this cuboid."""
        return [
            {
                "center": np.array([0.0, 0.0, 0.0]),
                "lengths": self.size.copy(),
                "quaternion": np.array([0.0, 0.0, 0.0, 1.0]),
                "label_id": self.label,
            }
        ]

    def serialize(self):
        return (Cuboid, (self.size.tolist(), [self.label]))


class Translate:
    """Applies a translation to a child node."""

    def __init__(self, child, vector):
        """
        Parameters
        ----------
        child : object
            DSL node to translate.
        vector : array_like
            (x, y, z) translation vector.
        """
        self.child = child
        self.vector = np.array(vector)

    def __str__(self):
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Translate(vec={_format_vec(self.vector)})\n{child_str}"

    def expand(self):
        """Add translation vector to all child centers."""
        child_boxes = self.child.expand()
        for box in child_boxes:
            box["center"] += self.vector
        return child_boxes

    def serialize(self):
        return (Translate, (self.vector.tolist(), [self.child]))


class Rotate:
    """Applies a rotation to a child node."""

    def __init__(self, child, quaternion):
        """
        Parameters
        ----------
        child : object
            DSL node to rotate.
        quaternion : array_like
            (x, y, z, w) quaternion.
        """
        self.child = child
        self.quaternion = np.array(quaternion)

    def __str__(self):
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Rotate(quat={_format_vec(self.quaternion, precision=4)})\n{child_str}"

    def expand(self):
        """Apply rotation to child centers and orientations."""
        child_boxes = self.child.expand()
        op_rotation = Rotation.from_quat(self.quaternion)
        
        for box in child_boxes:
            # Rotate orientation
            box_rotation = Rotation.from_quat(box["quaternion"])
            box["quaternion"] = (op_rotation * box_rotation).as_quat()
            # Rotate position
            box["center"] = op_rotation.apply(box["center"])
            
        return child_boxes

    def serialize(self):
        return (Rotate, (self.quaternion.tolist(), [self.child]))


class Union:
    """Joins two subtrees."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        left_str = textwrap.indent(str(self.left), "    ")
        right_str = textwrap.indent(str(self.right), "    ")
        return f"Union(\n{left_str},\n{right_str}\n)"

    def expand(self):
        """Combine lists of boxes."""
        return self.left.expand() + self.right.expand()

    def serialize(self):
        return (Union, ([], [self.left, self.right]))