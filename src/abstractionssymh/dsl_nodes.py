"""dsl_nodes.py

Definitions of DSL node types for 3D shape construction, including Box,
Transformations (Scale, Rotate, Translate), Unions, and Symmetry operations
(reflection, rotation, translation).

All node classes provide two key methods:
-   `expand()`: Recursively evaluates the tree and returns a list of
    dictionaries, each representing a primitive box's geometry (center,
    lengths, quaternion, label).
-   `serialize()`: Returns a tuple describing the node's type, its
    numerical parameters, and its children. This is used for
    tree analysis.

Classes
-------
- Box
- Scale
- Rotate
- Translate
- Union
- Symmetry (base class)
- SymRef
- SymRot
- SymTrans
"""

import textwrap
import numpy as np
import copy
from scipy.spatial.transform import Rotation


def _format_vec(vec, precision=3):
    """Format a vector or number as a string with given precision.

    Parameters
    ----------
    vec : list or np.ndarray or float or int
        The vector or single numeric value to format.
    precision : int, optional
        The number of decimal places to use. Default is 3.

    Returns
    -------
    str
        Formatted string representation of the vector or number.
    """
    if isinstance(vec, (list, np.ndarray)):
        return f"[{', '.join(f'{x:.{precision}f}' for x in vec)}]"
    else:
        return f"{vec:.{precision}f}"


class Box:
    """Represents a primitive unit cube with a label.
    
    This is a terminal node in the DSL tree.

    Attributes
    ----------
    label : int
        An integer ID representing the part category (e.g., seat, leg).
    """

    def __init__(self, label):
        """Initialize the Box.

        Parameters
        ----------
        label : int
            Integer label for the box.
        """
        self.label = label

    def __str__(self):
        """Return a simple string representation."""
        return f"Box(label={self.label})"

    def expand(self):
        """Return the geometry for this single unit box.

        Returns
        -------
        list[dict]
            A list containing a single dictionary representing a
            unit cube at the origin. The dictionary keys are:
            "center", "lengths", "quaternion", "label_id".
        """
        return [
            {
                "center": np.array([0.0, 0.0, 0.0]),
                "lengths": np.array([1.0, 1.0, 1.0]),
                "quaternion": np.array([0.0, 0.0, 0.0, 1.0]),
                "label_id": self.label,
            }
        ]

    def serialize(self):
        """Return a tuple for DSL processing.

        Returns
        -------
        tuple
            A tuple in the format: `(Class, (numeric_params, children))`.
            For Box, this is `(Box, ([], [self.label]))`.
        """
        return (Box, ([], [self.label]))


class Scale:
    """Applies an axis-aligned scaling transformation to a child node.

    Attributes
    ----------
    child : object
        The child DSL node to be scaled.
    lengths : np.ndarray
        The (x, y, z) scaling factors.
    """

    def __init__(self, child, lengths):
        """Initialize the Scale node.

        Parameters
        ----------
        child : object
            DSL node to scale.
        lengths : array_like
            Scaling factors for x, y, z axes (e.g., [sx, sy, sz]).
        """
        self.child = child
        self.lengths = np.array(lengths)

    def __str__(self):
        """Return a formatted, hierarchical string representation."""
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Scale(lengths={_format_vec(self.lengths)})\n{child_str}"

    def expand(self):
        """Expand the child and apply scaling to all resulting boxes.

        Scales both the `lengths` and `center` of each box.

        Returns
        -------
        list[dict]
            A list of transformed box dictionaries.
        """
        child_boxes = self.child.expand()
        for box in child_boxes:
            box["lengths"] *= self.lengths
            box["center"] *= self.lengths
        return child_boxes

    def serialize(self):
        """Return a tuple for DSL processing.

        Returns
        -------
        tuple
            A tuple in the format: `(Scale, (self.lengths, [self.child]))`.
        """
        return (Scale, (self.lengths.tolist(), [self.child]))


class Rotate:
    """Applies a rotation to a child node using a quaternion.

    Attributes
    ----------
    child : object
        The child DSL node to be rotated.
    quaternion : np.ndarray
        The (x, y, z, w) quaternion representing the rotation.
    """

    def __init__(self, child, quaternion):
        """Initialize the Rotate node.

        Parameters
        ----------
        child : object
            DSL node to rotate.
        quaternion : array_like
            (x, y, z, w) quaternion for the rotation.
        """
        self.child = child
        self.quaternion = np.array(quaternion)

    def __str__(self):
        """Return a formatted, hierarchical string representation."""
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Rotate(quat={_format_vec(self.quaternion, precision=4)})\n{child_str}"

    def expand(self):
        """Expand the child and apply rotation to all resulting boxes.

        Applies the rotation to each box's `center` and composes it
        with each box's existing `quaternion`.

        Returns
        -------
        list[dict]
            A list of transformed box dictionaries.
        """
        child_boxes = self.child.expand()
        op_rotation = Rotation.from_quat(self.quaternion)
        for box in child_boxes:
            box_rotation = Rotation.from_quat(box["quaternion"])
            box["quaternion"] = (op_rotation * box_rotation).as_quat()
            box["center"] = op_rotation.apply(box["center"])
        return child_boxes

    def serialize(self):
        """Return a tuple for DSL processing.

        Returns
        -------
        tuple
            A tuple in the format: `(Rotate, (self.quaternion, [self.child]))`.
        """
        return (Rotate, (self.quaternion.tolist(), [self.child]))


class Translate:
    """Applies a translation to a child node.

    Attributes
    ----------
    child : object
        The child DSL node to be translated.
    center : np.ndarray
        The (x, y, z) translation vector.
    """

    def __init__(self, child, center):
        """Initialize the Translate node.

        Parameters
        ----------
        child : object
            DSL node to translate.
        center : array_like
            (x, y, z) translation vector.
        """
        self.child = child
        self.center = np.array(center)

    def __str__(self):
        """Return a formatted, hierarchical string representation."""
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Translate(center={_format_vec(self.center)})\n{child_str}"

    def expand(self):
        """Expand the child and apply translation to all resulting boxes.

        Adds the translation vector to each box's `center`.

        Returns
        -------
        list[dict]
            A list of transformed box dictionaries.
        """
        child_boxes = self.child.expand()
        for box in child_boxes:
            box["center"] += self.center
        return child_boxes

    def serialize(self):
        """Return a tuple for DSL processing.

        Returns
        -------
        tuple
            A tuple in the format: `(Translate, (self.center, [self.child]))`.
        """
        return (Translate, (self.center.tolist(), [self.child]))


class Union:
    """Represents the union of two child nodes.

    This node joins two subtrees.

    Attributes
    ----------
    left : object
        The first child DSL node.
    right : object
        The second child DSL node.
    """

    def __init__(self, left, right):
        """Initialize the Union node.

        Parameters
        ----------
        left : object
            The first child DSL node.
        right : object
            The second child DSL node.
        """
        self.left = left
        self.right = right

    def __str__(self):
        """Return a formatted, hierarchical string representation."""
        left_str = textwrap.indent(str(self.left), "    ")
        right_str = textwrap.indent(str(self.right), "    ")
        return f"Union(\n{left_str},\n{right_str}\n)"

    def expand(self):
        """Expand both children and return their combined list of boxes.

        Returns
        -------
        list[dict]
            A list containing all box dictionaries from both the
            left and right subtrees.
        """
        return self.left.expand() + self.right.expand()

    def serialize(self):
        """Return a tuple for DSL processing.

        Returns
        -------
        tuple
            A tuple in the format: `(Union, ([], [self.left, self.right]))`.
        """
        return (Union, ([], [self.left, self.right]))


class Symmetry:
    """Base class for symmetry operations.

    Attributes
    ----------
    child : object
        The child DSL node to which symmetry is applied.
    """
    def __init__(self, child):
        """Initialize the base Symmetry node.

        Parameters
        ----------
        child : object
            The child DSL node.
        """
        self.child = child


class SymRef(Symmetry):
    """Reflects a child node across a defined plane.

    Generates the original child's geometry plus a reflected copy.

    Attributes
    ----------
    plane : np.ndarray
        The (x, y, z) normal vector of the reflection plane.
    point_on_plane : np.ndarray
        An (x, y, z) point that lies on the reflection plane.
    """

    def __init__(self, child, plane_normal, point_on_plane):
        """Initialize the reflection symmetry.

        Parameters
        ----------
        child : object
            The child DSL node to reflect.
        plane_normal : array_like
            (x, y, z) normal vector of the reflection plane.
        point_on_plane : array_like
            (x, y, z) point on the reflection plane.
        """
        super().__init__(child)
        self.plane = np.array(plane_normal)
        self.point_on_plane = np.array(point_on_plane)

    def __str__(self):
        """Return a formatted, hierarchical string representation."""
        info = (
            f"SymRef(\n"
            f"    plane={_format_vec(self.plane)},\n"
            f"    point_on_plane={_format_vec(self.point_on_plane)}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"

    def expand(self):
        """Expand the child and add reflected copies of all boxes.

        Calculates the reflected `center` and `quaternion` for each
        box from the child's expansion.

        Returns
        -------
        list[dict]
            A list containing the original boxes plus all
            newly generated reflected boxes.
        """
        child_boxes = self.child.expand()
        generated_boxes = []
        plane_normal = self.plane / (np.linalg.norm(self.plane) + 1e-8)

        for box in child_boxes:
            reflected_box = copy.deepcopy(box)
            vec_to_plane = box["center"] - self.point_on_plane
            dist = np.dot(vec_to_plane, plane_normal)
            reflected_box["center"] = box["center"] - 2 * dist * plane_normal

            R_orig = Rotation.from_quat(box["quaternion"]).as_matrix()
            M_reflect = np.identity(3) - 2 * np.outer(plane_normal, plane_normal)
            R_new = M_reflect @ R_orig
            if np.linalg.det(R_new) < 0:
                # Ensure a valid rotation matrix (no inversion)
                R_new[:, 0] *= -1
            reflected_box["quaternion"] = Rotation.from_matrix(R_new).as_quat()
            generated_boxes.append(reflected_box)

        return child_boxes + generated_boxes

    def serialize(self):
        """Return a tuple for DSL processing.

        Returns
        -------
        tuple
            A tuple in the format: `(SymRef, (params, [self.child]))`,
            where `params` is a list of plane normal and point components.
        """
        params = self.plane.tolist() + self.point_on_plane.tolist()
        return (SymRef, (params, [self.child]))


class SymRot(Symmetry):
    """Applies n-fold rotational symmetry around an axis and center.

    Generates `n` total copies (the original + `n-1` rotated copies)
    of the child's geometry.

    Attributes
    ----------
    axis : np.ndarray
        (x, y, z) vector defining the axis of rotation.
    center : np.ndarray
        (x, y, z) point defining the center of rotation.
    n : int
        The total number of copies (e.g., 4 for 90-degree steps).
    """

    def __init__(self, child, axis, center, n_fold):
        """Initialize the rotational symmetry.

        Parameters
        ----------
        child : object
            The child DSL node to rotate.
        axis : array_like
            (x, y, z) axis of rotation.
        center : array_like
            (x, y, z) center of rotation.
        n_fold : int
            The total number of copies (e.g., 4 for 90-degree steps).
        """
        super().__init__(child)
        self.axis = np.array(axis)
        self.center = np.array(center)
        self.n = n_fold

    def __str__(self):
        """Return a formatted, hierarchical string representation."""
        info = (
            f"SymRot(\n"
            f"    axis={_format_vec(self.axis)},\n"
            f"    center={_format_vec(self.center)},\n"
            f"    n={self.n}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"

    def expand(self):
        """Expand the child and add `n-1` rotated copies.

        Calculates the rotated `center` and `quaternion` for each
        copy of each box.

        Returns
        -------
        list[dict]
            A list containing the original boxes plus all
            newly generated rotated boxes.
        """
        child_boxes = self.child.expand()
        generated_boxes = []
        axis = self.axis / (np.linalg.norm(self.axis) + 1e-8)

        for i in range(1, self.n):
            angle = 2 * np.pi * i / self.n
            symmetry_rot = Rotation.from_rotvec(angle * axis)
            for box in child_boxes:
                rotated_box = copy.deepcopy(box)
                vec_from_center = box["center"] - self.center
                rotated_box["center"] = self.center + symmetry_rot.apply(
                    vec_from_center
                )
                original_rot = Rotation.from_quat(box["quaternion"])
                rotated_box["quaternion"] = (symmetry_rot * original_rot).as_quat()
                generated_boxes.append(rotated_box)

        return child_boxes + generated_boxes

    def serialize(self):
        """Return a tuple for DSL processing.

        Returns
        -------
        tuple
            A tuple in the format: `(SymRot, (params, [self.child, self.n]))`,
            where `params` is a list of axis and center components.
        """
        params = self.axis.tolist() + self.center.tolist()
        return (SymRot, (params, [self.child, self.n]))


class SymTrans(Symmetry):
    """Applies n-fold translational symmetry along a vector.

    Generates `n` total copies (the original + `n-1` translated copies)
    of the child's geometry, each offset by a step of `end_point`.

    Attributes
    ----------
    end_point : np.ndarray
        (x, y, z) vector representing a single translation step.
    n : int
        The total number of copies.
    """

    def __init__(self, child, end_point, n_fold):
        """Initialize the translational symmetry.

        Parameters
        ----------
        child : object
            The child DSL node to translate.
        end_point : array_like
            (x, y, z) vector for a single translation step.
        n_fold : int
            The total number of copies.
        """
        super().__init__(child)
        self.end_point = np.array(end_point)
        self.n = n_fold

    def __str__(self):
        """Return a formatted, hierarchical string representation."""
        info = (
            f"SymTrans(\n"
            f"    end_point={_format_vec(self.end_point)},\n"
            f"    n={self.n}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"

    def expand(self):
        """Expand the child and add `n-1` translated copies.

        Calculates the new `center` for each copy of each box.
        The `quaternion` and `lengths` remain unchanged.

        Returns
        -------
        list[dict]
            A list containing the original boxes plus all
            newly generated translated boxes.
        """
        child_boxes = self.child.expand()
        if not child_boxes:
            return []

        generated_boxes = []
        for i in range(1, self.n):
            total_translation = i * self.end_point
            for box in child_boxes:
                translated_box = copy.deepcopy(box)
                translated_box["center"] = box["center"] + total_translation
                generated_boxes.append(translated_box)

        return child_boxes + generated_boxes

    def serialize(self):
        """Return a tuple for DSL processing.

        Returns
        -------
        tuple
            A tuple in the format: `(SymTrans, (params, [self.child, self.n]))`,
            where `params` is a list of the end_point components.
        """
        params = self.end_point.tolist()
        return (SymTrans, (params, [self.child, self.n]))