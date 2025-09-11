import textwrap
import numpy as np
import copy # <-- Import the copy module
from scipy.spatial.transform import Rotation


# Helper to format vectors for printing
def _format_vec(vec, precision=3):
    """
    Formats a list of numbers into a clean string, handling both single numbers
    and lists of numbers.
    """
    if isinstance(vec, (list, np.ndarray)):
        return f"[{', '.join(f'{x:.{precision}f}' for x in vec)}]"
    else:
        return f"{vec:.{precision}f}"


# --- LEAF NODE (always a unit cube at the origin) ---
class Box:
    def __init__(self, label: int):
        self.label = label

    def __str__(self):
        return f"Box(label={self.label})"

    def expand(self):
        """Returns the dictionary for a single unit box at the origin."""
        return [{
            "center": np.array([0.0, 0.0, 0.0]),
            "lengths": np.array([1.0, 1.0, 1.0]),
            "quaternion": np.array([0.0, 0.0, 0.0, 1.0]), # Identity
            "label_id": self.label,
        }]

    def serialize(self):
        """Serializes the node into its type, float params, and other params."""
        return (Box, ([], [self.label]))


# --- TRANSFORMATION NODES ---
class Scale:
    def __init__(self, child, lengths):
        self.child = child
        self.lengths = np.array(lengths)

    def __str__(self):
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Scale(lengths={_format_vec(self.lengths)})\n{child_str}"

    def expand(self):
        child_boxes = self.child.expand()
        for box in child_boxes:
            box["lengths"] *= self.lengths
            box["center"] *= self.lengths
        return child_boxes

    def serialize(self):
        return (Scale, (self.lengths.tolist(), [self.child]))


class Rotate:
    def __init__(self, child, quaternion):
        self.child = child
        self.quaternion = np.array(quaternion)

    def __str__(self):
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Rotate(quat={_format_vec(self.quaternion, precision=4)})\n{child_str}"

    def expand(self):
        child_boxes = self.child.expand()
        op_rotation = Rotation.from_quat(self.quaternion)
        for box in child_boxes:
            box_rotation = Rotation.from_quat(box["quaternion"])
            box["quaternion"] = (op_rotation * box_rotation).as_quat()
            box["center"] = op_rotation.apply(box["center"])
        return child_boxes

    def serialize(self):
        return (Rotate, (self.quaternion.tolist(), [self.child]))


class Translate:
    def __init__(self, child, center):
        self.child = child
        self.center = np.array(center)

    def __str__(self):
        child_str = textwrap.indent(str(self.child), "    ")
        return f"Translate(center={_format_vec(self.center)})\n{child_str}"

    def expand(self):
        child_boxes = self.child.expand()
        for box in child_boxes:
            box["center"] += self.center
        return child_boxes

    def serialize(self):
        return (Translate, (self.center.tolist(), [self.child]))


# --- INTERNAL NODES ---
class Union:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        left_str = textwrap.indent(str(self.left), "    ")
        right_str = textwrap.indent(str(self.right), "    ")
        return f"Union(\n{left_str},\n{right_str}\n)"

    def expand(self):
        return self.left.expand() + self.right.expand()

    def serialize(self):
        return (Union, ([], [self.left, self.right]))


class Symmetry:
    def __init__(self, child):
        self.child = child


class SymRef(Symmetry):
    def __init__(self, child, plane_normal, point_on_plane):
        super().__init__(child)
        self.plane = np.array(plane_normal)
        self.point_on_plane = np.array(point_on_plane)

    def __str__(self):
        info = (
            f"SymRef(\n"
            f"    plane={_format_vec(self.plane)},\n"
            f"    point_on_plane={_format_vec(self.point_on_plane)}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"
        
    def expand(self):
        child_boxes = self.child.expand()
        generated_boxes = []
        plane_normal = self.plane / (np.linalg.norm(self.plane) + 1e-8)

        for box in child_boxes:
            # *** FIX: Use copy.deepcopy for a robust copy ***
            reflected_box = copy.deepcopy(box)
            
            vec_to_plane = box["center"] - self.point_on_plane
            dist = np.dot(vec_to_plane, plane_normal)
            reflected_box["center"] = box["center"] - 2 * dist * plane_normal

            R_orig = Rotation.from_quat(box["quaternion"]).as_matrix()
            M_reflect = np.identity(3) - 2 * np.outer(plane_normal, plane_normal)
            R_new = M_reflect @ R_orig
            if np.linalg.det(R_new) < 0:
                R_new[:, 0] *= -1 # Correct for improper rotation
            
            reflected_box["quaternion"] = Rotation.from_matrix(R_new).as_quat()
            generated_boxes.append(reflected_box)
            
        return child_boxes + generated_boxes

    def serialize(self):
        params = self.plane.tolist() + self.point_on_plane.tolist()
        return (SymRef, (params, [self.child]))


class SymRot(Symmetry):
    def __init__(self, child, axis, center, n_fold: int):
        super().__init__(child)
        self.axis = np.array(axis)
        self.center = np.array(center)
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
        
    def expand(self):
        child_boxes = self.child.expand()
        generated_boxes = []
        axis = self.axis / (np.linalg.norm(self.axis) + 1e-8)

        for i in range(1, self.n):
            angle = 2 * np.pi * i / self.n
            symmetry_rot = Rotation.from_rotvec(angle * axis)
            for box in child_boxes:
                # *** FIX: Use copy.deepcopy for a robust copy ***
                rotated_box = copy.deepcopy(box)

                vec_from_center = box["center"] - self.center
                rotated_box["center"] = self.center + symmetry_rot.apply(vec_from_center)
                original_rot = Rotation.from_quat(box["quaternion"])
                rotated_box["quaternion"] = (symmetry_rot * original_rot).as_quat()
                generated_boxes.append(rotated_box)

        return child_boxes + generated_boxes

    def serialize(self):
        params = self.axis.tolist() + self.center.tolist()
        return (SymRot, (params, [self.child, self.n]))


class SymTrans(Symmetry):
    def __init__(self, child, end_point, n_fold: int):
        super().__init__(child)
        self.end_point = np.array(end_point)
        self.n = n_fold

    def __str__(self):
        info = (
            f"SymTrans(\n"
            f"    end_point={_format_vec(self.end_point)},\n"
            f"    n={self.n}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"

    def expand(self):
        child_boxes = self.child.expand()
        if not child_boxes:
            return []
            
        generated_boxes = []
        for i in range(1, self.n):
            total_translation = i * self.end_point
            for box in child_boxes:
                # *** FIX: Use copy.deepcopy for a robust copy ***
                translated_box = copy.deepcopy(box)
                
                translated_box["center"] = box["center"] + total_translation
                generated_boxes.append(translated_box)
        
        return child_boxes + generated_boxes

    def serialize(self):
        params = self.end_point.tolist()
        return (SymTrans, (params, [self.child, self.n]))