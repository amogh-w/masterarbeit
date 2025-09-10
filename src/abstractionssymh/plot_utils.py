import k3d
import numpy as np
from scipy.spatial.transform import Rotation
from abstractionssymh.dsl_nodes import Box, Union, SymRef, SymRot, SymTrans, Scale, Rotate, Translate

# Color map for part labels
LABEL_COLORS = {
    0: 0xFF0000,  # red    (Backrest)
    1: 0x00FF00,  # green (Seat)
    2: 0x0000FF,  # blue  (Leg)
    3: 0xFFFF00,  # yellow (Armrest)
    -1: 0x808080,  # gray (Unknown)
}


def expand_dsl_tree(node):
    """
    Recursively traverses the new DSL object tree and expands all symmetries
    to generate a final, flat list of box dictionaries.
    """
    # Base Case: New simple Box node
    if isinstance(node, Box):
        return [
            {
                "center": np.array([0.0, 0.0, 0.0]),
                "lengths": np.array([1.0, 1.0, 1.0]),
                "quaternion": np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
                "label_id": node.label,
            }
        ]

    # Handle transformation nodes
    elif isinstance(node, Scale):
        child_boxes = expand_dsl_tree(node.child)
        for box in child_boxes:
            scale_vector = np.array(node.lengths)
            box["lengths"] *= scale_vector
            box["center"] *= scale_vector
        return child_boxes

    elif isinstance(node, Rotate):
        child_boxes = expand_dsl_tree(node.child)
        op_rotation = Rotation.from_quat(node.quaternion)
        for box in child_boxes:
            box_rotation = Rotation.from_quat(box["quaternion"])
            box["quaternion"] = (op_rotation * box_rotation).as_quat()
            box["center"] = op_rotation.apply(box["center"])
        return child_boxes

    elif isinstance(node, Translate):
        child_boxes = expand_dsl_tree(node.child)
        for box in child_boxes:
            box["center"] += np.array(node.center)
        return child_boxes

    # Union and Symmetry logic
    elif isinstance(node, Union):
        left_boxes = expand_dsl_tree(node.left)
        right_boxes = expand_dsl_tree(node.right)
        return left_boxes + right_boxes

    elif isinstance(node, (SymRef, SymRot, SymTrans)):
        child_boxes = expand_dsl_tree(node.child)
        generated_boxes = []

        if isinstance(node, SymRef):
            plane_normal = np.array(node.plane) / (np.linalg.norm(node.plane) + 1e-8)
            point_on_plane = np.array(node.point_on_plane)
            for box in child_boxes:
                reflected_box = box.copy()
                vec_to_plane = box["center"] - point_on_plane
                dist = np.dot(vec_to_plane, plane_normal)
                reflected_box["center"] = box["center"] - 2 * dist * plane_normal
                R_orig = Rotation.from_quat(box["quaternion"]).as_matrix()
                M_reflect = np.identity(3) - 2 * np.outer(plane_normal, plane_normal)
                R_new = M_reflect @ R_orig
                if np.linalg.det(R_new) < 0:
                    R_new[:, 0] *= -1
                reflected_box["quaternion"] = Rotation.from_matrix(R_new).as_quat()
                generated_boxes.append(reflected_box)

        elif isinstance(node, SymRot):
            axis = np.array(node.axis) / (np.linalg.norm(node.axis) + 1e-8)
            center = np.array(node.center)
            for i in range(1, node.n):
                angle = 2 * np.pi * i / node.n
                symmetry_rot = Rotation.from_rotvec(angle * axis)
                for box in child_boxes:
                    rotated_box = box.copy()
                    vec_from_center = box["center"] - center
                    rotated_box["center"] = center + symmetry_rot.apply(vec_from_center)
                    original_rot = Rotation.from_quat(box["quaternion"])
                    rotated_box["quaternion"] = (symmetry_rot * original_rot).as_quat()
                    generated_boxes.append(rotated_box)

        elif isinstance(node, SymTrans):
            if child_boxes:
                translation_vector = np.array(node.end_point)
                for i in range(1, node.n):
                    total_translation = i * translation_vector
                    for box in child_boxes:
                        translated_box = box.copy()
                        translated_box["center"] = box["center"] + total_translation
                        generated_boxes.append(translated_box)

        return child_boxes + generated_boxes

    return []


def plot_dsl_with_k3d(dsl_root_node):
    """
    Top-level function to expand a DSL tree and plot it using k3d.
    """
    print("Expanding DSL tree for visualization...")
    final_boxes = expand_dsl_tree(dsl_root_node)
    print(f"Found {len(final_boxes)} total boxes after expansion.")

    plot = k3d.plot(name="Reconstructed DSL Shape")
    for box in final_boxes:
        center, lengths, quaternion, label_id = (
            box["center"],
            box["lengths"],
            box["quaternion"],
            box.get("label_id", -1),
        )
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        d1, d2, d3 = [
            col * length / 2 for col, length in zip(rotation_matrix.T, lengths)
        ]
        corners = np.array(
            [
                center - d1 - d2 - d3,
                center - d1 + d2 - d3,
                center + d1 - d2 - d3,
                center + d1 + d2 - d3,
                center - d1 - d2 + d3,
                center - d1 + d2 + d3,
                center + d1 - d2 + d3,
                center + d1 + d2 + d3,
            ],
            dtype=np.float32,
        )

        faces = np.array(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
            ],
            dtype=np.uint32,
        )

        color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
        plot += k3d.mesh(corners, faces, color=color)

    plot.display()
