"""
plot_utils.py

Utilities for expanding DSL trees and visualizing them as 3D meshes using k3d,
with robust debug logging.
"""

import base64
import io
from pathlib import Path
import k3d
import numpy as np
from scipy.spatial.transform import Rotation
from abstractionssymh.debug_utils import debug_info, debug_error, debug_success
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Color map for part labels
LABEL_COLORS = {
    0: 0xFF0000,  # red    (Backrest)
    1: 0x00FF00,  # green  (Seat)
    2: 0x0000FF,  # blue   (Leg)
    3: 0xFFFF00,  # yellow (Armrest)
    -1: 0x808080,  # gray   (Unknown)
}


def expand_dsl_tree(node):
    """Expands a DSL tree node into its final geometry.

    Args:
        node: A DSL tree node object that must implement an ``expand()`` method.

    Returns:
        list: A list of box dictionaries containing geometry and label
        information. Returns an empty list if expansion fails.
    """
    try:
        return node.expand()
    except Exception as e:
        debug_error("Failed to expand DSL tree:", e)
        return []


def plot_dsl_with_k3d(dsl_root_node, save_path=None):
    """
    Expands and visualizes a DSL tree as a 3D mesh using k3d.
    Optionally saves the interactive scene as an HTML file.

    Args:
        dsl_root_node: The root DSL node of the tree.
        save_path (str or Path, optional): If provided, saves the k3d scene as an HTML file.
    """
    debug_info("Expanding DSL tree for visualization...")
    final_boxes = expand_dsl_tree(dsl_root_node)

    if not final_boxes:
        debug_error("No boxes found after DSL tree expansion. Aborting plot.")
        return

    debug_info(f"Found {len(final_boxes)} total boxes after expansion.")

    try:
        plot = k3d.plot(name="Reconstructed DSL Shape")

        for box in final_boxes:
            center, lengths, quaternion, label_id = (
                np.array(box["center"], dtype=float),
                np.asarray(box["lengths"], dtype=float).ravel(),
                box["quaternion"],
                box.get("label_id", -1),
            )

            if lengths.size == 1:
                lengths = np.repeat(lengths, 3)

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

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            html_content = plot.get_snapshot()
            with open(save_path, "w") as f:
                f.write(html_content)
            debug_success(f"Interactive snapshot saved as HTML: {save_path}")
        else:
            plot.display()
            debug_success("3D plot displayed successfully.")

    except Exception as e:
        debug_error("Failed to generate or display 3D plot:", e)


def hex_to_rgb_normalized(hex_color):
    """Converts a hex integer to a normalized RGB tuple."""
    r = ((hex_color >> 16) & 0xFF) / 255.0
    g = ((hex_color >> 8) & 0xFF) / 255.0
    b = (hex_color & 0xFF) / 255.0
    return (r, g, b)


def plot_dsl_with_matplotlib(
    dsl_root_node, save_path=None, figsize=(8, 8), axis_limits=(-1, 1)
):
    """
    Expands and visualizes a DSL tree as a 3D mesh using matplotlib.
    Optionally saves the plot as a PNG file.
    """
    debug_info("Expanding DSL tree for visualization...")
    final_boxes = expand_dsl_tree(dsl_root_node)

    if not final_boxes:
        debug_error("No boxes found after DSL tree expansion. Aborting plot.")
        return

    debug_info(f"Found {len(final_boxes)} total boxes after expansion.")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for box in final_boxes:
        center = np.array(box["center"], dtype=float)
        lengths = np.asarray(box["lengths"], dtype=float).ravel()
        quaternion = box["quaternion"]
        label_id = box.get("label_id", -1)

        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        d1, d2, d3 = [
            col * length / 2 for col, length in zip(rotation_matrix.T, lengths)
        ]

        # Define the 8 vertices of the box
        corners = np.array(
            [
                center - d1 - d2 - d3,  # 0
                center + d1 - d2 - d3,  # 1
                center + d1 + d2 - d3,  # 2
                center - d1 + d2 - d3,  # 3
                center - d1 - d2 + d3,  # 4
                center + d1 - d2 + d3,  # 5
                center + d1 + d2 + d3,  # 6
                center - d1 + d2 + d3,  # 7
            ]
        )

        # Faces logic: Define the 6 faces using correct vertex indices
        faces_indices = [
            [corners[0], corners[1], corners[2], corners[3]],  # Bottom
            [corners[4], corners[5], corners[6], corners[7]],  # Top
            [corners[0], corners[1], corners[5], corners[4]],  # Front
            [corners[2], corners[3], corners[7], corners[6]],  # Back
            [corners[0], corners[3], corners[7], corners[4]],  # Left
            [corners[1], corners[2], corners[6], corners[5]],  # Right
        ]

        hex_color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
        color = hex_to_rgb_normalized(hex_color)

        collection = Poly3DCollection(
            faces_indices, facecolors=color, edgecolors="k", linewidths=0.3, alpha=0.8
        )
        ax.add_collection3d(collection)

    # Set plot properties
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_zlim(axis_limits)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # plt.title(save_path)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        debug_success(f"3D plot saved as PNG: {save_path}")
    else:
        plt.show()
        debug_success("3D plot displayed successfully.")
    
    plt.close(fig)

def plot_dsl_with_matplotlib_dash(dsl_root_node, axis_limits=(-1, 1)):
    """
    Generates a Matplotlib 3D plot from a DSL object and returns a base64 PNG
    string ready for Dash display in html.Img.
    """

    # Expand DSL tree to get final boxes
    final_boxes = expand_dsl_tree(dsl_root_node)
    if not final_boxes:
        return None

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for box in final_boxes:
        center = np.array(box["center"], dtype=float)
        lengths = np.asarray(box["lengths"], dtype=float).ravel()
        quaternion = box["quaternion"]
        label_id = box.get("label_id", -1)

        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        d1, d2, d3 = [col * length / 2 for col, length in zip(rotation_matrix.T, lengths)]

        corners = np.array([
            center - d1 - d2 - d3,
            center + d1 - d2 - d3,
            center + d1 + d2 - d3,
            center - d1 + d2 - d3,
            center - d1 - d2 + d3,
            center + d1 - d2 + d3,
            center + d1 + d2 + d3,
            center - d1 + d2 + d3,
        ])

        faces_indices = [
            [corners[0], corners[1], corners[2], corners[3]],  # Bottom
            [corners[4], corners[5], corners[6], corners[7]],  # Top
            [corners[0], corners[1], corners[5], corners[4]],  # Front
            [corners[2], corners[3], corners[7], corners[6]],  # Back
            [corners[0], corners[3], corners[7], corners[4]],  # Left
            [corners[1], corners[2], corners[6], corners[5]],  # Right
        ]

        hex_color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
        color = hex_to_rgb_normalized(hex_color)
        collection = Poly3DCollection(faces_indices, facecolors=color, edgecolors="k", linewidths=0.3, alpha=0.8)
        ax.add_collection3d(collection)

    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_zlim(axis_limits)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()

    # Save to in-memory buffer and encode as base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"