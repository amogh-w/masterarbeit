"""
plot_utils.py

Utilities for expanding DSL trees and visualizing them as 3D meshes using k3d,
with robust debug logging.
"""

import k3d
import numpy as np
from scipy.spatial.transform import Rotation
from abstractionssymh.debug_utils import debug_info, debug_error, debug_success

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


def plot_dsl_with_k3d(dsl_root_node):
    """Expands and visualizes a DSL tree as a 3D mesh using k3d.

    Args:
        dsl_root_node: The root node of the DSL tree to be expanded and plotted.
            The expansion must yield a list of boxes where each box is a dict
            with the following keys:
                - ``center`` (np.ndarray): The (x, y, z) coordinates of the box center.
                - ``lengths`` (list or np.ndarray): The lengths along each axis.
                - ``quaternion`` (list or np.ndarray): Quaternion representing orientation.
                - ``label_id`` (int, optional): Part label ID, defaults to -1.

    Side Effects:
        Displays a 3D interactive mesh visualization in the notebook or viewer.

    Raises:
        Exception: Any errors during plotting are caught and logged using
        ``debug_error``.
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
        debug_success("3D plot displayed successfully.")
    except Exception as e:
        debug_error("Failed to generate or display 3D plot:", e)
