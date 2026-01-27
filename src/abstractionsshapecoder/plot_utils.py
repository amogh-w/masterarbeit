"""plot_utils.py

Utilities for expanding DSL trees and visualizing them as 3D meshes using k3d
and Matplotlib, with robust debug logging.

This module provides functions to:
- Expand Domain-Specific Language (DSL) tree structures into their
  constituent geometric primitives (boxes).
- Visualize the resulting geometries as 3D meshes using `k3d` for
  interactive plots.
- Visualize the resulting geometries using `matplotlib` for static images.
- Generate grid plots and base64-encoded images for Dash applications.
"""

import base64
import io
import math
from pathlib import Path
import k3d
import numpy as np
from scipy.spatial.transform import Rotation
from debug_utils import debug_info, debug_error, debug_success
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Module-level Constants ---

LABEL_COLORS = {
    0: 0xFF0000,  # red    (Backrest)
    1: 0x00FF00,  # green  (Seat)
    2: 0x0000FF,  # blue   (Leg)
    3: 0xFFFF00,  # yellow (Armrest)
    -1: 0x808080,  # gray   (Unknown)
}
"""dict: Color map for part labels, mapping integer IDs to hex color codes."""


# --- Functions ---

def expand_dsl_tree(node):
    """Expands a DSL tree node into its final geometry.

    Parameters
    ----------
    node : object
        A DSL tree node object that must implement an ``expand()`` method.

    Returns
    -------
    list[dict]
        A list of box dictionaries, each containing geometry (center,
        lengths, quaternion) and label information. Returns an empty
        list if expansion fails.
    """
    try:
        return node.expand()
    except Exception as e:
        debug_error("Failed to expand DSL tree:", e)
        return []


def plot_dsl_with_k3d(dsl_root_node, save_path=None):
    """Visualize a DSL tree as a 3D mesh using k3d.

    Expands the tree and renders it as an interactive 3D plot in
    a Jupyter environment or saves it as a standalone HTML file.

    Parameters
    ----------
    dsl_root_node : object
        The root DSL node of the tree to visualize.
    save_path : str or pathlib.Path, optional
        If provided, saves the k3d scene as an interactive HTML file
        at this location. If None (default), displays the plot inline.
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
    """Convert a hex integer to a normalized (0.0-1.0) RGB tuple.

    Parameters
    ----------
    hex_color : int
        The color as a 24-bit integer (e.g., 0xFF0000).

    Returns
    -------
    tuple[float, float, float]
        A tuple (r, g, b) with values ranging from 0.0 to 1.0.
    """
    r = ((hex_color >> 16) & 0xFF) / 255.0
    g = ((hex_color >> 8) & 0xFF) / 255.0
    b = (hex_color & 0xFF) / 255.0
    return (r, g, b)


def plot_dsl_with_matplotlib(
    dsl_root_node, save_path=None, figsize=(8, 8), axis_limits=(-1, 1)
):
    """Visualize a DSL tree as a 3D mesh using Matplotlib.

    Expands the tree and renders it as a static 3D plot.
    Optionally saves the plot as a PNG file.

    Parameters
    ----------
    dsl_root_node : object
        The root DSL node of the tree to visualize.
    save_path : str or pathlib.Path, optional
        If provided, saves the plot as a PNG file at this location.
        If None (default), displays the plot using ``plt.show()``.
    figsize : tuple[int, int], optional
        The figure size (width, height) in inches. Default is (8, 8).
    axis_limits : tuple[float, float], optional
        The (min, max) limits for the X, Y, and Z axes.
        Default is (-1, 1).
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
    """Generate a Matplotlib 3D plot as a base64 string for Dash.

    Renders a DSL object to a Matplotlib figure, saves it to an
    in-memory buffer, and returns it as a base64-encoded PNG string
    suitable for embedding in a Dash ``html.Img`` component.

    Parameters
    ----------
    dsl_root_node : object
        The root DSL node of the tree to visualize.
    axis_limits : tuple[float, float], optional
        The (min, max) limits for the X, Y, and Z axes.
        Default is (-1, 1).

    Returns
    -------
    str or None
        A base64-encoded PNG data URI (e.g., "data:image/png;base64,...").
        Returns None if the DSL tree expansion fails.
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
        d1, d2, d3 = [
            col * length / 2 for col, length in zip(rotation_matrix.T, lengths)
        ]

        corners = np.array(
            [
                center - d1 - d2 - d3,
                center + d1 - d2 - d3,
                center + d1 + d2 - d3,
                center - d1 + d2 - d3,
                center - d1 - d2 + d3,
                center + d1 - d2 + d3,
                center + d1 + d2 + d3,
                center - d1 + d2 + d3,
            ]
        )

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
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def plot_dsl_grid(
    dsl_objects,
    names,
    save_path=None,
    grid_cols=3,
    figsize_per_plot=(6, 6),
    axis_limits=(-0.8, 0.8),
    grid_title="",
):
    """Render a grid of DSL objects using Matplotlib.

    Generates a single image containing a grid of 3D plots, one for
    each provided DSL object. The grid layout is determined
    automatically based on the number of objects and `grid_cols`.

    Parameters
    ----------
    dsl_objects : list[object]
        A list of the DSL root nodes to plot.
    names : list[str]
        A list of strings, providing a title for each subplot.
        Must be the same length as `dsl_objects`.
    save_path : str or pathlib.Path, optional
        Path to save the entire grid as a single PNG. If None
        (default), displays the plot using ``plt.show()``.
    grid_cols : int, optional
        The number of columns to use for the grid layout. Default is 3.
    figsize_per_plot : tuple[int, int], optional
        The (width, height) in inches for each individual subplot.
        Default is (6, 6).
    axis_limits : tuple[float, float], optional
        The (min, max) limits for all axes in all subplots.
        Default is (-0.8, 0.8).
    grid_title : str, optional
        An overall title for the entire grid image. Default is "".
    """
    if len(dsl_objects) != len(names):
        debug_error("Error: The number of DSL objects must match the number of names.")
        return

    num_plots = len(dsl_objects)
    if num_plots == 0:
        debug_info("No DSL objects provided to plot.")
        return

    num_rows = math.ceil(num_plots / grid_cols)
    total_figsize = (grid_cols * figsize_per_plot[0], num_rows * figsize_per_plot[1])

    fig, axes = plt.subplots(
        num_rows,
        grid_cols,
        figsize=total_figsize,
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for i, dsl_root_node in enumerate(dsl_objects):
        ax = axes_flat[i]
        title = names[i]

        final_boxes = expand_dsl_tree(dsl_root_node)
        if not final_boxes:
            ax.set_title(f"{title}\n(No boxes found)", fontsize=10)
            continue

        for box in final_boxes:
            center = np.array(box.get("center", [0, 0, 0]), dtype=float)
            lengths = np.asarray(box.get("lengths", [1, 1, 1]), dtype=float).ravel()
            quaternion = box.get("quaternion", [0, 0, 0, 1])
            label_id = box.get("label_id", -1)

            rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
            d1, d2, d3 = [
                col * length / 2 for col, length in zip(rotation_matrix.T, lengths)
            ]

            corners = np.array(
                [
                    center - d1 - d2 - d3,
                    center + d1 - d2 - d3,
                    center + d1 + d2 - d3,
                    center - d1 + d2 - d3,
                    center - d1 - d2 + d3,
                    center + d1 - d2 + d3,
                    center + d1 + d2 + d3,
                    center - d1 + d2 + d3,
                ]
            )

            faces_indices = [
                [corners[0], corners[1], corners[2], corners[3]],
                [corners[4], corners[5], corners[6], corners[7]],
                [corners[0], corners[1], corners[5], corners[4]],
                [corners[2], corners[3], corners[7], corners[6]],
                [corners[0], corners[3], corners[7], corners[4]],
                [corners[1], corners[2], corners[6], corners[5]],
            ]

            hex_color = LABEL_COLORS.get(label_id, LABEL_COLORS.get(-1))
            color = hex_to_rgb_normalized(hex_color)

            collection = Poly3DCollection(
                faces_indices,
                facecolors=color,
                edgecolors="k",
                linewidths=0.2,
                alpha=0.85,
            )
            ax.add_collection3d(collection)

        ax.set_title(title, fontsize=10)
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.set_zlim(axis_limits)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)

    for i in range(num_plots, len(axes_flat)):
        axes_flat[i].set_axis_off()

    # --- NEW: Add a suptitle for the entire grid ---
    if grid_title:
        fig.suptitle(grid_title, fontsize=16, y=0.98)  # y adjusts position

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)  # Adjust layout

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        debug_success(f"Grid plot saved as PNG: {save_path}")
    else:
        plt.show()
        debug_success("Grid plot displayed successfully.")

    plt.close(fig)