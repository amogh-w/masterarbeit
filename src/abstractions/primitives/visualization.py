"""
visualization.py

Provides functions to visualize 2D geometric shapes (Box and Circle) using
multiple backends including matplotlib, k3d, and plotly.

Supports both static and interactive visualizations with options for
2D locked views in 3D space (k3d).
"""

import matplotlib.pyplot as plt
import k3d
import plotly.graph_objects as go
import numpy as np
from abstractions.dsl.core import Shape
from abstractions.primitives.shapes import Box, Circle


def show_boxes(
    boxes: list[Box], limits=(-1, 1), backend="matplotlib", lock_2d_view=True
):
    """
    Visualizes a list of Box objects using the specified backend.

    Parameters
    ----------
    boxes : list[Box]
        A list of Box instances to display.
    limits : tuple, optional
        The (min, max) bounds for both x and y axes. Default is (-1, 1).
    backend : str, optional
        One of 'matplotlib', 'k3d', or 'plotly'. Default is 'matplotlib'.
    lock_2d_view : bool, optional
        Only used for k3d. Locks camera to 2D if True. Default is True.

    Returns
    -------
    None
    """
    if backend == "matplotlib":
        show_boxes_matplotlib(boxes, limits)
    elif backend == "k3d":
        show_boxes_k3d(boxes, limits, lock_2d_view)
    elif backend == "plotly":
        show_boxes_plotly(boxes, limits)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def show_circles(
    circles: list[Circle], limits=(-1, 1), backend="matplotlib", lock_2d_view=True
):
    """
    Visualizes a list of Circle objects using the specified backend.

    Parameters
    ----------
    circles : list[Circle]
        A list of Circle instances to display.
    limits : tuple, optional
        The (min, max) bounds for both x and y axes. Default is (-1, 1).
    backend : str, optional
        One of 'matplotlib', 'k3d', or 'plotly'. Default is 'matplotlib'.
    lock_2d_view : bool, optional
        Only used for k3d. Locks camera to 2D if True. Default is True.

    Returns
    -------
    None
    """
    if backend == "matplotlib":
        show_circles_matplotlib(circles, limits)
    elif backend == "k3d":
        show_circles_k3d(circles, limits, lock_2d_view)
    elif backend == "plotly":
        show_circles_plotly(circles, limits)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def show_boxes_matplotlib(boxes: list[Box], limits=(-1, 1)):
    """
    Visualizes a list of Box objects using matplotlib in 2D.

    Parameters
    ----------
    boxes : list[Box]
        A list of Box instances to display.
    limits : tuple, optional
        The (min, max) bounds for both x and y axes. Default is (-1, 1).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)

    for i, box in enumerate(boxes):
        rect = plt.Rectangle(
            (
                box.center[0].item() - box.scale[0].item() / 2,
                box.center[1].item() - box.scale[1].item() / 2,
            ),
            box.scale[0].item(),
            box.scale[1].item(),
            rotation_point="center",
            color=f"C{i}",
            fill=False,
        )
        ax.add_patch(rect)

    plt.grid()
    plt.show()


def show_circles_matplotlib(circles: list[Circle], limits=(-1, 1)):
    """
    Visualizes a list of Circle objects using matplotlib in 2D.

    Parameters
    ----------
    circles : list[Circle]
        A list of Circle instances to display.
    limits : tuple, optional
        The (min, max) bounds for both x and y axes. Default is (-1, 1).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)

    for i, circle in enumerate(circles):
        circ = plt.Circle(
            (circle.center[0].item(), circle.center[1].item()),
            circle.radius.item(),
            color=f"C{i}",
            fill=False,
        )
        ax.add_patch(circ)

    plt.grid()
    plt.show()


def show_boxes_k3d(boxes: list[Box], limits=(-1, 1), lock_2d_view=True):
    """
    Visualizes a list of Box objects using an interactive k3d plot in 3D (2D locked view).

    Parameters
    ----------
    boxes : list[Box]
        A list of Box instances to display.
    limits : tuple, optional
        The (min, max) bounds for both x and y axes. Default is (-1, 1).
    lock_2d_view : bool, optional
        Whether to lock the view to 2D (top-down). Default is True.

    Returns
    -------
    None
    """
    plot = k3d.plot()
    for i, box in enumerate(boxes):
        cx, cy = box.center.tolist()
        sx, sy = box.scale.tolist()
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z = 0.0
        lines = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]]
        plot += k3d.line(np.array(lines, dtype=np.float32), color=0x0055FF, width=0.01)
    if lock_2d_view:
        plot.camera_auto_fit = False
        plot.camera_no_rotate = True
        plot.camera = [0, 0, 3, 0, 0, 0, 0, 1, 0]
    plot.display()


def show_circles_k3d(circles: list[Circle], limits=(-1, 1), lock_2d_view=True):
    """
    Visualizes a list of Circle objects using an interactive k3d plot in 3D (2D locked view).

    Parameters
    ----------
    circles : list[Circle]
        A list of Circle instances to display.
    limits : tuple, optional
        The (min, max) bounds for both x and y axes. Default is (-1, 1).
    lock_2d_view : bool, optional
        Whether to lock the view to 2D (top-down). Default is True.

    Returns
    -------
    None
    """
    plot = k3d.plot()
    theta = np.linspace(0, 2 * np.pi, 100)
    for i, circle in enumerate(circles):
        cx, cy = circle.center.tolist()
        r = circle.radius.item()
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        z = np.zeros_like(x)
        coords = np.vstack([x, y, z]).T.astype(np.float32)
        plot += k3d.line(coords, color=0xFF0000, width=0.01)
    if lock_2d_view:
        plot.camera_auto_fit = False
        plot.camera_no_rotate = True
        plot.camera = [0, 0, 3, 0, 0, 0, 0, 1, 0]
    plot.display()


def show_boxes_plotly(boxes: list[Box], limits=(-1, 1)):
    """
    Visualizes a list of Box objects using Plotly in 2D.

    Parameters
    ----------
    boxes : list[Box]
        A list of Box instances to display.
    limits : tuple, optional
        The (min, max) bounds for both x and y axes. Default is (-1, 1).

    Returns
    -------
    None
    """
    fig = go.Figure()
    for i, box in enumerate(boxes):
        cx, cy = box.center.tolist()
        sx, sy = box.scale.tolist()
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                name=f"Box {i}",
            )
        )
    fig.update_layout(
        xaxis=dict(range=limits),
        yaxis=dict(range=limits, scaleanchor="x", scaleratio=1),
        title="Boxes (Plotly)",
        showlegend=False,
    )
    fig.show()


def show_circles_plotly(circles: list[Circle], limits=(-1, 1)):
    """
    Visualizes a list of Circle objects using Plotly in 2D.

    Parameters
    ----------
    circles : list[Circle]
        A list of Circle instances to display.
    limits : tuple, optional
        The (min, max) bounds for both x and y axes. Default is (-1, 1).

    Returns
    -------
    None
    """
    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 100)
    for i, circle in enumerate(circles):
        cx, cy = circle.center.tolist()
        r = circle.radius.item()
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Circle {i}"))
    fig.update_layout(
        xaxis=dict(range=limits),
        yaxis=dict(range=limits, scaleanchor="x", scaleratio=1),
        title="Circles (Plotly)",
        showlegend=False,
    )
    fig.show()


import ipywidgets as widgets
from IPython.display import display, clear_output
from graphviz import Digraph


def _add_graphviz_nodes(dot, shape, parent_id=None, node_id=0):
    if isinstance(shape, Shape):
        label = shape.__class__.__name__
        dot.node(
            str(node_id),
            label,
            shape="box",
            style="filled",
            fillcolor="#e0f7fa",
            fontname="Helvetica",
        )
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id))
        _, args = shape.param_tuple()
        next_id = node_id + 1
        for arg in args:
            next_id = _add_graphviz_nodes(dot, arg, node_id, next_id)
        return next_id
    else:
        dot.node(
            str(node_id),
            repr(shape),
            shape="ellipse",
            style="filled",
            fillcolor="#fff9c4",
            fontname="Helvetica",
        )
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id))
        return node_id + 1


def print_tree(shape):
    """
    Renders a shape DSL tree as a styled Graphviz Digraph for display in Jupyter.

    Args:
        shape (Shape): Root of the DSL tree.

    Returns:
        graphviz.Digraph: A styled graph object that can be rendered inline.
    """
    dot = Digraph(format="svg")
    # dot.attr(
    #     rankdir="TB",
    #     fontname="Helvetica",
    #     fontsize="10",
    #     dpi="150",
    #     nodesep="0.4",
    #     ranksep="0.6",
    #     concentrate="true",
    # )
    # dot.attr(size="")  # allow Graphviz to autosize
    _add_graphviz_nodes(dot, shape)
    return dot


def visualize_combined_dataset(dataset, categories):
    """
    Creates an interactive widget to browse and visualize a combined dataset.

    Parameters
    ----------
    dataset : list
        Combined list of shape objects.
    categories : list of tuples
        List of (category_name, start_idx, end_idx) defining dataset segments.

    Returns
    -------
    None
        Displays interactive widgets in the notebook.
    """
    category_dropdown = widgets.Dropdown(
        options=[c[0] for c in categories], description="Category:"
    )

    index_slider = widgets.IntSlider(
        min=0, max=categories[0][2] - categories[0][1], description="Index:"
    )

    output = widgets.Output()

    def global_index(category_name, relative_idx):
        for name, start, end in categories:
            if name == category_name:
                return start + relative_idx
        return None

    def update_slider_range(change):
        selected = change["new"]
        for name, start, end in categories:
            if name == selected:
                index_slider.min = 0
                index_slider.max = end - start
                index_slider.value = 0
                break

    def update_visualization(change=None):
        output.clear_output()
        with output:
            cat = category_dropdown.value
            idx = index_slider.value
            g_idx = global_index(cat, idx)
            if g_idx is None or g_idx >= len(dataset):
                print("Invalid index!")
                return
            shape = dataset[g_idx]
            display(print_tree(shape))
            show_boxes(shape.get_box_list(), backend="plotly")

    category_dropdown.observe(update_slider_range, names="value")
    category_dropdown.observe(update_visualization, names="value")
    index_slider.observe(update_visualization, names="value")

    display(widgets.VBox([category_dropdown, index_slider]), output)

    # Initialize display
    update_visualization()
