"""
metrics.py

Provides functions to quantitatively and qualitatively evaluate the
effectiveness of DSL tree abstraction.

This module calculates metrics in four main categories:
1.  **Tree Complexity:** Reductions in total node count and max tree depth.
2.  **Parameter Compression:** The ratio of parameter reduction when replacing
    full parameters with compressed latent vectors.
3.  **Abstraction Coverage:** The percentage of original operational nodes
    (e.g., `Scale`, `Rotate`) that were successfully replaced by
    `Abstraction` nodes.
4.  **Geometric Fidelity:** The Chamfer distance between the point clouds
    of the original shape and the reconstructed shape (after expansion
    from the abstracted tree).

It also includes utilities for point cloud generation and visualization.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import k3d

from abstractionssymh.dsl_nodes import (
    Box,
    Scale,
    Rotate,
    Translate,
    Union,
    SymRef,
    SymRot,
    SymTrans,
)

from abstractionssymh.abstraction_utils import Abstraction

# ==============================================================================
# --- HELPER FUNCTIONS (Corrected for Abstraction Node) ---
# ==============================================================================

def _count_nodes_recursive(node):
    """Recursively count all operational nodes in a DSL tree.

    Handles both standard DSL nodes (with `.serialize()`) and `Abstraction`
    nodes (with `.children`).

    Parameters
    ----------
    node : object
        The current DSL node (e.g., `Box`, `Union`, `Abstraction`).

    Returns
    -------
    int
        Total count of this node plus all descendant nodes.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return 0

    count = 1  # Count the current node
    if isinstance(node, Abstraction):
        children = node.children
    elif hasattr(node, "serialize"):
        children = node.serialize()[1][1]
    else:
        children = []

    for child in children:
        count += _count_nodes_recursive(child)
    return count


def _get_max_depth_recursive(node, current_depth):
    """Recursively find the maximum depth of a DSL tree.

    Handles both standard DSL nodes and `Abstraction` nodes.

    Parameters
    ----------
    node : object
        The current DSL node.
    current_depth : int
        The depth of the `node` in the recursion.

    Returns
    -------
    int
        Maximum depth of the subtree rooted at `node`.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return current_depth

    if isinstance(node, Abstraction):
        children = node.children
    elif hasattr(node, "serialize"):
        children = node.serialize()[1][1]
    else:
        children = []

    # Filter for valid child nodes
    valid_children = [
        c for c in children
        if hasattr(c, "serialize") or isinstance(c, Abstraction)
    ]

    if not valid_children:
        return current_depth + 1

    return max(
        _get_max_depth_recursive(c, current_depth + 1) for c in valid_children
    )


def _count_parameters_recursive(node, is_abstracted_tree=False):
    """Recursively count parameters in a DSL tree.

    This function is used to count parameters for both original and
    abstracted trees.
    
    - When `is_abstracted_tree=False` (for an original tree), it
      counts the full parameters from `.serialize()`.
    - When `is_abstracted_tree=True` (for an abstracted tree), it
      counts the `compressed_params` for `Abstraction` nodes and
      the full parameters for any remaining concrete nodes.

    Parameters
    ----------
    node : object
        The current DSL node.
    is_abstracted_tree : bool, optional
        If True, counts compressed parameters for `Abstraction` nodes.
        If False (default), counts standard parameters for
        concrete nodes. An original tree should always use False.

    Returns
    -------
    int
        Total number of parameters in the subtree.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return 0

    if isinstance(node, Abstraction):
        # If it's an abstracted tree, count the compressed params.
        # If it's an original tree, this branch is never hit.
        params_count = len(node.compressed_params) if is_abstracted_tree else 0
        children = node.children
    else:
        # Standard node
        params, children = node.serialize()[1]
        params_count = len(params)

    for child in children:
        params_count += _count_parameters_recursive(child, is_abstracted_tree)
    return params_count


def _count_replaced_nodes_recursive(node):
    """Count all original nodes replaced by Abstraction nodes.

    Recursively traverses an *abstracted* tree. When it finds an
    `Abstraction` node, it counts 1 (for a singleton pattern like "Scale")
    or 2 (for a pair pattern like "Scale(Box)") and continues
    traversing its children.

    Parameters
    ----------
    node : object
        The current node of an *abstracted* DSL tree.

    Returns
    -------
    int
        Total number of original nodes represented by all
        `Abstraction` nodes in this subtree.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return 0

    total_replaced = 0
    if isinstance(node, Abstraction):
        children_to_scan = node.children
        # Check if it's a pair pattern (e.g., "Scale(Box)")
        if '(' in node.pattern_name and ')' in node.pattern_name:
            total_replaced += 2  # Pair node (e.g., Scale, Box)
        else:
            total_replaced += 1  # Singleton node (e.g., Scale)
    elif hasattr(node, "serialize"):
        children_to_scan = node.serialize()[1][1]
    else:
        children_to_scan = []

    for child in children_to_scan:
        total_replaced += _count_replaced_nodes_recursive(child)
    return total_replaced


def _traverse(node):
    """Generator to iterate through all nodes in a tree (pre-order).

    Handles both standard DSL nodes and `Abstraction` nodes.

    Parameters
    ----------
    node : object
        The root node to start traversal from.

    Yields
    ------
    object
        Each node in the tree.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return
    yield node
    
    if isinstance(node, Abstraction):
        children = node.children
    elif hasattr(node, "serialize"):
        children = node.serialize()[1][1]
    else:
        children = []

    for child in children:
        yield from _traverse(child)


# ==============================================================================
# --- METRIC 1: Tree Complexity Reduction ---
# ==============================================================================

def calculate_tree_complexity_reduction(original_chair, abstracted_chair):
    """Calculate the percentage reduction in node count and tree depth.

    Compares the total number of nodes and the maximum depth of the
    original tree versus the abstracted tree.

    Parameters
    ----------
    original_chair : object
        The root node of the original (non-abstracted) DSL tree.
    abstracted_chair : object
        The root node of the abstracted DSL tree.

    Returns
    -------
    dict
        A dictionary with "metrics" (percentage reductions) and
        "details" (raw counts).
    """
    original_nodes = _count_nodes_recursive(original_chair)
    abstracted_nodes = _count_nodes_recursive(abstracted_chair)
    original_depth = _get_max_depth_recursive(original_chair, 0)
    abstracted_depth = _get_max_depth_recursive(abstracted_chair, 0)

    node_reduction = (
        (original_nodes - abstracted_nodes) / original_nodes
        if original_nodes else 0
    )
    depth_reduction = (
        (original_depth - abstracted_depth) / original_depth
        if original_depth else 0
    )

    return {
        "metrics": {
            "node_count_reduction": f"{node_reduction:.2%}",
            "max_depth_reduction": f"{depth_reduction:.2%}"
        },
        "details": {
            "original_nodes": original_nodes,
            "abstracted_nodes": abstracted_nodes,
            "original_depth": original_depth,
            "abstracted_depth": abstracted_depth,
        }
    }


# ==============================================================================
# --- METRIC 2: Parameter Compression Ratio ---
# ==============================================================================

def calculate_parameter_compression(original_chair, abstracted_chair):
    """Calculate the reduction ratio of parameters after abstraction.

    Compares the total number of parameters in the original tree
    (e.g., `[sx, sy, sz]`, 3 params) against the total number in the
    abstracted tree (e.g., `[z1, z2]`, 2 params).

    Parameters
    ----------
    original_chair : object
        The root DSL node of the original shape.
    abstracted_chair : object
        The root DSL node of the abstracted shape.

    Returns
    -------
    dict
        A dictionary with "metrics" (compression ratio) and
        "details" (raw parameter counts).
    """
    original_params = _count_parameters_recursive(
        original_chair, is_abstracted_tree=False
    )
    abstracted_params = _count_parameters_recursive(
        abstracted_chair, is_abstracted_tree=True
    )
    ratio = (
        (original_params - abstracted_params) / original_params
        if original_params else 0
    )

    return {
        "metrics": {
            "parameter_compression_ratio": f"{ratio:.2%}"
        },
        "details": {
            "original_parameters": original_params,
            "abstracted_parameters": abstracted_params,
        }
    }


# ==============================================================================
# --- METRIC 3: Abstraction Coverage ---
# ==============================================================================

def calculate_abstraction_coverage(original_chair, abstracted_chair):
    """Calculate the percentage of original nodes replaced by abstractions.

    Measures how many of the original "operational" nodes (all nodes
    except `Box`) were successfully replaced by an `Abstraction` node.

    Parameters
    ----------
    original_chair : object
        The root DSL node of the original shape.
    abstracted_chair : object
        The root DSL node of the abstracted shape.

    Returns
    -------
    dict
        A dictionary with "metrics" (coverage percentage) and
        "details" (raw node counts).
    """
    original_op_nodes = sum(
        1 for node in _traverse(original_chair) if not isinstance(node, Box)
    )
    replaced_nodes = _count_replaced_nodes_recursive(abstracted_chair)
    coverage = replaced_nodes / original_op_nodes if original_op_nodes else 0

    return {
        "metrics": {
            "abstraction_coverage": f"{coverage:.2%}"
        },
        "details": {
            "original_operational_nodes": original_op_nodes,
            "nodes_replaced_by_abstractions": replaced_nodes
        }
    }


# ==============================================================================
# --- POINT CLOUD & CHAMFER METRICS ---
# ==============================================================================

def get_point_cloud_from_dsl(dsl_node, points_per_box=1000):
    """Generate a 3D point cloud by sampling boxes from a DSL node.

    This function calls the `.expand()` method of the given DSL node
    (which works for both concrete and `Abstraction` nodes) to get
    the final list of box geometries. It then samples `points_per_box`
    random points within each box.

    Parameters
    ----------
    dsl_node : object
        The root DSL node (original or abstracted).
    points_per_box : int, optional
        The number of points to sample inside each expanded box.

    Returns
    -------
    np.ndarray
        A combined point cloud of shape (N, 3), where N is
        (num_boxes * points_per_box). Returns an empty array
        if expansion fails or yields no boxes.
    """
    if dsl_node is None:
        return np.empty((0, 3))

    try:
        final_boxes = dsl_node.expand()
    except Exception as e:
        print(f"Error during DSL expansion: {e}")
        return np.empty((0, 3))

    if not final_boxes:
        return np.empty((0, 3))

    all_points = []
    for box in final_boxes:
        # 1. Create local points in a [-0.5, 0.5] unit cube
        local_points = np.random.rand(points_per_box, 3) - 0.5
        center = np.array(box["center"])
        lengths = np.array(box["lengths"])
        quaternion = np.array(box["quaternion"])

        # 2. Scale points
        scaled_points = local_points * lengths
        
        # 3. Rotate points
        rotated_points = Rotation.from_quat(quaternion).apply(scaled_points)
        
        # 4. Translate points
        world_points = rotated_points + center
        all_points.append(world_points)

    if not all_points:
        return np.empty((0, 3))
        
    return np.vstack(all_points)


def calculate_chamfer_distance(pc1, pc2):
    """Calculate the symmetric Chamfer distance between two point clouds.

    Uses `scipy.spatial.KDTree` for efficient nearest-neighbor lookup.
    The metric is the mean squared distance from pc1 to pc2 plus the
    mean squared distance from pc2 to pc1.

    Parameters
    ----------
    pc1 : np.ndarray
        Point cloud 1, shape (N, 3).
    pc2 : np.ndarray
        Point cloud 2, shape (M, 3).

    Returns
    -------
    float
        Symmetric Chamfer distance (L2 squared). Returns 0.0
        if either point cloud is empty.
    """
    if pc1.shape[0] == 0 or pc2.shape[0] == 0:
        # If one cloud is empty but not the other, distance is technically
        # infinite, but 0.0 is a reasonable fallback to avoid errors.
        # If both are empty, 0.0 is correct.
        return 0.0

    # Ensure inputs are valid
    pc1 = np.asarray(pc1)
    pc2 = np.asarray(pc2)
    if pc1.ndim != 2 or pc1.shape[1] != 3 or pc2.ndim != 2 or pc2.shape[1] != 3:
        raise ValueError("Point clouds must have shape (N, 3) and (M, 3).")

    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)

    dist1, _ = tree2.query(pc1)
    dist2, _ = tree1.query(pc2)

    # L2-squared (as is common)
    chamfer_dist = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer_dist


def plot_point_clouds_with_k3d(pc1, pc2, point_size=0.01):
    """Visualize two point clouds for comparison using K3D.

    Renders `pc1` as blue points and `pc2` as red points in
    an interactive K3D plot.

    Parameters
    ----------
    pc1 : np.ndarray
        The first point cloud (rendered in blue).
    pc2 : np.ndarray
        The second point cloud (rendered in red).
    point_size : float, optional
        The size of the points in the K3D plot.
    """
    plot = k3d.plot(name="Point Cloud Comparison")
    if pc1.shape[0] > 0:
        plot += k3d.points(
            pc1.astype(np.float32),
            point_size=point_size,
            color=0x0000FF,
            name='Point Cloud 1 (Blue)'
        )
    if pc2.shape[0] > 0:
        plot += k3d.points(
            pc2.astype(np.float32),
            point_size=point_size,
            color=0xFF0000,
            name='Point Cloud 2 (Red)'
        )
    print("Displaying K3D plot. Blue = Point Cloud 1, Red = Point Cloud 2.")
    plot.display()