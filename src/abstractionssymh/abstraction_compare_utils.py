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
    """Recursively counts the total number of operational nodes in a DSL tree.

    Args:
        node: The root DSL node.

    Returns:
        int: Total number of operational nodes.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return 0

    count = 1  # Count the current node
    children = node.children if isinstance(node, Abstraction) else node.serialize()[1][1] if hasattr(node, "serialize") else []

    for child in children:
        count += _count_nodes_recursive(child)
    return count


def _get_max_depth_recursive(node, current_depth):
    """Recursively finds the maximum depth of a DSL tree.

    Args:
        node: The root DSL node.
        current_depth: Current recursion depth.

    Returns:
        int: Maximum depth of the tree.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return current_depth

    children = node.children if isinstance(node, Abstraction) else node.serialize()[1][1] if hasattr(node, "serialize") else []

    if not children or all(not hasattr(c, "serialize") and not isinstance(c, Abstraction) for c in children):
        return current_depth + 1

    return max(_get_max_depth_recursive(c, current_depth + 1) for c in children)


def _count_parameters_recursive(node, is_abstracted_tree=False):
    """Recursively counts parameters in a DSL tree.

    Args:
        node: DSL node (original or abstracted).
        is_abstracted_tree (bool): If True, use compressed parameters for Abstraction nodes.

    Returns:
        int: Total number of parameters.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return 0

    if isinstance(node, Abstraction):
        params_count = len(node.compressed_params) if is_abstracted_tree else 0
        children = node.children
    else:
        params_count, children = len(node.serialize()[1][0]), node.serialize()[1][1]

    for child in children:
        params_count += _count_parameters_recursive(child, is_abstracted_tree)
    return params_count


def _count_replaced_nodes_recursive(node):
    """Counts all original nodes replaced by Abstraction nodes in a DSL tree.

    Args:
        node: DSL node (original or abstracted).

    Returns:
        int: Total number of nodes replaced by abstractions.
    """
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return 0

    total_replaced = 0
    children_to_scan = node.children if isinstance(node, Abstraction) else node.serialize()[1][1] if hasattr(node, "serialize") else []

    if isinstance(node, Abstraction):
        if '(' in node.pattern_name and ')' in node.pattern_name:
            total_replaced += 2  # Pair node
        else:
            total_replaced += 1  # Singleton node

    for child in children_to_scan:
        total_replaced += _count_replaced_nodes_recursive(child)
    return total_replaced


# ==============================================================================
# --- METRIC 1: Tree Complexity Reduction ---
# ==============================================================================

def calculate_tree_complexity_reduction(original_chair, abstracted_chair):
    """Calculates percentage reduction in node count and tree depth.

    Args:
        original_chair: Root node of the original DSL tree.
        abstracted_chair: Root node of the abstracted DSL tree.

    Returns:
        dict: Node count and depth reduction metrics with details.
    """
    original_nodes = _count_nodes_recursive(original_chair)
    abstracted_nodes = _count_nodes_recursive(abstracted_chair)
    original_depth = _get_max_depth_recursive(original_chair, 0)
    abstracted_depth = _get_max_depth_recursive(abstracted_chair, 0)

    node_reduction = (original_nodes - abstracted_nodes) / original_nodes if original_nodes else 0
    depth_reduction = (original_depth - abstracted_depth) / original_depth if original_depth else 0

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
    """Calculates the reduction ratio of parameters after abstraction.

    Args:
        original_chair: Root DSL node of the original shape.
        abstracted_chair: Root DSL node of the abstracted shape.

    Returns:
        dict: Parameter compression ratio with details.
    """
    original_params = _count_parameters_recursive(original_chair, is_abstracted_tree=False)
    abstracted_params = _count_parameters_recursive(abstracted_chair, is_abstracted_tree=True)
    ratio = (original_params - abstracted_params) / original_params if original_params else 0

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
    """Calculates percentage of original nodes replaced by abstractions.

    Args:
        original_chair: Root DSL node of the original shape.
        abstracted_chair: Root DSL node of the abstracted shape.

    Returns:
        dict: Abstraction coverage metric with details.
    """
    original_op_nodes = sum(1 for node in _traverse(original_chair) if not isinstance(node, Box))
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


def _traverse(node):
    """Generator to iterate through all nodes in a tree."""
    if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
        return
    yield node
    children = node.children if isinstance(node, Abstraction) else node.serialize()[1][1] if hasattr(node, "serialize") else []
    for child in children:
        yield from _traverse(child)


# ==============================================================================
# --- POINT CLOUD & CHAMFER METRICS ---
# ==============================================================================

def get_point_cloud_from_dsl(dsl_node, points_per_box=1000):
    """Generates a 3D point cloud by sampling boxes from a DSL node.

    Args:
        dsl_node: Root DSL node (original or abstracted).
        points_per_box: Number of points per box.

    Returns:
        np.ndarray: Combined point cloud of shape (N, 3).
    """
    if dsl_node is None:
        return np.empty((0, 3))

    final_boxes = dsl_node.expand()
    if not final_boxes:
        return np.empty((0, 3))

    all_points = []
    for box in final_boxes:
        local_points = np.random.rand(points_per_box, 3) - 0.5
        center, lengths, quaternion = box["center"], box["lengths"], box["quaternion"]
        scaled_points = local_points * lengths
        rotated_points = Rotation.from_quat(quaternion).apply(scaled_points)
        world_points = rotated_points + center
        all_points.append(world_points)

    return np.vstack(all_points)


def calculate_chamfer_distance(pc1, pc2):
    """Calculates the symmetric Chamfer distance between two point clouds.

    Args:
        pc1 (np.ndarray): Point cloud 1, shape (N, 3).
        pc2 (np.ndarray): Point cloud 2, shape (M, 3).

    Returns:
        float: Symmetric Chamfer distance.
    """
    if pc1.shape[0] == 0 or pc2.shape[0] == 0:
        return 0.0

    tree1, tree2 = KDTree(pc1), KDTree(pc2)
    dist1, _ = tree2.query(pc1)
    dist2, _ = tree1.query(pc2)
    return np.mean(dist1**2) + np.mean(dist2**2)


def plot_point_clouds_with_k3d(pc1, pc2, point_size=0.01):
    """Visualizes two point clouds for comparison using K3D.

    Args:
        pc1 (np.ndarray): First point cloud (blue).
        pc2 (np.ndarray): Second point cloud (red).
        point_size (float): Point size for visualization.
    """
    plot = k3d.plot(name="Point Cloud Comparison")
    if pc1.shape[0] > 0:
        plot += k3d.points(pc1, point_size=point_size, color=0x0000FF, name='Point Cloud 1')
    if pc2.shape[0] > 0:
        plot += k3d.points(pc2, point_size=point_size, color=0xFF0000, name='Point Cloud 2')
    print("Displaying K3D plot. Blue = Point Cloud 1, Red = Point Cloud 2.")
    plot.display()
