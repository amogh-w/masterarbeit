"""
dsl_conversion.py

Utilities to convert between hierarchical Tree structures and DSL object trees,
as well as between DSL objects and JSON-serializable dictionaries.
"""

import scipy
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from scipy.spatial.transform import Rotation
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

LABEL_NAMES = {0: "Backrest", 1: "Seat", 2: "Leg", 3: "Armrest"}


class Tree(object):
    """Represents a hierarchical tree structure for 3D shape composition."""

    class NodeType(Enum):
        """Node types in the tree."""
        BOX, ADJ, SYM = 0, 1, 2

    class Node(object):
        """A single node in the Tree."""

        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None, label=None):
            """Initializes a Node.

            Args:
                box: Box tensor for leaf nodes.
                left: Left child node.
                right: Right child node.
                node_type: Type of node (BOX, ADJ, SYM).
                sym: Symmetry parameters for SYM nodes.
                label: Label of the box.
            """
            self.box, self.sym, self.left, self.right, self.node_type, self.label = (
                box,
                sym,
                left,
                right,
                node_type,
                label,
            )

        def is_leaf(self):
            """Returns True if node is a leaf (BOX)."""
            return self.node_type == Tree.NodeType.BOX

        def is_adj(self):
            """Returns True if node is an adjacency (ADJ)."""
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            """Returns True if node is a symmetry (SYM)."""
            return self.node_type == Tree.NodeType.SYM

    def __init__(self, boxes, ops, syms, labels):
        """Constructs a Tree from box, op, sym, and label tensors."""
        box_list = [b for b in torch.split(boxes, 1, 0)]
        sym_param = [s for s in torch.split(syms, 1, 0)]
        label_list = [l for l in labels[0]]
        box_list.reverse()
        sym_param.reverse()
        label_list.reverse()
        queue = []

        for op_id in range(ops.size()[1]):
            if ops[0, op_id] == self.NodeType.BOX.value:
                queue.append(
                    self.Node(
                        box=box_list.pop(),
                        node_type=self.NodeType.BOX,
                        label=label_list.pop(),
                    )
                )
            elif ops[0, op_id] == self.NodeType.ADJ.value:
                right, left = queue.pop(), queue.pop()
                queue.append(
                    self.Node(left=left, right=right, node_type=self.NodeType.ADJ)
                )
            elif ops[0, op_id] == self.NodeType.SYM.value:
                node = queue.pop()
                queue.append(
                    self.Node(
                        left=node, sym=sym_param.pop(), node_type=self.NodeType.SYM
                    )
                )
        self.root = queue[0]


@dataclass
class ShapeData:
    """Holds all shape-related data, including boxes, labels, ops, syms, and the tree."""

    def __init__(self, row):
        """Loads shape data from MATLAB .mat files.

        Args:
            row: An object containing paths for boxes, labels, ops, and syms.
        """
        self.boxes = scipy.io.loadmat(row.boxes)["box"]
        self.labels = scipy.io.loadmat(row.labels)["label"]
        self.ops = scipy.io.loadmat(row.ops)["op"]
        sym_mat = scipy.io.loadmat(row.syms)
        self.syms = sym_mat["sym"]
        self.shapename = str(sym_mat["shapename"].squeeze())
        self.tree = None

    def construct_tree(self):
        """Converts loaded data into a Tree object."""
        boxes_t = torch.tensor(self.boxes, dtype=torch.float).t()
        ops_t = torch.tensor(self.ops, dtype=torch.int)
        syms_t = torch.tensor(self.syms, dtype=torch.float).t()
        labels_t = torch.tensor(self.labels, dtype=torch.int)
        self.tree = Tree(boxes_t, ops_t, syms_t, labels_t)
        print(f"Successfully constructed tree for shape: {self.shapename}")


def tree_to_dsl(node):
    """Recursively converts a Tree node into a corresponding DSL object.

    Args:
        node: Tree.Node object.

    Returns:
        DSL node representing the same structure.
    """
    if node.is_leaf():
        box_vec = node.box.squeeze()
        center = box_vec[0:3].tolist()
        dims = [box_vec[5].item(), box_vec[3].item(), box_vec[4].item()]
        label = node.label.item()
        raw_dir2, raw_dir3 = box_vec[6:9].numpy(), box_vec[9:12].numpy()

        def normalize_vector(vec):
            norm = np.linalg.norm(vec)
            return vec if norm == 0 else vec / norm

        dir2_norm = normalize_vector(raw_dir2)
        dir1_norm = normalize_vector(np.cross(dir2_norm, raw_dir3))
        dir3_norm = normalize_vector(np.cross(dir1_norm, dir2_norm))
        rotation_matrix = np.array([dir1_norm, dir2_norm, dir3_norm]).T
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat().tolist()

        base_box = Box(label=label)
        scaled_box = Scale(child=base_box, lengths=dims)
        rotated_box = Rotate(child=scaled_box, quaternion=quaternion)
        translated_box = Translate(child=rotated_box, center=center)
        return translated_box

    elif node.is_adj():
        left_child_dsl = tree_to_dsl(node.left)
        right_child_dsl = tree_to_dsl(node.right)
        return Union(left=left_child_dsl, right=right_child_dsl)

    elif node.is_sym():
        child_dsl = tree_to_dsl(node.left)
        sym_vec = node.sym.squeeze()
        sym_type = sym_vec[0].item()

        if sym_type == 0.0:
            return SymRef(
                child=child_dsl,
                plane_normal=sym_vec[1:4].tolist(),
                point_on_plane=sym_vec[4:7].tolist(),
            )
        elif sym_type == -1.0:
            n_fold_param = sym_vec[7].item()
            n_fold = int(round(1.0 / n_fold_param)) if n_fold_param != 0 else 1
            return SymRot(
                child=child_dsl,
                axis=sym_vec[1:4].tolist(),
                center=sym_vec[4:7].tolist(),
                n_fold=n_fold,
            )
        elif sym_type == 1.0:
            n_fold_param = sym_vec[7].item()
            n_fold = int(round(1.0 / n_fold_param)) if n_fold_param != 0 else 1
            return SymTrans(
                child=child_dsl, end_point=sym_vec[4:7].tolist(), n_fold=n_fold
            )

    return None


def dsl_to_dict(node):
    """Converts a DSL node into a JSON-serializable dictionary.

    Args:
        node: DSL node object.

    Returns:
        Dictionary representation of the node.
    """
    if isinstance(node, Box):
        return {"type": "Box", "label": node.label}
    elif isinstance(node, Translate):
        return {
            "type": "Translate",
            "center": node.center,
            "child": dsl_to_dict(node.child),
        }
    elif isinstance(node, Rotate):
        return {
            "type": "Rotate",
            "quaternion": node.quaternion,
            "child": dsl_to_dict(node.child),
        }
    elif isinstance(node, Scale):
        return {
            "type": "Scale",
            "lengths": node.lengths,
            "child": dsl_to_dict(node.child),
        }
    elif isinstance(node, Union):
        return {
            "type": "Union",
            "left": dsl_to_dict(node.left),
            "right": dsl_to_dict(node.right),
        }
    elif isinstance(node, SymRef):
        return {
            "type": "SymRef",
            "plane_normal": node.plane,
            "point_on_plane": node.point_on_plane,
            "child": dsl_to_dict(node.child),
        }
    elif isinstance(node, SymRot):
        return {
            "type": "SymRot",
            "axis": node.axis,
            "center": node.center,
            "n_fold": node.n,
            "child": dsl_to_dict(node.child),
        }
    elif isinstance(node, SymTrans):
        return {
            "type": "SymTrans",
            "end_point": node.end_point,
            "n_fold": node.n,
            "child": dsl_to_dict(node.child),
        }
    raise TypeError(f"Object of type {type(node).__name__} is not JSON serializable")


def dict_to_dsl(node_dict):
    """Converts a JSON-like dictionary back into a DSL node.

    Args:
        node_dict: Dictionary representation of a DSL node.

    Returns:
        Corresponding DSL node object.

    Raises:
        TypeError: If the input is not a dictionary.
        ValueError: If the node type is unknown.
    """
    if not isinstance(node_dict, dict):
        raise TypeError("Input must be a dictionary.")
    node_type = node_dict.get("type")
    if node_type == "Translate":
        return Translate(
            child=dict_to_dsl(node_dict["child"]), center=node_dict["center"]
        )
    elif node_type == "Rotate":
        return Rotate(
            child=dict_to_dsl(node_dict["child"]), quaternion=node_dict["quaternion"]
        )
    elif node_type == "Scale":
        return Scale(
            child=dict_to_dsl(node_dict["child"]), lengths=node_dict["lengths"]
        )
    elif node_type == "Box":
        return Box(label=node_dict["label"])
    elif node_type == "Union":
        return Union(
            left=dict_to_dsl(node_dict["left"]), right=dict_to_dsl(node_dict["right"])
        )
    elif node_type == "SymRef":
        return SymRef(
            child=dict_to_dsl(node_dict["child"]),
            plane_normal=node_dict["plane_normal"],
            point_on_plane=node_dict["point_on_plane"],
        )
    elif node_type == "SymRot":
        return SymRot(
            child=dict_to_dsl(node_dict["child"]),
            axis=node_dict["axis"],
            center=node_dict["center"],
            n_fold=node_dict["n_fold"],
        )
    elif node_type == "SymTrans":
        return SymTrans(
            child=dict_to_dsl(node_dict["child"]),
            end_point=node_dict["end_point"],
            n_fold=node_dict["n_fold"],
        )
    else:
        raise ValueError(f"Unknown node type found in JSON: '{node_type}'")


def parse_json_to_dsl(json_string):
    """Parses a JSON string into a DSL node.

    Args:
        json_string: JSON string representing a DSL structure.

    Returns:
        DSL node object.
    """
    import json

    return dict_to_dsl(json.loads(json_string))
