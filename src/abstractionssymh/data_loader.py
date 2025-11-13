"""dsl_conversion.py

Utilities to convert between hierarchical Tree structures and DSL object trees,
as well as between DSL objects and JSON-serializable dictionaries.

This module provides the bridges between:
1.  Raw tensor/MATLAB data (boxes, ops, syms, labels) and a
    hierarchical `Tree` structure.
2.  The `Tree` structure and the object-oriented DSL node hierarchy
    (from `dsl_nodes.py`).
3.  The DSL node hierarchy and JSON-serializable dictionaries for
    storage or transmission.

Classes
-------
- Tree
- ShapeData

Functions
---------
- tree_to_dsl(node)
- dsl_to_dict(node)
- dict_to_dsl(node_dict)
- parse_json_to_dsl(json_string)
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
"""dict: Maps integer part labels to human-readable names."""


class Tree(object):
    """Represents a hierarchical tree structure for 3D shape composition.

    This class parses raw tensor data (boxes, ops, syms, labels) from the
    original dataset format into a nested Python object structure. The tree
    is built using a queue-based, post-order traversal logic based on
    the `ops` tensor.

    Attributes
    ----------
    root : Tree.Node
        The root node of the constructed tree.

    Nested Classes
    --------------
    NodeType : Enum
        Defines the types of nodes: BOX, ADJ (adjacency/union), SYM (symmetry).
    Node : object
        Represents a single node within the tree.
    """

    class NodeType(Enum):
        """Enumeration of node types within the `Tree` structure."""
        BOX, ADJ, SYM = 0, 1, 2

    class Node(object):
        """A single node in the Tree.

        This node can be a leaf (BOX), a binary operator (ADJ),
        or a unary operator (SYM).

        Attributes
        ----------
        box : torch.Tensor or None
            Tensor data for a BOX leaf node.
        left : Tree.Node or None
            The left (or only) child node.
        right : Tree.Node or None
            The right child node (for ADJ nodes).
        node_type : Tree.NodeType
            The type of this node.
        sym : torch.Tensor or None
            Tensor data for a SYM node.
        label : torch.Tensor or None
            Label for a BOX leaf node.
        """

        def __init__(
            self,
            box=None,
            left=None,
            right=None,
            node_type=None,
            sym=None,
            label=None,
        ):
            """Initialize a Node.

            Parameters
            ----------
            box : torch.Tensor, optional
                Box tensor for leaf nodes.
            left : Tree.Node, optional
                Left child node.
            right : Tree.Node, optional
                Right child node.
            node_type : Tree.NodeType, optional
                Type of node (BOX, ADJ, SYM).
            sym : torch.Tensor, optional
                Symmetry parameters for SYM nodes.
            label : torch.Tensor, optional
                Label of the box.
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
            """Check if this node is a leaf node (BOX).

            Returns
            -------
            bool
                True if node type is BOX, False otherwise.
            """
            return self.node_type == Tree.NodeType.BOX

        def is_adj(self):
            """Check if this node is an adjacency node (ADJ).

            Returns
            -------
            bool
                True if node type is ADJ, False otherwise.
            """
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            """Check if this node is a symmetry node (SYM).

            Returns
            -------
            bool
                True if node type is SYM, False otherwise.
            """
            return self.node_type == Tree.NodeType.SYM

    def __init__(self, boxes, ops, syms, labels):
        """Construct a Tree from raw box, op, sym, and label tensors.

        This constructor implements a queue-based algorithm to build the
        tree structure from the flat list of operations (`ops`). It
        processes the operations in reverse order to build the tree
        from leaves up to the root.

        Parameters
        ----------
        boxes : torch.Tensor
            A tensor of all box parameters.
        ops : torch.Tensor
            A tensor defining the tree structure operations (0=BOX, 1=ADJ, 2=SYM).
        syms : torch.Tensor
            A tensor of all symmetry parameters.
        labels : torch.Tensor
            A tensor of all box labels.
        """
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
    """Holds all shape-related data, loading from `.mat` files.

    This class acts as a data loader and container, parsing MATLAB files
    for a single shape and providing a method to construct the
    hierarchical `Tree` object from them.

    Attributes
    ----------
    boxes : np.ndarray
        Raw box data loaded from file.
    labels : np.ndarray
        Raw label data loaded from file.
    ops : np.ndarray
        Raw operation data loaded from file.
    syms : np.ndarray
        Raw symmetry data loaded from file.
    shapename : str
        The name of the shape, extracted from the syms file.
    tree : Tree or None
        The hierarchical `Tree` object, or None if
        `construct_tree()` has not been called.
    """

    def __init__(self, row):
        """Load shape data from paths specified in a row object.

        Assumes `row` is an object with attributes `boxes`, `labels`,
        `ops`, and `syms`, each containing a file path to a `.mat` file.

        Parameters
        ----------
        row : object
            An object (e.g., a pandas Series or namedtuple) containing
            file paths for boxes, labels, ops, and syms.
        """
        self.boxes = scipy.io.loadmat(row.boxes)["box"]
        self.labels = scipy.io.loadmat(row.labels)["label"]
        self.ops = scipy.io.loadmat(row.ops)["op"]
        sym_mat = scipy.io.loadmat(row.syms)
        self.syms = sym_mat["sym"]
        self.shapename = str(sym_mat["shapename"].squeeze())
        self.tree = None

    def construct_tree(self):
        """Convert loaded NumPy arrays into PyTorch tensors and build the `Tree`.

        Populates the `self.tree` attribute with a `Tree` object based on
        the loaded file data.
        """
        boxes_t = torch.tensor(self.boxes, dtype=torch.float).t()
        ops_t = torch.tensor(self.ops, dtype=torch.int)
        syms_t = torch.tensor(self.syms, dtype=torch.float).t()
        labels_t = torch.tensor(self.labels, dtype=torch.int)
        self.tree = Tree(boxes_t, ops_t, syms_t, labels_t)
        print(f"Successfully constructed tree for shape: {self.shapename}")


def tree_to_dsl(node):
    """Recursively convert a `Tree.Node` into a corresponding DSL object.

    This is the main conversion from the raw `Tree` structure
    (loaded from tensors) into the symbolic, object-oriented `dsl_nodes`
    representation.

    - `BOX` nodes are converted into a `Translate(Rotate(Scale(Box)))` stack.
    - `ADJ` nodes are converted into a `Union`.
    - `SYM` nodes are converted into `SymRef`, `SymRot`, or `SymTrans`.

    Parameters
    ----------
    node : Tree.Node
        The `Tree.Node` object to convert.

    Returns
    -------
    object or None
        A DSL node (e.g., `Box`, `Translate`, `Union`, `SymRef`)
        corresponding to the input `Tree.Node`. Returns `None`
        for unhandled node types.
    """
    if node.is_leaf():
        box_vec = node.box.squeeze()
        center = box_vec[0:3].tolist()
        # Note: Original vector order is [?, ?, ?, y, z, x, dir2..., dir3...]
        dims = [box_vec[5].item(), box_vec[3].item(), box_vec[4].item()]
        label = node.label.item()
        raw_dir2, raw_dir3 = box_vec[6:9].numpy(), box_vec[9:12].numpy()

        def normalize_vector(vec):
            norm = np.linalg.norm(vec)
            return vec if norm == 0 else vec / norm

        # Reconstruct orthonormal basis (rotation matrix)
        dir2_norm = normalize_vector(raw_dir2)
        dir1_norm = normalize_vector(np.cross(dir2_norm, raw_dir3))
        dir3_norm = normalize_vector(np.cross(dir1_norm, dir2_norm))
        rotation_matrix = np.array([dir1_norm, dir2_norm, dir3_norm]).T
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat().tolist()

        # Build the DSL node stack
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

        if sym_type == 0.0:  # Reflection Symmetry
            return SymRef(
                child=child_dsl,
                plane_normal=sym_vec[1:4].tolist(),
                point_on_plane=sym_vec[4:7].tolist(),
            )
        elif sym_type == -1.0:  # Rotational Symmetry
            n_fold_param = sym_vec[7].item()
            n_fold = int(round(1.0 / n_fold_param)) if n_fold_param != 0 else 1
            return SymRot(
                child=child_dsl,
                axis=sym_vec[1:4].tolist(),
                center=sym_vec[4:7].tolist(),
                n_fold=n_fold,
            )
        elif sym_type == 1.0:  # Translational Symmetry
            n_fold_param = sym_vec[7].item()
            n_fold = int(round(1.0 / n_fold_param)) if n_fold_param != 0 else 1
            return SymTrans(
                child=child_dsl, end_point=sym_vec[4:7].tolist(), n_fold=n_fold
            )

    return None


def dsl_to_dict(node):
    """Convert a DSL node into a serializable dictionary.

    Recursively traverses the DSL object tree and converts each node
    (e.g., `Box`, `Scale`) into a dictionary with a "type" field
    and associated parameters.

    Note
    ----
    This function stores parameters (e.g., `center`, `lengths`)
    as `np.ndarray` objects. For pure JSON serialization, these
    arrays must be converted to lists (e.g., using `.tolist()`)
    or a custom JSON encoder is required.

    Parameters
    ----------
    node : object
        A DSL node object from `dsl_nodes` (e.g., `Box`, `Scale`).

    Returns
    -------
    dict
        A nested dictionary representation of the DSL node.

    Raises
    ------
    TypeError
        If an object of an unknown type is encountered.
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
    """Convert a dictionary representation back into a DSL node.

    Recursively reconstructs the DSL object tree from a nested
    dictionary (presumably from `dsl_to_dict` or JSON). Assumes
    parameter values (e.g., `center`) are in a format
    (like list or `np.ndarray`) consumable by the DSL node constructors.

    Parameters
    ----------
    node_dict : dict
        Dictionary representation of a DSL node.

    Returns
    -------
    object
        The corresponding DSL node object (e.g., `Box`, `Translate`).

    Raises
    ------
    TypeError
        If the input `node_dict` is not a dictionary.
    ValueError
        If the dictionary contains an unknown "type" field.
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
    """Parse a JSON string into a DSL node.

    A simple wrapper that calls `json.loads` and then `dict_to_dsl`.
    This assumes the JSON string does not contain `np.ndarray`s
    and that all arrays are represented as standard JSON lists.

    Parameters
    ----------
    json_string : str
        A JSON string representing a DSL structure.

    Returns
    -------
    object
        The corresponding DSL node object.
    """
    import json

    return dict_to_dsl(json.loads(json_string))