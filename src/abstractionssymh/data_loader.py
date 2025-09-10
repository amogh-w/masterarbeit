import scipy
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from abstractionssymh.dsl_nodes import Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans

# Global mapping for part labels
LABEL_NAMES = {0: "Backrest", 1: "Seat", 2: "Leg", 3: "Armrest"}


class Tree(object):
    class NodeType(Enum):
        BOX, ADJ, SYM = 0, 1, 2

    class Node(object):
        def __init__(
            self, box=None, left=None, right=None, node_type=None, sym=None, label=None
        ):
            self.box, self.sym, self.left, self.right, self.node_type, self.label = (
                box,
                sym,
                left,
                right,
                node_type,
                label,
            )

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX

        def is_adj(self):
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            return self.node_type == Tree.NodeType.SYM

    def __init__(self, boxes, ops, syms, labels):
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
    """
    A self-contained class that loads all data for a single shape,
    constructs its hierarchical tree, and expands all symmetries.
    """

    def __init__(self, row):
        # Load all raw data from files
        self.boxes = scipy.io.loadmat(row.boxes)["box"]
        self.labels = scipy.io.loadmat(row.labels)["label"]
        self.ops = scipy.io.loadmat(row.ops)["op"]
        sym_mat = scipy.io.loadmat(row.syms)
        self.syms = sym_mat["sym"]
        self.shapename = str(sym_mat["shapename"].squeeze())
        self.tree = None

    def construct_tree(self):
        """Uses the loaded raw data to build the Tree object."""
        boxes_t = torch.tensor(self.boxes, dtype=torch.float).t()
        ops_t = torch.tensor(self.ops, dtype=torch.int)
        syms_t = torch.tensor(self.syms, dtype=torch.float).t()
        labels_t = torch.tensor(self.labels, dtype=torch.int)
        self.tree = Tree(boxes_t, ops_t, syms_t, labels_t)
        print(f"Successfully constructed tree for shape: {self.shapename}")


def tree_to_dsl(node):
    """
    Converts a node from the original Tree structure into its equivalent
    DSL object representation using explicit transformation nodes.
    """
    if node.is_leaf():
        box_vec = node.box.squeeze()

        # 1. Extract raw parameters
        center = box_vec[0:3].tolist()
        dims = [box_vec[5].item(), box_vec[3].item(), box_vec[4].item()]
        label = node.label.item()
        raw_dir2, raw_dir3 = box_vec[6:9].numpy(), box_vec[9:12].numpy()

        # 2. Calculate rotation quaternion from raw vectors
        def normalize_vector(vec):
            norm = np.linalg.norm(vec)
            return vec if norm == 0 else vec / norm

        dir2_norm = normalize_vector(raw_dir2)
        dir1_norm = normalize_vector(np.cross(dir2_norm, raw_dir3))
        dir3_norm = normalize_vector(np.cross(dir1_norm, dir2_norm))

        rotation_matrix = np.array([dir1_norm, dir2_norm, dir3_norm]).T
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat().tolist()

        # 3. Build the new hierarchy: Translate(Rotate(Scale(Box)))
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

        if sym_type == 0.0:  # Reflection
            plane_normal = sym_vec[1:4].tolist()
            point_on_plane = sym_vec[4:7].tolist()
            return SymRef(
                child=child_dsl,
                plane_normal=plane_normal,
                point_on_plane=point_on_plane,
            )
        elif sym_type == -1.0:  # Rotation
            axis = sym_vec[1:4].tolist()
            center = sym_vec[4:7].tolist()
            n_fold_param = sym_vec[7].item()
            n_fold = int(round(1.0 / n_fold_param)) if n_fold_param != 0 else 1
            return SymRot(child=child_dsl, axis=axis, center=center, n_fold=n_fold)
        elif sym_type == 1.0:  # Translation
            end_point = sym_vec[4:7].tolist()
            n_fold_param = sym_vec[7].item()
            n_fold = int(round(1.0 / n_fold_param)) if n_fold_param != 0 else 1
            return SymTrans(child=child_dsl, end_point=end_point, n_fold=n_fold)

    return None


def dsl_to_dict(node):
    """
    Recursively converts a DSL object tree into a JSON-serializable dictionary.
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
    """
    Recursively traverses a dictionary and converts it into the corresponding
    DSL object tree.
    """
    if not isinstance(node_dict, dict):
        raise TypeError("Input must be a dictionary.")

    node_type = node_dict.get("type")

    if node_type == "Translate":
        child_obj = dict_to_dsl(node_dict["child"])
        return Translate(child=child_obj, center=node_dict["center"])
    elif node_type == "Rotate":
        child_obj = dict_to_dsl(node_dict["child"])
        return Rotate(child=child_obj, quaternion=node_dict["quaternion"])
    elif node_type == "Scale":
        child_obj = dict_to_dsl(node_dict["child"])
        return Scale(child=child_obj, lengths=node_dict["lengths"])
    elif node_type == "Box":
        return Box(label=node_dict["label"])
    elif node_type == "Union":
        left_child = dict_to_dsl(node_dict["left"])
        right_child = dict_to_dsl(node_dict["right"])
        return Union(left=left_child, right=right_child)
    elif node_type == "SymRef":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymRef(
            child=child_obj,
            plane_normal=node_dict["plane_normal"],
            point_on_plane=node_dict["point_on_plane"],
        )
    elif node_type == "SymRot":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymRot(
            child=child_obj,
            axis=node_dict["axis"],
            center=node_dict["center"],
            n_fold=node_dict["n_fold"],
        )
    elif node_type == "SymTrans":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymTrans(
            child=child_obj,
            end_point=node_dict["end_point"],
            n_fold=node_dict["n_fold"],
        )
    else:
        raise ValueError(f"Unknown node type found in JSON: '{node_type}'")


def parse_json_to_dsl(json_string):
    """
    A convenient wrapper function that parses a JSON string into a DSL object tree.
    """
    import json

    data_dictionary = json.loads(json_string)
    return dict_to_dsl(data_dictionary)
