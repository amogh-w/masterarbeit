#!/usr/bin/env python
# coding: utf-8

# In[41]:


import scipy
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
from dataclasses import dataclass
import numpy as np
from enum import Enum
import torch
from scipy.spatial.transform import Rotation
import k3d


# In[42]:


class Tree(object):
    class NodeType(Enum):
        BOX, ADJ, SYM = 0, 1, 2

    class Node(object):
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None, label=None):
            self.box, self.sym, self.left, self.right, self.node_type, self.label = box, sym, left, right, node_type, label
        def is_leaf(self): return self.node_type == Tree.NodeType.BOX
        def is_adj(self): return self.node_type == Tree.NodeType.ADJ
        def is_sym(self): return self.node_type == Tree.NodeType.SYM

    def __init__(self, boxes, ops, syms, labels):
        box_list = [b for b in torch.split(boxes, 1, 0)]
        sym_param = [s for s in torch.split(syms, 1, 0)]
        label_list = [l for l in labels[0]]
        box_list.reverse(); sym_param.reverse(); label_list.reverse()
        queue = []
        for op_id in range(ops.size()[1]):
            if ops[0, op_id] == self.NodeType.BOX.value:
                queue.append(self.Node(box=box_list.pop(), node_type=self.NodeType.BOX, label=label_list.pop()))
            elif ops[0, op_id] == self.NodeType.ADJ.value:
                right, left = queue.pop(), queue.pop()
                queue.append(self.Node(left=left, right=right, node_type=self.NodeType.ADJ))
            elif ops[0, op_id] == self.NodeType.SYM.value:
                node = queue.pop()
                queue.append(self.Node(left=node, sym=sym_param.pop(), node_type=self.NodeType.SYM))
        self.root = queue[0]

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def format_box(bv):
    center = bv[0:3]
    lengths = [bv[5], bv[3], bv[4]] # Reorder to x,y,z
    raw_dir2, raw_dir3 = bv[6:9], bv[9:12]

    dir2 = normalize_vector(raw_dir2)
    dir1 = normalize_vector(np.cross(dir2, raw_dir3))
    dir3 = normalize_vector(np.cross(dir1, dir2))

    rotation_matrix = np.array([dir1, dir2, dir3]).T
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

    return {'center': center, 'lengths': lengths, 'quaternion': quaternion}


# In[43]:


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
        self.expanded_tree = []

    def construct_tree(self):
        """Uses the loaded raw data to build the Tree object."""
        boxes_t = torch.tensor(self.boxes, dtype=torch.float).t()
        ops_t = torch.tensor(self.ops, dtype=torch.int)
        syms_t = torch.tensor(self.syms, dtype=torch.float).t()
        labels_t = torch.tensor(self.labels, dtype=torch.int)
        self.tree = Tree(boxes_t, ops_t, syms_t, labels_t)
        print(f"Successfully constructed tree for shape: {self.shapename}")

    def expand_tree(self):
        self.expanded_tree = _expand_tree_recursive(self.tree.root)
        print(f"Tree expanded (vectors). Found {len(self.expanded_tree)} final boxes.")


def _expand_tree_recursive(node):
    """
    A clean, recursive function to expand the tree nodes. This version
    uses quaternions for orientation storage.
    """
    if node.is_leaf():
        box_values = node.box.squeeze().tolist()
        box_dict = format_box(box_values)
        box_dict['label_id'] = node.label.item()
        box_dict['label_name'] = label_names.get(node.label.item(), "Unknown")
        return [box_dict]

    elif node.is_adj():
        left_boxes = _expand_tree_recursive(node.left) if node.left else []
        right_boxes = _expand_tree_recursive(node.right) if node.right else []
        return left_boxes + right_boxes

    elif node.is_sym():
        child_boxes = _expand_tree_recursive(node.left) if node.left else []
        sym_values = node.sym.squeeze().tolist()
        sym_type = sym_values[0]
        generated_boxes = []

        if sym_type == 0.0: # Reflection
            plane_normal = np.array(sym_values[1:4])
            point_on_plane = np.array(sym_values[4:7])
            for box in child_boxes:
                reflected_box = box.copy()
                vec_to_plane = box['center'] - point_on_plane
                reflected_box['center'] = box['center'] - 2 * np.dot(vec_to_plane, plane_normal) * plane_normal

                # Convert quaternion to matrix, reflect, convert back
                R_orig = Rotation.from_quat(box['quaternion']).as_matrix()
                M_reflect = np.identity(3) - 2 * np.outer(plane_normal, plane_normal)
                R_new = M_reflect @ R_orig
                R_new[:, 0] *= -1 # Fix handedness for valid quaternion
                reflected_box['quaternion'] = Rotation.from_matrix(R_new).as_quat()

                generated_boxes.append(reflected_box)

        elif sym_type == -1.0: # Rotation
            axis = normalize_vector(np.array(sym_values[1:4]))
            center = np.array(sym_values[4:7])
            n_fold = int(round(1.0 / sym_values[7])) if sym_values[7] != 0 else 1
            for i in range(1, n_fold):
                angle = 2 * np.pi * i / n_fold
                symmetry_rot = Rotation.from_rotvec(angle * axis)
                for box in child_boxes:
                    rotated_box = box.copy()
                    vec_from_center = box['center'] - center
                    rotated_box['center'] = center + symmetry_rot.apply(vec_from_center)

                    # Combine rotations using quaternion multiplication
                    original_rot = Rotation.from_quat(box['quaternion'])
                    rotated_box['quaternion'] = (symmetry_rot * original_rot).as_quat()

                    generated_boxes.append(rotated_box)

        elif sym_type == 1.0: # Translation
            end_point = np.array(sym_values[4:7])
            n_fold = int(round(1.0 / sym_values[7])) if sym_values[7] != 0 else 1
            if n_fold > 1:
                for box in child_boxes:
                    start_point = box['center']
                    step_vector = (end_point - start_point) / (n_fold - 1)
                    for i in range(1, n_fold):
                        translated_box = box.copy()
                        translated_box['center'] = start_point + i * step_vector
                        # Orientation doesn't change in translation, so just copy quaternion
                        translated_box['quaternion'] = box['quaternion']
                        generated_boxes.append(translated_box)

        return child_boxes + generated_boxes

    return []


# In[44]:


label_names = {0: "Backrest", 1: "Seat", 2: "Leg", 3: "Armrest"}

def print_full_tree(node, prefix="", is_last=True):
    branch = "└─ " if is_last else "├─ "
    if node.is_leaf():
        part_name = label_names.get(node.label.item(), f"Unknown({node.label.item()})")
        box_values = node.box.squeeze().tolist()
        box_dict = format_box(box_values)
        print(f"{prefix}{branch}Leaf: {part_name}, Label={node.label.item()}, Box={box_dict}")
    elif node.is_adj():
        print(f"{prefix}{branch}Adjacency Node")
        children = [node.left, node.right]
        for i, child in enumerate(children):
            print_full_tree(child, prefix + ("    " if is_last else "│   "), i == len(children)-1)
    elif node.is_sym():
        sym_values = node.sym.squeeze().tolist()
        print(f"{prefix}{branch}Symmetry Node: Sym Params={[round(x,6) for x in sym_values]}")
        print_full_tree(node.left, prefix + ("    " if is_last else "│   "), True)


# In[45]:


df = pd.read_csv("dataset_overview.csv", index_col="shape")
shape_row = df.sample(n=1).iloc[0]

# 1. Instantiate ShapeData. Tree construction and symmetry
#    expansion happen automatically.
my_shape = ShapeData(shape_row)


# In[46]:


my_shape.construct_tree()
print_full_tree(my_shape.tree.root)


# In[47]:


my_shape.expand_tree()
my_shape.expanded_tree


# In[48]:


import numpy as np
import k3d

# fixed color map for label ids
LABEL_COLORS = {
    0: 0xFF0000,  # red   (Backrest)
    1: 0x00FF00,  # green (Seat)
    2: 0x0000FF,  # blue  (Leg)
    3: 0xFFFF00,  # yellow (Armrest)
}

def plot_boxes(box_list):
    """
    Plots a list of box dictionaries that use QUATERNIONS for orientation.
    """
    plot = k3d.plot(name="Expanded Shape Visualization")

    for box in box_list:
        center = np.array(box["center"])
        lengths = np.array(box["lengths"])
        quaternion = np.array(box["quaternion"])
        label_id = box.get("label_id", -1)

        # 1. Convert quaternion back to rotation matrix (dir vectors)
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        dir1 = rotation_matrix[:, 0]
        dir2 = rotation_matrix[:, 1]
        dir3 = rotation_matrix[:, 2]

        # 2. Calculate the 8 corner points using the dir vectors
        d1 = 0.5 * lengths[0] * dir1
        d2 = 0.5 * lengths[1] * dir2
        d3 = 0.5 * lengths[2] * dir3

        corners = np.array([
            center - d1 - d2 - d3, center - d1 + d2 - d3,
            center + d1 - d2 - d3, center + d1 + d2 - d3,
            center - d1 - d2 + d3, center - d1 + d2 + d3,
            center + d1 - d2 + d3, center + d1 + d2 + d3
        ], dtype=np.float32)

        # 3. Define the 12 triangular faces of the box
        faces = np.array([
            [0,1,3], [0,3,2], [4,6,7], [4,7,5],
            [0,2,6], [0,6,4], [1,5,7], [1,7,3],
            [0,4,5], [0,5,1], [2,3,7], [2,7,6]
        ], dtype=np.uint32)

        color = LABEL_COLORS.get(label_id, 0x808080) # Fallback to gray
        plot += k3d.mesh(corners, faces, color=color, opacity=1.0)

    plot.display()

plot_boxes(my_shape.expanded_tree)


# In[49]:


import torch
import textwrap
import numpy as np
from scipy.spatial.transform import Rotation as R

# Helper to format vectors for printing
def _format_vec(vec, precision=3):
    """Formats a list of numbers into a clean string like '[1.234, 5.678]'."""
    return f"[{', '.join(f'{x:.{precision}f}' for x in vec)}]"

# Label mapping for better readability
LABEL_NAMES = {0: "Backrest", 1: "Seat", 2: "Leg", 3: "Armrest"}

# --- LEAF NODE ---
class Box:
    def __init__(self, label: int, center, dims, quaternion):
        self.label = label
        self.center = center
        self.lengths = dims
        self.quaternion = quaternion # Store the quaternion directly

    def __str__(self):
        # The __str__ method now simply reads the stored quaternion
        # label_name = LABEL_NAMES.get(self.label, "Unknown")
        info = (
            # f"Box(label={self.label} '{label_name}',\n"
            f"Box(label={self.label},\n"
            f"    center={_format_vec(self.center)},\n"
            f"    lengths={_format_vec(self.lengths)},\n"
            f"    quat={_format_vec(self.quaternion, precision=4)}\n)"
        )
        return textwrap.indent(info, "    ").lstrip()

# --- INTERNAL NODES ---
class Union:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        left_str = textwrap.indent(str(self.left), "    ")
        right_str = textwrap.indent(str(self.right), "    ")
        return f"Union(\n{left_str},\n{right_str}\n)"

class Symmetry:
    def __init__(self, child):
        self.child = child

class SymRef(Symmetry):
    def __init__(self, child, plane_normal, point_on_plane):
        super().__init__(child)
        self.plane = plane_normal
        self.point_on_plane = point_on_plane

    def __str__(self):
        info = (
            f"SymRef(\n"
            f"    plane={_format_vec(self.plane)},\n"
            f"    point_on_plane={_format_vec(self.point_on_plane)}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"

class SymRot(Symmetry):
    def __init__(self, child, axis, center, n_fold: int):
        super().__init__(child)
        self.axis = axis
        self.center = center
        self.n = n_fold

    def __str__(self):
        info = (
            f"SymRot(\n"
            f"    axis={_format_vec(self.axis)},\n"
            f"    center={_format_vec(self.center)},\n"
            f"    n={self.n}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"

class SymTrans(Symmetry):
    def __init__(self, child, end_point, n_fold: int):
        super().__init__(child)
        self.end_point = end_point
        self.n = n_fold
        # Note: The translation_vector is a step vector calculated during expansion,
        # not a fixed property. The end_point is the defining parameter.

    def __str__(self):
        info = (
            f"SymTrans(\n"
            # The 'end_point' defines the goal of the symmetric translation.
            f"    end_point={_format_vec(self.end_point)},\n"
            f"    n={self.n}\n)"
        )
        child_str = textwrap.indent(str(self.child), "    ")
        return f"{info}(\n{child_str}\n)"


# In[50]:


from scipy.spatial.transform import Rotation as R

def tree_to_dsl(node):
    """
    Converts a node from the original Tree structure into its equivalent
    DSL object representation, now using quaternions directly.
    """
    # Base Case: The node is a leaf (a primitive box)
    if node.is_leaf():
        box_vec = node.box.squeeze()

        center = box_vec[0:3].tolist()
        dims = [box_vec[5].item(), box_vec[3].item(), box_vec[4].item()]
        label = node.label.item()

        # --- THIS IS THE MODIFIED PART ---
        # Calculate the quaternion from the raw direction vectors
        raw_dir2, raw_dir3 = box_vec[6:9].numpy(), box_vec[9:12].numpy()

        dir2_norm = raw_dir2 / (np.linalg.norm(raw_dir2) + 1e-8)
        dir1_norm = np.cross(dir2_norm, raw_dir3)
        dir1_norm /= (np.linalg.norm(dir1_norm) + 1e-8)
        dir3_norm = np.cross(dir1_norm, dir2_norm)

        rotation_matrix = np.array([dir1_norm, dir2_norm, dir3_norm]).T
        quaternion = R.from_matrix(rotation_matrix).as_quat().tolist()
        # --- END OF MODIFICATION ---

        # Pass the calculated quaternion to the Box constructor
        return Box(label=label, center=center, dims=dims, quaternion=quaternion)

    # Recursive cases for Adjacency and Symmetry remain the same
    elif node.is_adj():
        left_child_dsl = tree_to_dsl(node.left)
        right_child_dsl = tree_to_dsl(node.right)
        return Union(left=left_child_dsl, right=right_child_dsl)

    elif node.is_sym():
        child_dsl = tree_to_dsl(node.left)
        sym_vec = node.sym.squeeze()
        sym_type = sym_vec[0].item()

        if sym_type == 0.0: # Reflection
            plane_normal = sym_vec[1:4].tolist()
            point_on_plane = sym_vec[4:7].tolist()
            return SymRef(child=child_dsl, plane_normal=plane_normal, point_on_plane=point_on_plane)
        elif sym_type == -1.0: # Rotation
            axis = sym_vec[1:4].tolist()
            center = sym_vec[4:7].tolist()
            n_fold_param = sym_vec[7].item()
            n_fold = int(round(1.0 / n_fold_param)) if n_fold_param != 0 else 1
            return SymRot(child=child_dsl, axis=axis, center=center, n_fold=n_fold)
        elif sym_type == 1.0: # Translation
            end_point = sym_vec[4:7].tolist()
            n_fold_param = sym_vec[7].item()
            n_fold = int(round(1.0 / n_fold_param)) if n_fold_param != 0 else 1
            return SymTrans(child=child_dsl, end_point=end_point, n_fold=n_fold)

    return None


# In[51]:


# Assuming 'my_shape' is an object from your first script
# and you have already run:
# my_shape.construct_tree()

# Convert the tree to our new DSL representation
dsl_representation = tree_to_dsl(my_shape.tree.root)

# Print the readable DSL version of the tree
print(dsl_representation)


# In[52]:


def dsl_to_dict(node):
    """
    Recursively converts a DSL object tree into a JSON-serializable dictionary.
    This version expects Box objects to have a .quaternion attribute.
    """
    if isinstance(node, Box):
        return {
            "type": "Box",
            "label": node.label,
            "center": node.center,
            "lengths": node.lengths,
            "quaternion": node.quaternion  # Changed from dir_vectors
        }
    # ... the rest of the function for Union, SymRef, etc. is unchanged ...
    elif isinstance(node, Union):
        return {"type": "Union", "left": dsl_to_dict(node.left), "right": dsl_to_dict(node.right)}
    elif isinstance(node, SymRef):
        return {"type": "SymRef", "plane_normal": node.plane, "point_on_plane": node.point_on_plane, "child": dsl_to_dict(node.child)}
    elif isinstance(node, SymRot):
        return {"type": "SymRot", "axis": node.axis, "center": node.center, "n_fold": node.n, "child": dsl_to_dict(node.child)}
    elif isinstance(node, SymTrans):
        return {"type": "SymTrans", "end_point": node.end_point, "n_fold": node.n, "child": dsl_to_dict(node.child)}
    raise TypeError(f"Object of type {type(node).__name__} is not JSON serializable")


# In[53]:


import json


# 1. Re-create the DSL representation using the updated function
# This ensures Box objects are created with .quaternion
dsl_representation = tree_to_dsl(my_shape.tree.root)

# 2. Convert the new DSL object tree into a dictionary
dsl_dictionary = dsl_to_dict(dsl_representation)

# 3. Serialize the dictionary into a JSON string
json_output = json.dumps(dsl_dictionary, indent=2)

# 4. Print the final result, which will now have quaternions
print(json_output)


# In[54]:


import json

# Note: This parser assumes that your DSL classes (Box, Union, SymRef, etc.)
# are already defined in your notebook's memory.

def dict_to_dsl(node_dict):
    """
    Recursively traverses a dictionary and converts it into the corresponding
    DSL object tree (Box, Union, SymRef, etc.).
    """
    if not isinstance(node_dict, dict):
        raise TypeError("Input must be a dictionary.")

    node_type = node_dict.get("type")

    # --- Base Case: Reconstruct a Box object ---
    if node_type == "Box":
        return Box(
            label=node_dict["label"],
            center=node_dict["center"],
            dims=node_dict["lengths"],
            quaternion=node_dict["quaternion"]
        )

    # --- Recursive Cases for Internal Nodes ---
    elif node_type == "Union":
        # Recursively build the left and right children first
        left_child = dict_to_dsl(node_dict["left"])
        right_child = dict_to_dsl(node_dict["right"])
        return Union(left=left_child, right=right_child)

    elif node_type == "SymRef":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymRef(
            child=child_obj,
            plane_normal=node_dict["plane_normal"],
            point_on_plane=node_dict["point_on_plane"]
        )

    elif node_type == "SymRot":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymRot(
            child=child_obj,
            axis=node_dict["axis"],
            center=node_dict["center"],
            n_fold=node_dict["n_fold"]
        )

    elif node_type == "SymTrans":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymTrans(
            child=child_obj,
            end_point=node_dict["end_point"],
            n_fold=node_dict["n_fold"]
        )

    # --- Error Case for unknown types ---
    else:
        raise ValueError(f"Unknown node type found in JSON: '{node_type}'")


def parse_json_to_dsl(json_string):
    """
    A convenient wrapper function that parses a JSON string into a DSL object tree.
    """
    # 1. Convert the JSON string into a Python dictionary
    data_dictionary = json.loads(json_string)

    # 2. Convert the dictionary into DSL objects
    return dict_to_dsl(data_dictionary)


# In[55]:


# 'json_output' is the JSON string created in the previous step.

print("--- Parsing JSON back into Python objects... ---")

# Use the parser to reconstruct the entire object tree
reconstructed_dsl_root = parse_json_to_dsl(json_output)

print("--- Parse Complete. Verifying the result. ---")
print(f"The reconstructed object is of type: {type(reconstructed_dsl_root)}\n")

# To visually confirm it worked, print the reconstructed object.
# This should produce the nicely formatted text output.
print("--- Formatted String Output of the Reconstructed Object: ---")
print(reconstructed_dsl_root)


# In[56]:


import k3d
import numpy as np
from scipy.spatial.transform import Rotation

# Color map for part labels
LABEL_COLORS = {
    0: 0xFF0000,  # red   (Backrest)
    1: 0x00FF00,  # green (Seat)
    2: 0x0000FF,  # blue  (Leg)
    3: 0xFFFF00,  # yellow (Armrest)
    -1: 0x808080 # gray (Unknown)
}

def expand_dsl_tree(node):
    """
    Recursively traverses the DSL object tree and expands all symmetries to generate
    a final, flat list of box dictionaries.
    """
    # Base Case: A primitive Box
    if isinstance(node, Box):
        return [{
            'center': np.array(node.center),
            'lengths': np.array(node.lengths),
            'quaternion': np.array(node.quaternion),
            'label_id': node.label
        }]

    # Recursive Case: A Union of two parts
    elif isinstance(node, Union):
        left_boxes = expand_dsl_tree(node.left)
        right_boxes = expand_dsl_tree(node.right)
        return left_boxes + right_boxes

    # Recursive Case: A Symmetry operation
    elif isinstance(node, (SymRef, SymRot, SymTrans)):
        child_boxes = expand_dsl_tree(node.child)
        generated_boxes = []

        if isinstance(node, SymRef):
            plane_normal = np.array(node.plane) / (np.linalg.norm(node.plane) + 1e-8)
            point_on_plane = np.array(node.point_on_plane)
            for box in child_boxes:
                reflected_box = box.copy()
                vec_to_plane = box['center'] - point_on_plane
                dist = np.dot(vec_to_plane, plane_normal)
                reflected_box['center'] = box['center'] - 2 * dist * plane_normal
                R_orig = Rotation.from_quat(box['quaternion']).as_matrix()
                M_reflect = np.identity(3) - 2 * np.outer(plane_normal, plane_normal)
                R_new = M_reflect @ R_orig
                if np.linalg.det(R_new) < 0: R_new[:, 0] *= -1
                reflected_box['quaternion'] = Rotation.from_matrix(R_new).as_quat()
                generated_boxes.append(reflected_box)

        elif isinstance(node, SymRot):
            axis = np.array(node.axis) / (np.linalg.norm(node.axis) + 1e-8)
            center = np.array(node.center)
            for i in range(1, node.n):
                angle = 2 * np.pi * i / node.n
                symmetry_rot = Rotation.from_rotvec(angle * axis)
                for box in child_boxes:
                    rotated_box = box.copy()
                    vec_from_center = box['center'] - center
                    rotated_box['center'] = center + symmetry_rot.apply(vec_from_center)
                    original_rot = Rotation.from_quat(box['quaternion'])
                    rotated_box['quaternion'] = (symmetry_rot * original_rot).as_quat()
                    generated_boxes.append(rotated_box)

        elif isinstance(node, SymTrans):
            if child_boxes:
                start_point = child_boxes[0]['center']
                step_vector = (np.array(node.end_point) - start_point) / (node.n - 1)
                for i in range(1, node.n):
                    translation = i * step_vector
                    for box in child_boxes:
                        translated_box = box.copy()
                        translated_box['center'] = box['center'] + translation
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
        center, lengths, quaternion, label_id = box["center"], box["lengths"], box["quaternion"], box.get("label_id", -1)

        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        d1, d2, d3 = [col * length / 2 for col, length in zip(rotation_matrix.T, lengths)]

        corners = np.array([
            center-d1-d2-d3, center-d1+d2-d3, center+d1-d2-d3, center+d1+d2-d3,
            center-d1-d2+d3, center-d1+d2+d3, center+d1-d2+d3, center+d1+d2+d3
        ], dtype=np.float32)

        faces = np.array([
            [0,1,3], [0,3,2], [4,6,7], [4,7,5], [0,2,6], [0,6,4],
            [1,5,7], [1,7,3], [0,4,5], [0,5,1], [2,3,7], [2,7,6]
        ], dtype=np.uint32)

        color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
        plot += k3d.mesh(corners, faces, color=color)

    plot.display()


# In[57]:


plot_dsl_with_k3d(reconstructed_dsl_root)


# In[60]:


import os
import pandas as pd
import json

# --- Configuration ---
CSV_FILE_PATH = "dataset_overview.csv"
OUTPUT_DIRECTORY = "dataset"

# --- Create the output directory if it doesn't already exist ---
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
print(f"Output directory '{OUTPUT_DIRECTORY}' is ready.")

# --- Load the dataset overview, using the 'shape' column as the index ---
print(f"Loading shape data from '{CSV_FILE_PATH}'...")
df = pd.read_csv(CSV_FILE_PATH, index_col="shape")
total_shapes = len(df)
print(f"Found {total_shapes} shapes to process.")

# --- Main Processing Loop ---
# The .iterrows() method gives us the index (which we call 'shapename') and the row data
for i, (shapename, row) in enumerate(df.iterrows()):
    print(f"\n[{i+1}/{total_shapes}] Processing shape: {shapename}")

    try:
        # 1. Load the raw shape data
        shape_obj = ShapeData(row)

        # 2. Construct the initial binary tree
        shape_obj.construct_tree()

        # 3. Convert the tree to our custom DSL object representation
        dsl_representation = tree_to_dsl(shape_obj.tree.root)

        # 4. Convert the DSL object tree to a JSON-serializable dictionary
        dsl_dictionary = dsl_to_dict(dsl_representation)

        # 5. Define the output path using the 'shapename' from the first column
        #    This line creates filenames like "Bag_1.json", "Bag_10.json", etc.
        json_filepath = os.path.join(OUTPUT_DIRECTORY, f"{shapename}.json")

        # 6. Write the dictionary to the JSON file
        with open(json_filepath, 'w') as f:
            json.dump(dsl_dictionary, f, indent=2)

        print(f"✅ Successfully saved to {json_filepath}")

    except Exception as e:
        # If anything goes wrong, print an error and continue to the next shape
        print(f"❌ ERROR: Failed to process shape {shapename}. Reason: {e}")

print("\n--- All shapes processed. ---")


# In[ ]:




