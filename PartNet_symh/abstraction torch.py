#!/usr/bin/env python
# coding: utf-8

# # Imports and Setup
# 
# First, we import the necessary libraries. We'll use torch for all numerical computations, k3d for 3D visualization, and ipywidgets for the interactive editor.

# In[1]:


import json
import os
import k3d
import torch
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from IPython.display import display, clear_output


# # Configuration
# 
# We define a dictionary to map semantic labels (represented by integers) to specific colors for visualization. This helps in visually distinguishing different parts of a reconstructed shape. A default color is set for any unknown labels.

# In[2]:


# A mapping from integer labels to hex color codes for visualization
LABEL_COLORS = {
    0: 0xFF0000,  # red
    1: 0x00FF00,  # green
    2: 0x0000FF,  # blue
    3: 0xFFFF00,  # yellow
    -1: 0x808080 # gray (Unknown)
}


# # PyTorch-based Rotation Utilities
# 
# The original code used scipy.spatial.transform.Rotation, which is based on NumPy. To make this project fully compatible with PyTorch, we need to replicate the necessary rotation functionalities (quaternion-to-matrix, matrix-to-quaternion, etc.) using torch operations. These utility functions will be crucial for calculating the orientation of the boxes.

# In[3]:


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion from (x, y, z, w) format to a 3x3 rotation matrix."""
    x, y, z, w = quat
    # Pre-calculate reused terms
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Create the rotation matrix
    return torch.tensor([
        [1 - 2 * (y2 + z2), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (x2 + z2), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (x2 + y2)]
    ], dtype=quat.dtype, device=quat.device)

def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Converts a 3x3 rotation matrix to a quaternion in (x, y, z, w) format."""
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    return torch.tensor([x, y, z, w], dtype=matrix.dtype, device=matrix.device)


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Converts a rotation vector to a 3x3 rotation matrix using Rodrigues' formula."""
    angle = torch.linalg.norm(rotvec)
    if angle == 0:
        return torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)

    axis = rotvec / angle
    k_x, k_y, k_z = axis

    K = torch.tensor([
        [0,   -k_z, k_y],
        [k_z, 0,    -k_x],
        [-k_y, k_x, 0]
    ], dtype=rotvec.dtype, device=rotvec.device)

    I = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    # Rodrigues' rotation formula
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


# # DSL Class Definitions
# 
# Here we define the classes that represent the nodes in our DSL tree. Each class corresponds to a specific shape primitive or operation. All numerical attributes (like center, dimensions, etc.) are stored as PyTorch tensors.
# 
# - Box: The basic building block, representing a cuboid in 3D space.
# - Union: A node that combines two sub-trees into a single object.
# - Symmetry: A base class for symmetry operations.
# - SymRef, SymRot, SymTrans: Specific symmetry types for reflection, rotation, and translation, respectively.

# In[4]:


class Box:
    def __init__(self, label: int, center, dims, quaternion):
        self.label = label
        self.center = torch.tensor(center, dtype=torch.float32)
        self.lengths = torch.tensor(dims, dtype=torch.float32)
        self.quaternion = torch.tensor(quaternion, dtype=torch.float32)

    def __str__(self):
        center_str = torch.round(self.center, decimals=2)
        lengths_str = torch.round(self.lengths, decimals=2)
        quat_str = torch.round(self.quaternion, decimals=2)
        return f"Box(label_id={self.label}, center={center_str.tolist()}, dims={lengths_str.tolist()}, quat={quat_str.tolist()})"

class Union:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return "Union"

class Symmetry:
    def __init__(self, child):
        self.child = child

class SymRef(Symmetry):
    def __init__(self, child, plane_normal, point_on_plane):
        super().__init__(child)
        self.plane = torch.tensor(plane_normal, dtype=torch.float32)
        self.point_on_plane = torch.tensor(point_on_plane, dtype=torch.float32)

    def __str__(self):
        plane_vec = torch.round(self.plane, decimals=2)
        point_vec = torch.round(self.point_on_plane, decimals=2)
        return f"Symmetry(Reflection) across plane with normal={plane_vec.tolist()} at point={point_vec.tolist()}"

class SymRot(Symmetry):
    def __init__(self, child, axis, center, n_fold: int):
        super().__init__(child)
        self.axis = torch.tensor(axis, dtype=torch.float32)
        self.center = torch.tensor(center, dtype=torch.float32)
        self.n = n_fold

    def __str__(self):
        axis_vec = torch.round(self.axis, decimals=2)
        center_vec = torch.round(self.center, decimals=2)
        return f"Symmetry(Rotation) of {self.n}-fold around axis={axis_vec.tolist()} at center={center_vec.tolist()}"

class SymTrans(Symmetry):
    def __init__(self, child, end_point, n_fold: int):
        super().__init__(child)
        self.end_point = torch.tensor(end_point, dtype=torch.float32)
        self.n = n_fold

    def __str__(self):
        end_point_vec = torch.round(self.end_point, decimals=2)
        return f"Symmetry(Translation) of {self.n} items towards end_point={end_point_vec.tolist()}"


# # JSON to DSL Parser
# 
# These functions handle the conversion from a JSON string to our DSL object tree. The dict_to_dsl function recursively traverses the dictionary loaded from JSON and instantiates the appropriate class for each node.

# In[5]:


def dict_to_dsl(node_dict):
    """
    Recursively traverses a dictionary and converts it into the corresponding
    DSL object tree (Box, Union, SymRef, etc.).
    """
    if not isinstance(node_dict, dict):
        raise TypeError("Input must be a dictionary.")

    node_type = node_dict.get("type")

    if node_type == "Box":
        return Box(
            label=node_dict["label"],
            center=node_dict["center"],
            dims=node_dict["lengths"],
            quaternion=node_dict["quaternion"]
        )
    elif node_type == "Union":
        left_child = dict_to_dsl(node_dict["left"])
        right_child = dict_to_dsl(node_dict["right"])
        return Union(left=left_child, right=right_child)
    elif node_type == "SymRef":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymRef(child=child_obj, plane_normal=node_dict["plane_normal"], point_on_plane=node_dict["point_on_plane"])
    elif node_type == "SymRot":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymRot(child=child_obj, axis=node_dict["axis"], center=node_dict["center"], n_fold=node_dict["n_fold"])
    elif node_type == "SymTrans":
        child_obj = dict_to_dsl(node_dict["child"])
        return SymTrans(child=child_obj, end_point=node_dict["end_point"], n_fold=node_dict["n_fold"])
    else:
        raise ValueError(f"Unknown node type found in JSON: '{node_type}'")

def parse_json_to_dsl(json_string):
    """Wrapper function to parse a JSON string into a DSL object tree."""
    data_dictionary = json.loads(json_string)
    return dict_to_dsl(data_dictionary)


# # DSL Tree Printing Utility
# 
# To better understand the structure of a parsed shape, these helper functions print the DSL tree in a readable, indented format.

# In[6]:


def _print_dsl_recursive(node, prefix="", is_last=True):
    """Helper function to recursively print the tree."""
    # Define the connectors for the tree structure
    connector = "└── " if is_last else "├── "
    print(prefix + connector + str(node))

    # Adjust the prefix for child nodes
    child_prefix = prefix + ("    " if is_last else "│   ")

    # Recurse into the children of the node
    if isinstance(node, Union):
        # A Union has two children, handle them with correct connectors
        _print_dsl_recursive(node.left, child_prefix, is_last=False)
        _print_dsl_recursive(node.right, child_prefix, is_last=True)
    elif isinstance(node, Symmetry):
        # Symmetries have only one child
        _print_dsl_recursive(node.child, child_prefix, is_last=True)

def print_dsl(root_node):
    """Prints a readable, tree-like structure of the DSL object hierarchy."""
    print("\n" + "="*25)
    print("  DSL Object Structure")
    print("="*25)
    _print_dsl_recursive(root_node)
    print("="*25 + "\n")


# # DSL Tree Expansion
# 
# This is the core logic of the program. The expand_dsl_tree function recursively walks the DSL tree and "unrolls" all the symmetry operations. For example, a SymRot node with a 4-fold rotation will be expanded into four copies of its child, each rotated by the appropriate angle. The final output is a flat list of dictionaries, where each dictionary represents a single Box with its final position, size, and orientation.
# 
# All calculations are performed with PyTorch tensors.

# In[7]:


def expand_dsl_tree(node):
    """
    Recursively traverses the DSL object tree and expands all symmetries to generate
    a final, flat list of box dictionaries with torch tensors.
    """
    if isinstance(node, Box):
        return [{'center': node.center, 'lengths': node.lengths, 'quaternion': node.quaternion, 'label_id': node.label}]

    elif isinstance(node, Union):
        return expand_dsl_tree(node.left) + expand_dsl_tree(node.right)

    elif isinstance(node, (SymRef, SymRot, SymTrans)):
        child_boxes, generated_boxes = expand_dsl_tree(node.child), []

        if isinstance(node, SymRef):
            plane_normal = node.plane / (torch.linalg.norm(node.plane) + 1e-8)
            point_on_plane = node.point_on_plane
            for box in child_boxes:
                reflected_box = box.copy()
                vec_to_plane = box['center'] - point_on_plane
                dist = torch.dot(vec_to_plane, plane_normal)
                reflected_box['center'] = box['center'] - 2 * dist * plane_normal

                # Reflect the rotation
                R_orig = quat_to_matrix(box['quaternion'])
                M_reflect = torch.eye(3) - 2 * torch.outer(plane_normal, plane_normal)
                R_new = M_reflect @ R_orig

                # Ensure it's a valid rotation matrix (no reflection)
                if torch.linalg.det(R_new) < 0:
                    R_new[:, 0] *= -1

                reflected_box['quaternion'] = matrix_to_quat(R_new)
                generated_boxes.append(reflected_box)

        elif isinstance(node, SymRot):
            axis = node.axis / (torch.linalg.norm(node.axis) + 1e-8)
            center = node.center
            for i in range(1, node.n):
                angle = 2 * torch.pi * i / node.n
                symmetry_rot_mat = rotvec_to_matrix(angle * axis)

                for box in child_boxes:
                    rotated_box = box.copy()
                    # Apply rotation to the center
                    rotated_box['center'] = center + (symmetry_rot_mat @ (box['center'] - center))

                    # Compose rotations
                    original_rot_mat = quat_to_matrix(box['quaternion'])
                    new_rot_mat = symmetry_rot_mat @ original_rot_mat
                    rotated_box['quaternion'] = matrix_to_quat(new_rot_mat)
                    generated_boxes.append(rotated_box)

        elif isinstance(node, SymTrans):
            if child_boxes:
                start_point = child_boxes[0]['center']
                if node.n > 1:
                    step_vector = (node.end_point - start_point) / (node.n - 1)
                    for i in range(1, node.n):
                        translation = i * step_vector
                        for box in child_boxes:
                            translated_box = box.copy()
                            translated_box['center'] = box['center'] + translation
                            generated_boxes.append(translated_box)

        return child_boxes + generated_boxes

    return []


# # Visualization with k3d
# 
# This function takes the root of a DSL tree, expands it into a list of boxes, and then uses k3d to render each box as a mesh in a 3D plot. Note that while all calculations use torch, the k3d.mesh function expects NumPy arrays as input, so we call .cpu().numpy() at the very last moment before plotting.

# In[8]:


def plot_dsl_with_k3d(dsl_root_node):
    """Top-level function to expand a DSL tree and plot it using k3d."""
    print("Expanding DSL tree for visualization...")
    final_boxes = expand_dsl_tree(dsl_root_node)
    print(f"Found {len(final_boxes)} total boxes after expansion.")

    plot = k3d.plot(name="Reconstructed DSL Shape Visualization")

    for box in final_boxes:
        center, lengths, quaternion, label_id = box["center"], box["lengths"], box["quaternion"], box.get("label_id", -1)
        rotation_matrix = quat_to_matrix(quaternion)

        # Calculate the 8 corners of the oriented box
        d_vectors = [col * length / 2 for col, length in zip(rotation_matrix.T, lengths)]
        d1, d2, d3 = d_vectors[0], d_vectors[1], d_vectors[2]

        # THE CORRECTED LINE: The definition of the 8 unique corners is now correct.
        corners = torch.stack([
            center-d1-d2-d3, center-d1+d2-d3, center+d1-d2-d3, center+d1+d2-d3,
            center-d1-d2+d3, center-d1+d2+d3, center+d1-d2+d3, center+d1+d2+d3
        ])

        # Define the 12 triangular faces of the box
        faces = torch.tensor([
            [0,1,3], [0,3,2], [4,6,7], [4,7,5], [0,2,6], [0,6,4],
            [1,5,7], [1,7,3], [0,4,5], [0,5,1], [2,3,7], [2,7,6]
        ], dtype=torch.uint32)

        color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
        # Convert to numpy ONLY for plotting
        plot += k3d.mesh(corners.cpu().numpy(), faces.cpu().numpy(), color=color)

    plot.display()


# # Example: Loading an External JSON File
# 
# This cell shows how to load a shape definition from a JSON file.

# In[9]:


import random

random.seed(10)

# Define the path to the directory containing your JSON files
dataset_dir = "dataset/Chair"

# List all files in the directory and filter for those ending with .json
json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]

# Pick a random filename from the list and create the full path
random_filename = random.choice(json_files)
file_to_load = os.path.join(dataset_dir, random_filename)

print(f"Randomly selected: {file_to_load}")

# 1. Read the content of the randomly chosen JSON file
with open(file_to_load, 'r') as f:
    json_string = f.read()

# 2. Parse the JSON string into our live DSL object hierarchy
reconstructed_shape = parse_json_to_dsl(json_string)
print(f"Successfully parsed. Reconstructed object type is: {type(reconstructed_shape)}")

# 3. Print the readable structure of the DSL tree
print_dsl(reconstructed_shape)

# 4. Pass the reconstructed object to the K3D plotting function
plot_dsl_with_k3d(reconstructed_shape)


# # Interactive GUI Editor
# 
# This final section creates an interactive editor for our DSL tree using ipywidgets. The NodeEditor class dynamically generates sliders and text boxes for all parameters in the tree. When you modify a value (e.g., the center of a box or the n_fold of a symmetry), the 3D plot on the right automatically updates. This provides a powerful way to experiment with the DSL and see the effects of parameter changes in real-time.

# In[10]:


class NodeEditor:
    def __init__(self, root_node):
        self.root = root_node
        self.out = widgets.Output()
        self.tree_widget = self._make_node_widget(self.root)
        self.ui = HBox([
            VBox([widgets.Label("DSL Tree Editor"), self.tree_widget], layout=Layout(width="40%", border='1px solid lightgray', padding='10px', overflow='auto')),
            VBox([widgets.Label("3D Viewer"), self.out], layout=Layout(width="60%"))
        ])
        self._update_plot()

    def _make_array_editor(self, arr, label, callback):
        """Creates a row of FloatText widgets for editing a vector/tensor."""
        boxes = [widgets.FloatText(value=float(v), layout=Layout(width="80px"), step=0.01) for v in arr]
        box = HBox([widgets.Label(label, layout=Layout(width="80px"))] + boxes)

        def on_change(change):
            new_vals = [b.value for b in boxes]
            callback(new_vals) # The callback will update the torch tensor
            self._update_plot()

        for b in boxes:
            b.observe(on_change, names="value")
        return box

    def _make_node_widget(self, node):
        node_editors = []
        children_widgets = []
        title = "Node"

        # Dynamically create editors based on the node type
        if isinstance(node, Box):
            title = f"Box (Label: {node.label})"
            node_editors.append(self._make_array_editor(node.center, "center", lambda v: setattr(node, 'center', torch.tensor(v, dtype=torch.float32))))
            node_editors.append(self._make_array_editor(node.lengths, "dims", lambda v: setattr(node, 'lengths', torch.tensor(v, dtype=torch.float32))))
            node_editors.append(self._make_array_editor(node.quaternion, "quat", lambda v: setattr(node, 'quaternion', torch.tensor(v, dtype=torch.float32))))

        elif isinstance(node, Union):
            title = "Union"
            children_widgets.append(self._make_node_widget(node.left))
            children_widgets.append(self._make_node_widget(node.right))

        elif isinstance(node, SymRef):
            title = "Symmetry (Reflection)"
            node_editors.append(self._make_array_editor(node.plane, "plane", lambda v: setattr(node, 'plane', torch.tensor(v, dtype=torch.float32))))
            node_editors.append(self._make_array_editor(node.point_on_plane, "point", lambda v: setattr(node, 'point_on_plane', torch.tensor(v, dtype=torch.float32))))
            children_widgets.append(self._make_node_widget(node.child))

        elif isinstance(node, SymRot):
            title = f"{node.n}-Fold Rotation"
            node_editors.append(self._make_array_editor(node.axis, "axis", lambda v: setattr(node, 'axis', torch.tensor(v, dtype=torch.float32))))
            node_editors.append(self._make_array_editor(node.center, "center", lambda v: setattr(node, 'center', torch.tensor(v, dtype=torch.float32))))
            n_editor = widgets.IntText(value=node.n, description="n-fold", layout=Layout(width='auto'), style={'description_width': 'initial'})
            def on_n_change(change):
                node.n = change["new"]
                self._update_plot()
            n_editor.observe(on_n_change, names="value")
            node_editors.append(n_editor)
            children_widgets.append(self._make_node_widget(node.child))

        # Combine parameter editors and child widgets
        content = VBox(node_editors + children_widgets)
        accordion = widgets.Accordion(children=[content])
        accordion.set_title(0, title)
        return accordion

    def _update_plot(self):
        """Redraws the 3D plot based on the current DSL tree state."""
        with self.out:
            clear_output(wait=True)
            plot = k3d.plot(name="Interactive Shape Viewer")
            final_boxes = expand_dsl_tree(self.root)
            for box in final_boxes:
                center, lengths, quat, label_id = box["center"], box["lengths"], box["quaternion"], box.get("label_id", -1)
                rotation_matrix = quat_to_matrix(quat)
                d_vectors = [col * length / 2 for col, length in zip(rotation_matrix.T, lengths)]
                d1, d2, d3 = d_vectors[0], d_vectors[1], d_vectors[2]
                corners = torch.stack([
                    center-d1-d2-d3, center-d1+d2-d3, center+d1-d2-d3, center+d1+d2-d3,
                    center-d1-d2+d3, center-d1+d2+d3, center+d1-d2+d3, center+d1+d2+d3
                ])
                faces = torch.tensor([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,2,6],[0,6,4],[1,5,7],[1,7,3],[0,4,5],[0,5,1],[2,3,7],[2,7,6]], dtype=torch.uint32)
                color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
                plot += k3d.mesh(corners.cpu().numpy(), faces.cpu().numpy(), color=color)
            display(plot)

    def show(self):
        """Displays the UI."""
        display(self.ui)

# --- Main execution for the interactive editor ---
# We can reuse the shape we parsed earlier
editor = NodeEditor(reconstructed_shape)
editor.show()


# # Abstraction

# In[11]:


import json
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Configuration & Setup ---

# Set to True to enable detailed debug prints throughout the forward passes.
ENABLE_DEBUG = False

# A mapping from integer labels to hex color codes for visualization
LABEL_COLORS = {
    0: 0xFF0000, # red
    1: 0x00FF00, # green
    2: 0x0000FF, # blue
    3: 0xFFFF00, # yellow
    -1: 0x808080 # gray (Unknown)
}

# Use a GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Vocabulary for the Model
NODE_TYPE_TO_IDX = {
    "Box": 0, "Union": 1, "SymRef": 2, "SymRot": 3, "SymTrans": 4, "<EOS>": 5,
}
IDX_TO_NODE_TYPE = {v: k for k, v in NODE_TYPE_TO_IDX.items()}

# Number of parameters for each node type
PARAM_SIZE_MAP = {
    "Box": 11, "Union": 0, "SymRef": 6, "SymRot": 7, "SymTrans": 4
}

# Model Hyperparameters
HIDDEN_DIM = 128
LATENT_DIM = 64


# In[12]:


# --- Vectorization and Devectorization Utilities ---

def vectorize_tree(node, device):
    """Converts a DSL tree to a nested dictionary of tensors, ready for the model."""
    node_type = node.__class__.__name__
    vectorized_node = {
        "type": torch.tensor(NODE_TYPE_TO_IDX[node_type], dtype=torch.long, device=device),
        "children": []
    }
    if isinstance(node, Box):
        params = [float(node.label)] + node.center.tolist() + \
                 node.lengths.tolist() + node.quaternion.tolist()
        vectorized_node["params"] = torch.tensor(params, dtype=torch.float32, device=device)
    elif isinstance(node, Union):
        vectorized_node["children"].append(vectorize_tree(node.left, device))
        vectorized_node["children"].append(vectorize_tree(node.right, device))
    elif isinstance(node, Symmetry):
        params = []
        if isinstance(node, SymRef):
            params = node.plane.tolist() + node.point_on_plane.tolist()
        elif isinstance(node, SymRot):
            params = node.axis.tolist() + node.center.tolist() + [float(node.n)]
        elif isinstance(node, SymTrans):
            params = node.end_point.tolist() + [float(node.n)]
        vectorized_node["params"] = torch.tensor(params, dtype=torch.float32, device=device)
        vectorized_node["children"].append(vectorize_tree(node.child, device))

    return vectorized_node

def devectorize_tree(recon_node):
    """Converts a decoder's output dictionary back into a symbolic DSL object tree."""
    if recon_node is None: return None
    predicted_type_idx = torch.argmax(recon_node['type_logits']).item()
    predicted_type_str = IDX_TO_NODE_TYPE[predicted_type_idx]
    if predicted_type_str == "Box":
        p = recon_node['params'].cpu()
        return Box(label=int(p[0]), center=p[1:4], dims=p[4:7], quaternion=p[7:11])
    elif predicted_type_str == "Union":
        if len(recon_node.get('children', [])) < 2: return None
        left_child = devectorize_tree(recon_node['children'][0])
        right_child = devectorize_tree(recon_node['children'][1])
        if left_child is None or right_child is None: return None
        return Union(left=left_child, right=right_child)
    elif predicted_type_str in ["SymRef", "SymRot", "SymTrans"]:
        if len(recon_node.get('children', [])) < 1: return None
        child = devectorize_tree(recon_node['children'][0])
        if child is None: return None
        p = recon_node['params'].cpu()
        if predicted_type_str == "SymRef":
            return SymRef(child=child, plane_normal=p[0:3], point_on_plane=p[3:6])
        elif predicted_type_str == "SymRot":
            return SymRot(child=child, axis=p[0:3], center=p[3:6], n_fold=int(p[6]))
        elif predicted_type_str == "SymTrans":
            return SymTrans(child=child, end_point=p[0:3], n_fold=int(p[3]))
    return None

def devectorize_tree_debug(recon_node, depth=0):
    """A debug version of devectorize_tree that prints step-by-step logic."""
    indent = "  " * depth
    if recon_node is None:
        print(f"{indent}[DEBUG] Input node is None. Branch failed.")
        return None
    predicted_type_idx = torch.argmax(recon_node['type_logits']).item()
    predicted_type_str = IDX_TO_NODE_TYPE[predicted_type_idx]
    print(f"{indent}[DEBUG] Model PREDICTED node type: {predicted_type_str}")
    if predicted_type_str == "Box":
        print(f"{indent}[DEBUG] SUCCESS: Reached a valid leaf node (Box).")
        p = recon_node['params'].cpu()
        return Box(label=int(p[0]), center=p[1:4], dims=p[4:7], quaternion=p[7:11])
    elif predicted_type_str == "Union":
        num_children = len(recon_node.get('children', []))
        if num_children < 2:
            print(f"{indent}[DEBUG] ❌ FAIL: Predicted Union but found only {num_children} children. Expected 2.")
            return None
        print(f"{indent}[DEBUG] Recursing into 2 children for Union...")
        left_child = devectorize_tree_debug(recon_node['children'][0], depth + 1)
        right_child = devectorize_tree_debug(recon_node['children'][1], depth + 1)
        if left_child is None or right_child is None:
            print(f"{indent}[DEBUG] ❌ FAIL: A child of the Union failed to reconstruct.")
            return None
        print(f"{indent}[DEBUG] SUCCESS: Reconstructed Union with 2 children.")
        return Union(left=left_child, right=right_child)
    elif predicted_type_str in ["SymRef", "SymRot", "SymTrans"]:
        num_children = len(recon_node.get('children', []))
        if num_children < 1:
            print(f"{indent}[DEBUG] ❌ FAIL: Predicted Symmetry but found {num_children} children. Expected 1.")
            return None
        print(f"{indent}[DEBUG] Recursing into 1 child for Symmetry...")
        child = devectorize_tree_debug(recon_node['children'][0], depth + 1)
        if child is None:
            print(f"{indent}[DEBUG] ❌ FAIL: The child of the Symmetry node failed to reconstruct.")
            return None
        p = recon_node['params'].cpu()
        print(f"{indent}[DEBUG] SUCCESS: Reconstructed {predicted_type_str} with 1 child.")
        if predicted_type_str == "SymRef":
            return SymRef(child=child, plane_normal=p[0:3], point_on_plane=p[3:6])
        elif predicted_type_str == "SymRot":
            return SymRot(child=child, axis=p[0:3], center=p[3:6], n_fold=int(p[6]))
        elif predicted_type_str == "SymTrans":
            return SymTrans(child=child, end_point=p[0:3], n_fold=int(p[3]))
    else:
        print(f"{indent}[DEBUG] ❌ FAIL: Predicted an invalid or unknown node type '{predicted_type_str}'.")
        return None


# In[13]:


# --- Model Architecture ---

class BoxEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PARAM_SIZE_MAP["Box"], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, params):
        if ENABLE_DEBUG: print(f"  [BoxEncoder] Input shape: {params.shape}")
        return self.net(params)

class CompositionEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.nets = nn.ModuleDict({
            "Union": nn.Linear(2 * hidden_dim, hidden_dim),
            "SymRef": nn.Linear(hidden_dim + PARAM_SIZE_MAP["SymRef"], hidden_dim),
            "SymRot": nn.Linear(hidden_dim + PARAM_SIZE_MAP["SymRot"], hidden_dim),
            "SymTrans": nn.Linear(hidden_dim + PARAM_SIZE_MAP["SymTrans"], hidden_dim),
        })

    def forward(self, node_type_str, params, child_features):
        if ENABLE_DEBUG: print(f"  [CompositionEncoder] Node Type: {node_type_str}")
        combined_child_features = torch.cat(child_features, dim=-1)
        if ENABLE_DEBUG: print(f"  [CompositionEncoder] Combined child features shape: {combined_child_features.shape}")
        if params is not None:
            if ENABLE_DEBUG: print(f"  [CompositionEncoder] Param shape: {params.shape}")
            model_input = torch.cat([combined_child_features, params], dim=-1)
        else:
            model_input = combined_child_features
        if ENABLE_DEBUG: print(f"  [CompositionEncoder] Final input shape for '{node_type_str}' net: {model_input.shape}")
        return self.nets[node_type_str](model_input)

class TreeEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.box_encoder = BoxEncoder(hidden_dim)
        self.composition_encoder = CompositionEncoder(hidden_dim)
        self.root_to_latent = nn.Linear(hidden_dim, latent_dim)

    def _forward_recursive(self, v_node):
        node_type_str = IDX_TO_NODE_TYPE[v_node["type"].item()]
        if ENABLE_DEBUG: print(f" [Encoder] Processing node: {node_type_str}")
        if node_type_str == "Box":
            return self.box_encoder(v_node["params"])
        child_features = [self._forward_recursive(c) for c in v_node["children"]]
        params = v_node.get("params")
        return self.composition_encoder(node_type_str, params, child_features)

    def forward(self, v_tree):
        if ENABLE_DEBUG: print("--- Starting ENCODER forward pass ---")
        root_hidden_vec = self._forward_recursive(v_tree)
        if ENABLE_DEBUG: print(f" [Encoder] Root hidden vector shape: {root_hidden_vec.shape}")
        z = self.root_to_latent(root_hidden_vec)
        if ENABLE_DEBUG: print("--- Finished ENCODER forward pass ---")
        return z

class NodeDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.type_head = nn.Linear(hidden_dim, len(NODE_TYPE_TO_IDX))
        self.param_head = nn.Linear(hidden_dim, max(PARAM_SIZE_MAP.values()))
        self.child_head = nn.Linear(hidden_dim, 2 * hidden_dim)
    def forward(self, parent_feature):
        if ENABLE_DEBUG: print(f"  [NodeDecoder] Input parent feature shape: {parent_feature.shape}")
        h = self.net(parent_feature)
        return self.type_head(h), self.param_head(h), self.child_head(h)

class TreeDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.node_decoder = NodeDecoder(hidden_dim)
    def _forward_recursive(self, parent_feature, gt_node):
        type_logits, param_preds, child_features = self.node_decoder(parent_feature)
        reconstructed_node = {
            "type_logits": type_logits,
            "params": param_preds,
            "children": []
        }
        if gt_node["children"]:
            child1_feature, child2_feature = torch.chunk(child_features, 2, dim=-1)
            child_features_split = [child1_feature, child2_feature]
            for i, gt_child_node in enumerate(gt_node["children"]):
                reconstructed_child = self._forward_recursive(child_features_split[i], gt_child_node)
                reconstructed_node["children"].append(reconstructed_child)
        return reconstructed_node
    def forward(self, z, gt_tree):
        initial_hidden = self.latent_to_hidden(z)
        return self._forward_recursive(initial_hidden, gt_tree)

class RecursiveAutoencoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = TreeEncoder(hidden_dim, latent_dim)
        self.decoder = TreeDecoder(latent_dim, hidden_dim)
    def forward(self, v_tree):
        z = self.encoder(v_tree)
        reconstructed_tree = self.decoder(z, v_tree)
        return reconstructed_tree


# In[14]:


# --- Loss Function ---

def calculate_tree_loss(recon_node, gt_node):
    """Recursively calculates the reconstruction loss for a tree."""
    loss_fn_type = nn.CrossEntropyLoss()
    loss = loss_fn_type(recon_node['type_logits'].unsqueeze(0), gt_node['type'].unsqueeze(0))
    gt_node_type_str = IDX_TO_NODE_TYPE[gt_node["type"].item()]
    num_params = PARAM_SIZE_MAP.get(gt_node_type_str, 0)
    if num_params > 0:
        loss_fn_params = nn.MSELoss()
        predicted_params = recon_node['params'][:num_params]
        ground_truth_params = gt_node['params']
        loss += loss_fn_params(predicted_params, ground_truth_params)
    for i, recon_child in enumerate(recon_node['children']):
        gt_child = gt_node['children'][i]
        loss += calculate_tree_loss(recon_child, gt_child)
    return loss

# --- Training Loop and Plotting ---

def train_model(model, dataset, epochs=20, learning_rate=0.001):
    """Trains the Recursive Autoencoder."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_epoch_loss = 0.0
        pbar = tqdm(dataset, desc=f"Epoch [{epoch+1}/{epochs}]")
        for v_tree in pbar:
            optimizer.zero_grad()
            reconstructed_tree = model(v_tree)
            loss = calculate_tree_loss(reconstructed_tree, v_tree)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = total_epoch_loss / len(dataset)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] complete. Average Loss: {avg_loss:.4f}")
    print("Training finished.")
    return loss_history

def plot_loss_chart(losses):
    """Generates and saves a chart of the training loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Recursive Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_chart.png')
    print("Loss chart saved to 'training_loss_chart.png'")


# In[29]:


from tqdm import tqdm

# --- 1. Load and Prepare the Dataset ---
dataset_dir = "dataset/Chair"
if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory '{dataset_dir}' not found.")
    print("Please ensure you have the dataset folder correctly set up.")
else:
    all_json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
    files_to_process = all_json_files[:500]
    full_dataset = []
    print(f"Loading and vectorizing {len(files_to_process)} shapes...")
    for filename in tqdm(files_to_process, desc="Loading and Vectorizing Chairs"):
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, 'r') as f:
            json_string = f.read()
        dsl_tree = parse_json_to_dsl(json_string)
        vectorized_tree = vectorize_tree(dsl_tree, DEVICE)
        full_dataset.append((dsl_tree, vectorized_tree))
    print("Dataset prepared.")

    # --- 2. Split Data and Train the Model ---
    train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    training_vectors = [v_tree for _, v_tree in train_dataset]
    autoencoder = RecursiveAutoencoder(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    loss_history = train_model(autoencoder, training_vectors, epochs=200, learning_rate=0.001)

    # --- 3. Visualize Loss and Check Reconstruction ---
    plot_loss_chart(loss_history)
    print("\n" + "#"*50)
    print("### VISUALIZING A RECONSTRUCTION FROM THE TEST SET ###")
    print("#"*50 + "\n")
    original_dsl_shape, vectorized_shape = random.choice(test_dataset)
    autoencoder.eval()
    with torch.no_grad():
        reconstruction_output = autoencoder(vectorized_shape)
    print("--- Original Shape Tree ---")
    print_dsl(original_dsl_shape)
    print("\n--- STARTING RECONSTRUCTION WITH DEBUG TRACE ---\n")
    final_reconstructed_shape = devectorize_tree_debug(reconstruction_output)
    print("\n--- FINISHED RECONSTRUCTION ATTEMPT ---\n")
    print("\n--- Reconstructed Shape Tree (after training) ---")
    if final_reconstructed_shape:
        print_dsl(final_reconstructed_shape)
    else:
        print("Reconstruction failed (the model predicted an invalid or incomplete tree).")


# In[30]:


torch.save(autoencoder.state_dict(), 'recursive_autoencoder.pth')
print("Model saved to recursive_autoencoder.pth")


# In[63]:


import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
import os
from tqdm import tqdm
from IPython.display import display

# --- Re-importing necessary components ---
# These are the model classes that need to be defined to load the weights.
class BoxEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, params): return self.net(params)

class CompositionEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.nets = nn.ModuleDict({
            "Union": nn.Linear(2 * hidden_dim, hidden_dim),
            "SymRef": nn.Linear(hidden_dim + 6, hidden_dim),
            "SymRot": nn.Linear(hidden_dim + 7, hidden_dim),
            "SymTrans": nn.Linear(hidden_dim + 4, hidden_dim),
        })
    def forward(self, node_type_str, params, child_features):
        combined_child_features = torch.cat(child_features, dim=-1)
        if params is not None:
            model_input = torch.cat([combined_child_features, params], dim=-1)
        else:
            model_input = combined_child_features
        return self.nets[node_type_str](model_input)

class TreeEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.box_encoder = BoxEncoder(hidden_dim)
        self.composition_encoder = CompositionEncoder(hidden_dim)
        self.root_to_latent = nn.Linear(hidden_dim, latent_dim)
    def _forward_recursive(self, v_node):
        node_type_str = IDX_TO_NODE_TYPE[v_node["type"].item()]
        if node_type_str == "Box": return self.box_encoder(v_node["params"])
        child_features = [self._forward_recursive(c) for c in v_node["children"]]
        params = v_node.get("params")
        return self.composition_encoder(node_type_str, params, child_features)
    def forward(self, v_tree):
        root_hidden_vec = self._forward_recursive(v_tree)
        z = self.root_to_latent(root_hidden_vec)
        return z

class NodeDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.type_head = nn.Linear(hidden_dim, len(NODE_TYPE_TO_IDX))
        self.param_head = nn.Linear(hidden_dim, max([11,6,7,4])) # Use max of all param sizes
        self.child_head = nn.Linear(hidden_dim, 2 * hidden_dim)
    def forward(self, parent_feature):
        h = self.net(parent_feature)
        return self.type_head(h), self.param_head(h), self.child_head(h)

class TreeDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.node_decoder = NodeDecoder(hidden_dim)
    def _forward_recursive(self, parent_feature, gt_node):
        type_logits, param_preds, child_features = self.node_decoder(parent_feature)
        reconstructed_node = {"type_logits": type_logits, "params": param_preds, "children": []}
        if gt_node["children"]:
            child1_feature, child2_feature = torch.chunk(child_features, 2, dim=-1)
            child_features_split = [child1_feature, child2_feature]
            for i, gt_child_node in enumerate(gt_node["children"]):
                reconstructed_child = self._forward_recursive(child_features_split[i], gt_child_node)
                reconstructed_node["children"].append(reconstructed_child)
        return reconstructed_node
    def forward(self, z, gt_tree):
        initial_hidden = self.latent_to_hidden(z)
        return self._forward_recursive(initial_hidden, gt_tree)

class RecursiveAutoencoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = TreeEncoder(hidden_dim, latent_dim)
        self.decoder = TreeDecoder(latent_dim, hidden_dim)
    def forward(self, v_tree):
        z = self.encoder(v_tree)
        reconstructed_tree = self.decoder(z, v_tree)
        return reconstructed_tree

# --- Main execution to load the model and reconstruct a shape ---
if __name__ == "__main__":
    # --- Configuration
    HIDDEN_DIM = 128
    LATENT_DIM = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NODE_TYPE_TO_IDX = {"Box": 0, "Union": 1, "SymRef": 2, "SymRot": 3, "SymTrans": 4, "<EOS>": 5}
    IDX_TO_NODE_TYPE = {v: k for k, v in NODE_TYPE_TO_IDX.items()}
    # --- Load the saved model
    autoencoder = RecursiveAutoencoder(HIDDEN_DIM, LATENT_DIM).to(DEVICE)
    try:
        autoencoder.load_state_dict(torch.load('recursive_autoencoder.pth', map_location=DEVICE))
        print("Model loaded successfully from recursive_autoencoder.pth")
    except FileNotFoundError:
        print("Error: The model file 'recursive_autoencoder.pth' was not found. Please save the model first.")
        # Exit to prevent further errors
        exit()
    # Set the model to evaluation mode
    autoencoder.eval()
    # --- Prepare a shape for reconstruction
    # Assumes 'dataset/Chair' directory exists
    dataset_dir = "dataset/Chair"
    all_json_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.json')]
    if not all_json_files:
        print("Error: No chair files found in the dataset directory.")
        exit()
    # Pick a random file from the test set
    full_dataset = []
    for filepath in tqdm(all_json_files[:100], desc="Loading Shapes"):
        with open(filepath, 'r') as f:
            json_string = f.read()
        dsl_tree = parse_json_to_dsl(json_string)
        vectorized_tree = vectorize_tree(dsl_tree, DEVICE)
        full_dataset.append((dsl_tree, vectorized_tree))
    _, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    original_dsl_shape, vectorized_shape = random.choice(test_dataset)
    # --- Reconstruct the shape
    with torch.no_grad():
        reconstruction_output = autoencoder(vectorized_shape)
    final_reconstructed_shape = devectorize_tree_debug(reconstruction_output)
    # --- Display the results
    print("\n--- Original Shape ---")
    print_dsl(original_dsl_shape)

    print("\n--- Reconstructed Shape ---")
    if final_reconstructed_shape:
        print_dsl(final_reconstructed_shape)
        # Assuming plot_dsl_with_k3d is available from your other notebook cell
        plot_dsl_with_k3d(original_dsl_shape)
        plot_dsl_with_k3d(final_reconstructed_shape)
    else:
        print("Reconstruction failed.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




