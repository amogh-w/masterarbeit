#!/usr/bin/env python
# coding: utf-8

# # Imports and Setup
# 
# First, we import the necessary libraries. We'll use torch for all numerical computations, k3d for 3D visualization, and ipywidgets for the interactive editor.

# In[31]:


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

# In[32]:


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

# In[33]:


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

# In[34]:


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

# In[35]:


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

# In[36]:


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

# In[37]:


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

# In[38]:


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

# In[39]:


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

# In[41]:


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


# In[ ]:




