#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import k3d
import numpy as np
from scipy.spatial.transform import Rotation
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from functools import partial


# In[2]:


LABEL_COLORS = {
    0: 0xFF0000,  # red
    1: 0x00FF00,  # green
    2: 0x0000FF,  # blue
    3: 0xFFFF00,  # yellow
    -1: 0x808080 # gray (Unknown)
}


# In[3]:


class Box:
    def __init__(self, label: int, center, dims, quaternion):
        self.label = label
        self.center = center
        self.lengths = dims
        self.quaternion = quaternion

    def __str__(self):
        center_str = np.round(np.array(self.center), 2)
        lengths_str = np.round(np.array(self.lengths), 2)
        quat_str = np.round(np.array(self.quaternion), 2)
        return f"Box(label_id={self.label}, center={center_str}, dims={lengths_str}, quat={quat_str})"

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
        self.plane = plane_normal
        self.point_on_plane = point_on_plane

    def __str__(self):
        plane_vec = np.round(np.array(self.plane), 2)
        point_vec = np.round(np.array(self.point_on_plane), 2)
        return f"Symmetry(Reflection) across plane with normal={plane_vec} at point={point_vec}"

class SymRot(Symmetry):
    def __init__(self, child, axis, center, n_fold: int):
        super().__init__(child)
        self.axis = axis
        self.center = center
        self.n = n_fold

    def __str__(self):
        axis_vec = np.round(np.array(self.axis), 2)
        center_vec = np.round(np.array(self.center), 2)
        return f"Symmetry(Rotation) of {self.n}-fold around axis={axis_vec} at center={center_vec}"

class SymTrans(Symmetry):
    def __init__(self, child, end_point, n_fold: int):
        super().__init__(child)
        self.end_point = end_point
        self.n = n_fold

    def __str__(self):
        end_point_vec = np.round(np.array(self.end_point), 2)
        return f"Symmetry(Translation) of {self.n} items towards end_point={end_point_vec}"


# In[4]:


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


# In[5]:


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


# In[6]:


def expand_dsl_tree(node):
    """
    Recursively traverses the DSL object tree and expands all symmetries to generate
    a final, flat list of box dictionaries.
    """
    if isinstance(node, Box):
        return [{'center': np.array(node.center), 'lengths': np.array(node.lengths), 'quaternion': np.array(node.quaternion), 'label_id': node.label}]
    elif isinstance(node, Union):
        return expand_dsl_tree(node.left) + expand_dsl_tree(node.right)
    elif isinstance(node, (SymRef, SymRot, SymTrans)):
        child_boxes, generated_boxes = expand_dsl_tree(node.child), []
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
                    rotated_box['center'] = center + symmetry_rot.apply(box['center'] - center)
                    original_rot = Rotation.from_quat(box['quaternion'])
                    rotated_box['quaternion'] = (symmetry_rot * original_rot).as_quat()
                    generated_boxes.append(rotated_box)
        elif isinstance(node, SymTrans):
            if child_boxes:
                # Assuming the first child box is the reference for translation start
                start_point = child_boxes[0]['center']
                # Correct calculation for step vector when n is the total count
                if node.n > 1:
                    step_vector = (np.array(node.end_point) - start_point) / (node.n - 1)
                    for i in range(1, node.n):
                        translation = i * step_vector
                        for box in child_boxes:
                            translated_box = box.copy()
                            translated_box['center'] = box['center'] + translation
                            generated_boxes.append(translated_box)
        return child_boxes + generated_boxes
    return []


# In[7]:


def plot_dsl_with_k3d(dsl_root_node):
    """Top-level function to expand a DSL tree and plot it using k3d."""
    print("Expanding DSL tree for visualization...")
    final_boxes = expand_dsl_tree(dsl_root_node)
    print(f"Found {len(final_boxes)} total boxes after expansion.")
    plot = k3d.plot(name="Reconstructed DSL Shape Visualization")
    for box in final_boxes:
        center, lengths, quaternion, label_id = box["center"], box["lengths"], box["quaternion"], box.get("label_id", -1)
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        d1, d2, d3 = [col * length / 2 for col, length in zip(rotation_matrix.T, lengths)]
        corners = np.array([center-d1-d2-d3, center-d1+d2-d3, center+d1-d2-d3, center+d1+d2-d3, center-d1-d2+d3, center-d1+d2+d3, center+d1-d2+d3, center+d1+d2+d3], dtype=np.float32)
        faces = np.array([[0,1,3], [0,3,2], [4,6,7], [4,7,5], [0,2,6], [0,6,4], [1,5,7], [1,7,3], [0,4,5], [0,5,1], [2,3,7], [2,7,6]], dtype=np.uint32)
        color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
        plot += k3d.mesh(corners, faces, color=color)
    plot.display()


# In[28]:


file_to_load = "dataset/Chair_200.json" # You can change this to another file

if os.path.exists(file_to_load):
    print(f"Loading shape from: {file_to_load}")

    # 1. Read the entire content of the JSON file
    with open(file_to_load, 'r') as f:
        json_string = f.read()

    # 2. Parse the JSON string into our live DSL object hierarchy
    reconstructed_shape = parse_json_to_dsl(json_string)
    print(f"Successfully parsed. Reconstructed object type is: {type(reconstructed_shape)}")

    # 3. NEW: Print the readable structure of the DSL tree
    print_dsl(reconstructed_shape)

    # 4. Pass the reconstructed object to the K3D plotting function
    plot_dsl_with_k3d(reconstructed_shape)

else:
    print(f"❌ ERROR: File not found at '{file_to_load}'. Please check the path.")


# In[33]:


# --- NEW: Simple Interactive GUI based on user's example ---

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
        boxes = [widgets.FloatText(value=float(v), layout=Layout(width="80px")) for v in arr]
        box = HBox([widgets.Label(label, layout=Layout(width="80px"))] + boxes)
        def on_change(change):
            new_vals = [b.value for b in boxes]
            callback(new_vals)
            self._update_plot()
        for b in boxes:
            b.observe(on_change, names="value")
        return box

    def _make_node_widget(self, node):
        node_editors = []
        children_widgets = []
        title = "Node"

        if isinstance(node, Box):
            title = f"Box (Label: {node.label})"
            node_editors.append(self._make_array_editor(node.center, "center", lambda v: setattr(node, 'center', np.array(v))))
            node_editors.append(self._make_array_editor(node.lengths, "dims", lambda v: setattr(node, 'lengths', np.array(v))))
            node_editors.append(self._make_array_editor(node.quaternion, "quat", lambda v: setattr(node, 'quaternion', np.array(v))))
        elif isinstance(node, Union):
            title = "Union"
            children_widgets.append(self._make_node_widget(node.left))
            children_widgets.append(self._make_node_widget(node.right))
        elif isinstance(node, SymRef):
            title = "Symmetry (Reflection)"
            node_editors.append(self._make_array_editor(node.plane, "plane", lambda v: setattr(node, 'plane', np.array(v))))
            node_editors.append(self._make_array_editor(node.point_on_plane, "point", lambda v: setattr(node, 'point_on_plane', np.array(v))))
            children_widgets.append(self._make_node_widget(node.child))
        elif isinstance(node, SymRot):
            title = f"{node.n}-Fold Rotation"
            node_editors.append(self._make_array_editor(node.axis, "axis", lambda v: setattr(node, 'axis', np.array(v))))
            node_editors.append(self._make_array_editor(node.center, "center", lambda v: setattr(node, 'center', np.array(v))))
            n_editor = widgets.IntText(value=node.n, description="n-fold", layout=Layout(width='auto'), style={'description_width': 'initial'})
            def on_n_change(change):
                node.n = change["new"]
                self._update_plot()
            n_editor.observe(on_n_change, names="value")
            node_editors.append(n_editor)
            children_widgets.append(self._make_node_widget(node.child))
        elif isinstance(node, SymTrans):
            title = f"{node.n}-Fold Translation"
            node_editors.append(self._make_array_editor(node.end_point, "end", lambda v: setattr(node, 'end_point', np.array(v))))
            n_editor = widgets.IntText(value=node.n, description="n-fold", layout=Layout(width='auto'), style={'description_width': 'initial'})
            def on_n_change(change):
                node.n = change["new"]
                self._update_plot()
            n_editor.observe(on_n_change, names="value")
            node_editors.append(n_editor)
            children_widgets.append(self._make_node_widget(node.child))
        else:
            return widgets.Label(f"Unknown node: {type(node)}")

        # Combine parameter editors and child widgets
        content = VBox(node_editors + children_widgets)

        # Create a collapsible accordion for the node
        accordion = widgets.Accordion(children=[content])
        accordion.set_title(0, title)
        return accordion

    def _update_plot(self):
        with self.out:
            clear_output(wait=True)
            plot = k3d.plot(name="Interactive Shape Viewer")
            final_boxes = expand_dsl_tree(self.root)
            for box in final_boxes:
                center, lengths, quat, label_id = box["center"], box["lengths"], box["quaternion"], box.get("label_id", -1)
                rotation_matrix = Rotation.from_quat(quat).as_matrix()
                d1, d2, d3 = [col * length / 2 for col, length in zip(rotation_matrix.T, lengths)]
                corners = np.array([
                    center-d1-d2-d3, center-d1+d2-d3, center+d1-d2-d3, center+d1+d2-d3,
                    center-d1-d2+d3, center-d1+d2+d3, center+d1-d2+d3, center+d1+d2+d3
                ], dtype=np.float32)
                faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,2,6],[0,6,4],[1,5,7],[1,7,3],[0,4,5],[0,5,1],[2,3,7],[2,7,6]], dtype=np.uint32)
                color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
                plot += k3d.mesh(corners, faces, color=color)
            display(plot)

    def display(self):
        display(self.ui)

# --- Main execution ---
file_to_load = "dataset/Chair_500.json"

if os.path.exists(file_to_load):
    print(f"Loading shape from: {file_to_load}")
    with open(file_to_load, 'r') as f:
        json_string = f.read()

    reconstructed_shape = parse_json_to_dsl(json_string)

    editor = NodeEditor(reconstructed_shape)
    editor.display()
else:
    print(f"❌ ERROR: File not found at '{file_to_load}'. Please check the path.")


# In[34]:


reconstructed_shape


# In[ ]:




