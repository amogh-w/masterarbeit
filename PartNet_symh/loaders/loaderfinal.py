#!/usr/bin/env python
# coding: utf-8

# In[11]:


# List all subfolders in the Chair directory
import os

chair_dir = "dropbox/Chair"
subfolders = [f.name for f in os.scandir(chair_dir) if f.is_dir()]

print("Subfolders in Chair directory:")
for folder in sorted(subfolders):
    print(f"  - {folder}")


# In[12]:


# Explore the contents of the boxes subfolder
import scipy.io as sio
import numpy as np

# Get the first file from boxes subfolder
boxes_dir = os.path.join(chair_dir, "boxes")
box_files = sorted([f for f in os.listdir(boxes_dir) if f.endswith('.mat')])

if box_files:
    first_box_file = os.path.join(boxes_dir, box_files[0])
    print(f"Analyzing file: {first_box_file}")

    # Load and examine the MATLAB file
    mat_data = sio.loadmat(first_box_file)

    print("\nMATLAB file structure:")
    print("Keys in .mat file:", list(mat_data.keys()))

    for key in mat_data:
        if not key.startswith("__"):
            print(f"\n{key}:")
            print(f"  Type: {type(mat_data[key])}")
            print(f"  Shape: {mat_data[key].shape}")
            print(f"  Data type: {mat_data[key].dtype}")
            print(f"  Sample data:\n{mat_data[key][:2] if len(mat_data[key]) > 1 else mat_data[key]}")

    # Documentation
    print("\n" + "="*50)
    print("DOCUMENTATION:")
    print("This MATLAB file contains bounding box data for object parts.")
    print("The 'box' array typically contains oriented bounding boxes (OBBs)")
    print("with parameters like center, dimensions, and orientation vectors.")
    print("Shape (N, 12) suggests N boxes, each with 12 parameters.")
else:
    print("No .mat files found in boxes directory")


# In[13]:


# Helper functions for exploring the GRASS dataset
def explore_mat_file(file_path, max_print=5):
    """
    Explore the contents of a MATLAB .mat file

    Parameters:
    file_path (str): Path to the .mat file
    max_print (int): Maximum number of elements to print for large arrays

    Returns:
    dict: The loaded MATLAB data
    """
    print(f"Exploring: {file_path}")
    mat_data = sio.loadmat(file_path)

    print(f"Keys: {list(mat_data.keys())}")

    for key in mat_data:
        if not key.startswith("__"):
            data = mat_data[key]
            print(f"\n{key}:")
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")

            if data.size <= max_print:
                print(f"  Data:\n{data}")
            else:
                print(f"  First {max_print} elements:\n{data.flat[:max_print]}")

    return mat_data

def get_folder_stats(folder_path):
    """
    Get statistics about files in a folder

    Parameters:
    folder_path (str): Path to the folder

    Returns:
    dict: Statistics about the folder contents
    """
    if not os.path.exists(folder_path):
        return {"exists": False}

    files = sorted([f for f in os.listdir(folder_path) if not f.startswith('.')])
    mat_files = [f for f in files if f.endswith('.mat')]
    obj_files = [f for f in files if f.endswith('.obj')]
    txt_files = [f for f in files if f.endswith('.txt')]

    return {
        "exists": True,
        "total_files": len(files),
        "mat_files": len(mat_files),
        "obj_files": len(obj_files),
        "txt_files": len(txt_files),
        "first_5_files": files[:5]
    }

# Test the helper functions
print("Testing helper functions:")
print("=" * 40)

# Test folder stats
boxes_stats = get_folder_stats(os.path.join(chair_dir, "boxes"))
print(f"Boxes folder stats: {boxes_stats}")

# Test exploring another file type
labels_dir = os.path.join(chair_dir, "labels")
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.mat')])

if label_files:
    first_label_file = os.path.join(labels_dir, label_files[0])
    explore_mat_file(first_label_file)


# In[14]:


# Tree structure implementation for hierarchical representation
class Tree:
    """
    Tree structure representing the hierarchical organization of object parts.

    The tree encodes how primitive parts are combined using adjacency (ADJ) 
    and symmetry (SYM) operations to form complex objects.
    """

    class NodeType(Enum):
        """Enumeration of node types in the hierarchical tree"""
        BOX = 0    # Leaf node containing a primitive part (bounding box)
        ADJ = 1    # Adjacency operation (combining two sub-trees)
        SYM = 2    # Symmetry operation (applying symmetry to a sub-tree)

    class Node:
        """Node in the hierarchical tree structure"""
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None, label=None):
            self.box = box      # Bounding box parameters (for BOX nodes)
            self.sym = sym      # Symmetry parameters (for SYM nodes)
            self.left = left    # Left child node
            self.right = right  # Right child node (for ADJ nodes)
            self.node_type = node_type  # Type of node (BOX, ADJ, or SYM)
            self.label = label  # Part label/identifier

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX and self.box is not None

        def is_adj(self):
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            return self.node_type == Tree.NodeType.SYM

        def __repr__(self):
            return f"Node(type={self.node_type}, label={self.label})"

    def __init__(self, boxes, ops, syms, labels):
        """
        Construct a tree from the hierarchical representation

        Parameters:
        boxes (torch.Tensor): Primitive parts (bounding boxes)
        ops (torch.Tensor): Operations sequence defining the hierarchy
        syms (torch.Tensor): Symmetry parameters
        labels (torch.Tensor): Part labels
        """
        # Convert tensors to lists and reverse for stack processing
        box_list = [b for b in torch.split(boxes, 1, 0)]
        sym_param = [s for s in torch.split(syms, 1, 0)]
        label_list = [l for l in labels[0]]

        box_list.reverse()
        sym_param.reverse()
        label_list.reverse()

        queue = []

        # Process each operation in the sequence
        for id in range(ops.size()[1]):
            if ops[0, id] == Tree.NodeType.BOX.value:
                # BOX operation: create a leaf node
                queue.append(Tree.Node(
                    box=box_list.pop(), 
                    node_type=Tree.NodeType.BOX, 
                    label=label_list.pop()
                ))
            elif ops[0, id] == Tree.NodeType.ADJ.value:
                # ADJ operation: combine two nodes
                left_node = queue.pop()
                right_node = queue.pop()
                queue.append(Tree.Node(
                    left=left_node, 
                    right=right_node, 
                    node_type=Tree.NodeType.ADJ
                ))
            elif ops[0, id] == Tree.NodeType.SYM.value:
                # SYM operation: apply symmetry to a node
                node = queue.pop()
                queue.append(Tree.Node(
                    left=node, 
                    sym=sym_param.pop(), 
                    node_type=Tree.NodeType.SYM
                ))

        assert len(queue) == 1, "Tree construction should result in single root node"
        self.root = queue[0]

    def traverse(self, node=None, depth=0):
        """Traverse the tree and print its structure"""
        if node is None:
            node = self.root

        indent = "  " * depth
        print(f"{indent}{node}")

        if node.left:
            self.traverse(node.left, depth + 1)
        if node.right:
            self.traverse(node.right, depth + 1)

# Test the tree structure
print("Testing tree structure:")
print("=" * 40)

# Load sample data to test tree construction
ops_dir = os.path.join(chair_dir, "ops")
ops_files = sorted([f for f in os.listdir(ops_dir) if f.endswith('.mat')])

if ops_files:
    first_ops_file = os.path.join(ops_dir, ops_files[0])
    ops_data = explore_mat_file(first_ops_file)

    print("\nTree structure explanation:")
    print("The tree represents how primitive parts are hierarchically combined:")
    print("- BOX nodes: Leaf nodes containing primitive parts")
    print("- ADJ nodes: Combine two sub-trees (adjacency operation)")
    print("- SYM nodes: Apply symmetry transformations to a sub-tree")
    print("The operations sequence defines the construction order")


# In[ ]:


# Create an interactive widget for exploring the GRASS dataset
import ipywidgets as widgets
from IPython.display import display, clear_output
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

class GRASSDatasetExplorer:
    """Interactive widget for exploring GRASS dataset with visualization"""

    def __init__(self, base_dir="dropbox"):
        self.base_dir = base_dir
        self.categories = self._get_categories()
        self.current_dataset = None
        self.setup_widgets()

    def _get_categories(self):
        """Get available object categories"""
        return [d for d in os.listdir(self.base_dir) 
                if os.path.isdir(os.path.join(self.base_dir, d))]

    def setup_widgets(self):
        """Setup the interactive widgets"""
        # Category selection
        self.category_dropdown = widgets.Dropdown(
            options=self.categories,
            value='Chair',
            description='Category:',
            style={'description_width': 'initial'}
        )

        # Model selection
        self.model_slider = widgets.IntSlider(
            value=1,
            min=1,
            max=100,
            step=1,
            description='Model ID:',
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        # Visualization type
        self.viz_type = widgets.Dropdown(
            options=['BBox Hierarchy', 'Raw Data', 'Folder Stats', '3D Visualization', 'Model Mesh'],
            value='BBox Hierarchy',
            description='View:',
            style={'description_width': 'initial'}
        )

        # Action button
        self.explore_button = widgets.Button(
            description='Explore',
            button_style='primary',
            tooltip='Explore the selected dataset'
        )

        # Output area
        self.output = widgets.Output()

        # Set up event handlers
        self.explore_button.on_click(self.on_explore_click)
        self.category_dropdown.observe(self.on_category_change, names='value')

        # Display widgets
        display(widgets.VBox([
            widgets.HBox([self.category_dropdown, self.model_slider]),
            self.viz_type,
            self.explore_button,
            self.output
        ]))

    def on_category_change(self, change):
        """Update model slider range when category changes"""
        category = change['new']
        category_dir = os.path.join(self.base_dir, category)

        # Find maximum model ID
        boxes_dir = os.path.join(category_dir, "boxes")
        if os.path.exists(boxes_dir):
            mat_files = [f for f in os.listdir(boxes_dir) if f.endswith('.mat')]
            max_id = len(mat_files)
            self.model_slider.max = max(1, max_id)

    def load_dataset(self, category, model_id):
        """Load GRASS dataset for specific category and model"""
        category_dir = os.path.join(self.base_dir, category)

        try:
            # Load all components
            boxes = torch.from_numpy(
                loadmat(os.path.join(category_dir, 'boxes', f'{model_id}.mat'))['box']
            ).t().float()

            ops = torch.from_numpy(
                loadmat(os.path.join(category_dir, 'ops', f'{model_id}.mat'))['op']
            ).int()

            syms = torch.from_numpy(
                loadmat(os.path.join(category_dir, 'syms', f'{model_id}.mat'))['sym']
            ).t().float()

            labels = torch.from_numpy(
                loadmat(os.path.join(category_dir, 'labels', f'{model_id}.mat'))['label']
            ).int()

            # Create tree
            tree = Tree(boxes, ops, syms, labels)
            return tree

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def show_bbox_hierarchy(self, tree):
        """Display bounding box hierarchy"""
        with self.output:
            clear_output(wait=True)
            print("Bounding Box Hierarchy:")
            print("=" * 40)
            tree.traverse()

            # Count node types
            def count_nodes(node):
                if node is None:
                    return {'BOX': 0, 'ADJ': 0, 'SYM': 0}

                counts = {
                    'BOX': 1 if node.is_leaf() else 0,
                    'ADJ': 1 if node.is_adj() else 0,
                    'SYM': 1 if node.is_sym() else 0
                }

                if node.left:
                    left_counts = count_nodes(node.left)
                    counts = {k: counts[k] + left_counts[k] for k in counts}
                if node.right:
                    right_counts = count_nodes(node.right)
                    counts = {k: counts[k] + right_counts[k] for k in counts}

                return counts

            counts = count_nodes(tree.root)
            print(f"\nNode counts: BOX={counts['BOX']}, ADJ={counts['ADJ']}, SYM={counts['SYM']}")

    def show_raw_data(self, category, model_id):
        """Show raw data from all files"""
        with self.output:
            clear_output(wait=True)
            print(f"Raw Data for {category} model {model_id}:")
            print("=" * 50)

            category_dir = os.path.join(self.base_dir, category)
            subfolders = ['boxes', 'labels', 'ops', 'syms']

            for folder in subfolders:
                file_path = os.path.join(category_dir, folder, f'{model_id}.mat')
                if os.path.exists(file_path):
                    print(f"\n{folder.upper()} data:")
                    explore_mat_file(file_path)
                else:
                    print(f"\nNo file found: {file_path}")

    def show_folder_stats(self, category):
        """Show statistics for all subfolders"""
        with self.output:
            clear_output(wait=True)
            print(f"Folder Statistics for {category}:")
            print("=" * 40)

            category_dir = os.path.join(self.base_dir, category)
            subfolders = ['boxes', 'labels', 'models', 'obbs', 'ops', 'part mesh indices', 'syms']

            for folder in subfolders:
                folder_path = os.path.join(category_dir, folder)
                stats = get_folder_stats(folder_path)

                print(f"\n{folder}:")
                if stats['exists']:
                    print(f"  Files: {stats['total_files']}")
                    if stats['mat_files'] > 0:
                        print(f"  .mat files: {stats['mat_files']}")
                    if stats['obj_files'] > 0:
                        print(f"  .obj files: {stats['obj_files']}")
                    if stats['txt_files'] > 0:
                        print(f"  .txt files: {stats['txt_files']}")
                    print(f"  Sample: {stats['first_5_files']}")
                else:
                    print("  Folder not found")

    def show_3d_visualization(self, tree):
        """Show 3D visualization of the bounding boxes"""
        with self.output:
            clear_output(wait=True)
            print("3D Bounding Box Visualization")
            print("=" * 40)

            # Decode the hierarchical structure into flat boxes
            def decode_boxes(node):
                if node.is_leaf():
                    return [node.box.squeeze().numpy()]

                boxes = []
                if node.left:
                    boxes.extend(decode_boxes(node.left))
                if node.right:
                    boxes.extend(decode_boxes(node.right))

                return boxes

            boxes = decode_boxes(tree.root)
            print(f"Decoded {len(boxes)} bounding boxes")

            # Simple matplotlib 3D visualization
            try:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                colors = plt.cm.tab10(np.linspace(0, 1, len(boxes)))

                for i, box in enumerate(boxes):
                    if box.shape[0] >= 12:  # Ensure we have enough parameters
                        center = box[0:3]
                        lengths = box[3:6]
                        dir1 = box[6:9] / np.linalg.norm(box[6:9])
                        dir2 = box[9:12] / np.linalg.norm(box[9:12])
                        dir3 = np.cross(dir1, dir2)

                        # Create box corners
                        d1 = 0.5 * lengths[0] * dir1
                        d2 = 0.5 * lengths[1] * dir2
                        d3 = 0.5 * lengths[2] * dir3

                        corners = np.array([
                            center - d1 - d2 - d3,
                            center - d1 + d2 - d3,
                            center + d1 - d2 - d3,
                            center + d1 + d2 - d3,
                            center - d1 - d2 + d3,
                            center - d1 + d2 + d3,
                            center + d1 - d2 + d3,
                            center + d1 + d2 + d3
                        ])

                        # Plot edges
                        edges = [
                            (0,1),(0,2),(1,3),(2,3),
                            (4,5),(4,6),(5,7),(6,7),
                            (0,4),(1,5),(2,6),(3,7)
                        ]

                        for start, end in edges:
                            ax.plot([corners[start,0], corners[end,0]],
                                   [corners[start,1], corners[end,1]],
                                   [corners[start,2], corners[end,2]],
                                   color=colors[i], linewidth=2)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('3D Bounding Boxes')
                plt.show()

            except Exception as e:
                print(f"3D visualization error: {e}")
                print("Showing box parameters instead:")
                for i, box in enumerate(boxes):
                    print(f"Box {i}: {box[:6]}...")  # Show first 6 parameters

    def show_model_mesh(self, category, model_id):
        """Show the actual 3D model mesh if available"""
        with self.output:
            clear_output(wait=True)
            print("3D Model Mesh")
            print("=" * 40)

            model_path = os.path.join(self.base_dir, category, 'models', f'{model_id}.obj')

            if os.path.exists(model_path):
                try:
                    import trimesh
                    mesh = trimesh.load(model_path)
                    print(f"Mesh loaded: {mesh}")
                    print(f"Vertices: {mesh.vertices.shape}")
                    print(f"Faces: {mesh.faces.shape}")

                    # Simple visualization
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], 
                                   mesh.vertices[:,2], triangles=mesh.faces,
                                   alpha=0.8, cmap='viridis')

                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title('3D Model Mesh')
                    plt.show()

                except Exception as e:
                    print(f"Mesh loading error: {e}")
            else:
                print(f"Model file not found: {model_path}")

    def on_explore_click(self, b):
        """Handle explore button click"""
        category = self.category_dropdown.value
        model_id = self.model_slider.value
        viz_type = self.viz_type.value

        try:
            if viz_type == 'BBox Hierarchy':
                tree = self.load_dataset(category, model_id)
                if tree:
                    self.show_bbox_hierarchy(tree)
                else:
                    with self.output:
                        print(f"Could not load model {model_id} from {category}")

            elif viz_type == 'Raw Data':
                self.show_raw_data(category, model_id)

            elif viz_type == 'Folder Stats':
                self.show_folder_stats(category)

            elif viz_type == '3D Visualization':
                tree = self.load_dataset(category, model_id)
                if tree:
                    self.show_3d_visualization(tree)
                else:
                    with self.output:
                        print(f"Could not load model {model_id} from {category}")

            elif viz_type == 'Model Mesh':
                self.show_model_mesh(category, model_id)

        except Exception as e:
            with self.output:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

# Create and display the explorer
print("GRASS Dataset Explorer with Visualization")
print("=" * 40)
explorer = GRASSDatasetExplorer()


# In[ ]:




