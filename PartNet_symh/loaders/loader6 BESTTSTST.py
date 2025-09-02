#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io as sio

# Base directory for Chair category
chair_dir = "dropbox/Chair"

# Subfolders
folders = ["boxes", "labels", "models", "obbs", "ops", "part mesh indices", "syms"]


# In[2]:


for folder in folders:
    folder_path = os.path.join(chair_dir, folder)
    files = sorted(os.listdir(folder_path))
    print(f"\nFolder: {folder} | Number of files: {len(files)}")
    print("First 5 files:", files[:5])


# In[3]:


# boxes

boxes_path = os.path.join(chair_dir, "boxes", sorted(os.listdir(os.path.join(chair_dir, "boxes")))[0])
print("Boxes file path:", boxes_path)

boxes_mat = sio.loadmat(boxes_path)
print("Keys in .mat file:", boxes_mat.keys())

# Print shapes of arrays
for key in boxes_mat:
    if not key.startswith("__"):
        print(f"{key} shape:", boxes_mat[key].shape)
        print(boxes_mat[key])


# In[4]:


# labels

labels_path = os.path.join(chair_dir, "labels", sorted(os.listdir(os.path.join(chair_dir, "labels")))[0])
print("Labels file path:", labels_path)

labels_mat = sio.loadmat(labels_path)
print("Keys in .mat file:", labels_mat.keys())

for key in labels_mat:
    if not key.startswith("__"):
        print(f"{key} shape:", labels_mat[key].shape)
        print(labels_mat[key])


# In[5]:


# models

import trimesh

models_path = os.path.join(chair_dir, "models", sorted(os.listdir(os.path.join(chair_dir, "models")))[0])
print("Model file path:", models_path)

# Load mesh
mesh = trimesh.load(models_path)
print(mesh)
print("Vertices shape:", mesh.vertices.shape)
print("Faces shape:", mesh.faces.shape)


# In[6]:


# obbs

obbs_path = os.path.join(chair_dir, "obbs", sorted(os.listdir(os.path.join(chair_dir, "obbs")))[0])
print("OBB file path:", obbs_path)

with open(obbs_path, 'r') as f:
    lines = f.readlines()

print(f"First 20 lines of {os.path.basename(obbs_path)}:")
for line in lines[:20]:
    print(line.strip())


# In[7]:


# ops

ops_path = os.path.join(chair_dir, "ops", sorted(os.listdir(os.path.join(chair_dir, "ops")))[0])
print("Ops file path:", ops_path)

ops_mat = sio.loadmat(ops_path)
print("Keys in .mat file:", ops_mat.keys())

for key in ops_mat:
    if not key.startswith("__"):
        print(f"{key} shape:", ops_mat[key].shape)
        print(ops_mat[key][:10])  # print first 10 entries


# In[8]:


# part mesh indices

part_mesh_path = os.path.join(chair_dir, "part mesh indices", sorted(os.listdir(os.path.join(chair_dir, "part mesh indices")))[0])
print("Part mesh indices file path:", part_mesh_path)

part_mesh_mat = sio.loadmat(part_mesh_path)
print("Keys in .mat file:", part_mesh_mat.keys())

for key in part_mesh_mat:
    if not key.startswith("__"):
        print(f"{key} shape:", part_mesh_mat[key].shape)
        print(part_mesh_mat[key])


# In[9]:


# syms

syms_path = os.path.join(chair_dir, "syms", sorted(os.listdir(os.path.join(chair_dir, "syms")))[0])
print("Syms file path:", syms_path)

syms_mat = sio.loadmat(syms_path)
print("Keys in .mat file:", syms_mat.keys())

for key in syms_mat:
    if not key.startswith("__"):
        print(f"{key} type: {type(syms_mat[key])}, shape: {getattr(syms_mat[key], 'shape', None)}")
        print(syms_mat[key])


# In[10]:


boxes = boxes_mat['box']  # shape (12,3)
num_parts = boxes.shape[0] // 4

for i in range(num_parts):
    part_box = boxes[i*4:(i+1)*4]
    print(f"Part {i} box points:")
    print(part_box)


# In[11]:


# Cell 1: Load a box file and inspect
import numpy as np
from scipy.io import loadmat

# Path to a single boxes file
box_file = 'dropbox/Chair/boxes/1.mat'

# Load .mat file
boxes_mat = loadmat(box_file)
print("Keys in .mat file:", boxes_mat.keys())

# Extract box data
boxes = boxes_mat['box']  # shape (N_parts, 12)
print("Boxes shape:", boxes.shape)
print("First box vector:", boxes[0])


# In[12]:


# Cell 2: Function to convert 12-dim box vector to 8 corner points
import numpy.linalg as LA

def get_box_corners(box_vec):
    """
    box_vec: 12 numbers -> center + dir0 + dir1 + dir2
    returns: 8x3 array of corner points
    """
    center = box_vec[0:3]
    dir0 = box_vec[3:6]
    dir1 = box_vec[6:9]
    dir2 = box_vec[9:12]

    dir0 = dir0 / LA.norm(dir0)
    dir1 = dir1 / LA.norm(dir1)
    dir2 = dir2 / LA.norm(dir2)

    # Half lengths along each axis (assuming normalized directions)
    d0 = 0.5 * dir0
    d1 = 0.5 * dir1
    d2 = 0.5 * dir2

    # Compute 8 corners
    corners = np.array([
        center - d0 - d1 - d2,
        center - d0 + d1 - d2,
        center + d0 - d1 - d2,
        center + d0 + d1 - d2,
        center - d0 - d1 + d2,
        center - d0 + d1 + d2,
        center + d0 - d1 + d2,
        center + d0 + d1 + d2,
    ])

    return corners


# In[13]:


# Cell 3: Visualize boxes in 3D using matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_box(ax, corners, color='r'):
    # Draw box edges
    edges = [
        (0,1),(0,2),(1,3),(2,3),
        (4,5),(4,6),(5,7),(6,7),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i,j in edges:
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]], color=color)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, box_vec in enumerate(boxes):
    corners = get_box_corners(box_vec)
    plot_box(ax, corners, color=plt.cm.jet(i/len(boxes)))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Boxes Visualization')
plt.show()


# In[ ]:


# Cell 1: Load the box file
import numpy as np
from scipy.io import loadmat

box_file = 'dropbox/Chair/boxes/1.mat'
boxes_mat = loadmat(box_file)
boxes = boxes_mat['box']  # shape (N_parts, 12)
print("Boxes shape:", boxes.shape)


# In[ ]:


# Cell 2: Function to convert 12-dim box vector to 8 corners
import numpy.linalg as LA

def get_box_corners(box_vec):
    center = box_vec[0:3]
    dir0 = box_vec[3:6]
    dir1 = box_vec[6:9]
    dir2 = box_vec[9:12]

    dir0 = dir0 / LA.norm(dir0)
    dir1 = dir1 / LA.norm(dir1)
    dir2 = dir2 / LA.norm(dir2)

    d0 = 0.5 * dir0
    d1 = 0.5 * dir1
    d2 = 0.5 * dir2

    corners = np.array([
        center - d0 - d1 - d2,
        center - d0 + d1 - d2,
        center + d0 - d1 - d2,
        center + d0 + d1 - d2,
        center - d0 - d1 + d2,
        center - d0 + d1 + d2,
        center + d0 - d1 + d2,
        center + d0 + d1 + d2,
    ])
    return corners


# In[ ]:


# Cell 3: Function to create k3d mesh from corners
import k3d

def box_to_k3d_mesh(corners, color=0xFF0000):
    """
    corners: 8x3 array
    returns: k3d mesh object
    """
    # define faces of cube (12 triangles)
    faces = np.array([
        [0,1,2],[1,3,2],
        [4,5,6],[5,7,6],
        [0,1,4],[1,5,4],
        [2,3,6],[3,7,6],
        [0,2,4],[2,6,4],
        [1,3,5],[3,7,5]
    ], dtype=np.uint32)

    vertices = corners.astype(np.float32)
    indices = faces.flatten().astype(np.uint32)

    return k3d.mesh(vertices, indices, color=color, wireframe=True)


# In[14]:


# Cell 4: Visualize all boxes in k3d
plot = k3d.plot()

for i, box_vec in enumerate(boxes):
    corners = get_box_corners(box_vec)
    mesh = box_to_k3d_mesh(corners, color=int(0xFF0000*(i/len(boxes))))
    plot += mesh

plot.display()


# In[15]:


# loader.ipynb
# ==========================
# Notebook for loading and visualizing shapes from GRASSDataset

# Cell 1: Imports
from enum import Enum
import os
import torch
from scipy.io import loadmat
from torch.utils import data
import math
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# Cell 2: Tree and GRASSDataset classes
# --------------------------
class Tree(object):
    class NodeType(Enum):
        BOX = 0
        ADJ = 1
        SYM = 2

    class Node(object):
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None, label=None):
            self.box = box
            self.sym = sym
            self.left = left
            self.right = right
            self.node_type = node_type
            self.label = label

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX and self.box is not None

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
        for id in range(ops.size()[1]):
            if ops[0, id] == Tree.NodeType.BOX.value:
                queue.append(Tree.Node(box=box_list.pop(), node_type=Tree.NodeType.BOX, label=label_list.pop()))
            elif ops[0, id] == Tree.NodeType.ADJ.value:
                left_node = queue.pop()
                right_node = queue.pop()
                queue.append(Tree.Node(left=left_node, right=right_node, node_type=Tree.NodeType.ADJ))
            elif ops[0, id] == Tree.NodeType.SYM.value:
                node = queue.pop()
                queue.append(Tree.Node(left=node, sym=sym_param.pop(), node_type=Tree.NodeType.SYM))
        assert len(queue) == 1
        self.root = queue[0]

class GRASSDataset(data.Dataset):
    def __init__(self, dir, models_num=0, transform=None):
        self.dir = dir
        num_examples = len(os.listdir(os.path.join(dir, 'ops')))
        self.transform = transform
        self.trees = []
        for i in range(models_num):
            boxes = torch.from_numpy(loadmat(os.path.join(dir, 'boxes', '%d.mat' % (i+1)))['box']).t().float()
            ops = torch.from_numpy(loadmat(os.path.join(dir, 'ops', '%d.mat' % (i+1)))['op']).int()
            syms = torch.from_numpy(loadmat(os.path.join(dir, 'syms', '%d.mat' % (i+1)))['sym']).t().float()
            labels = torch.from_numpy(loadmat(os.path.join(dir, 'labels', '%d.mat' % (i+1)))['label']).int()
            tree = Tree(boxes, ops, syms, labels)
            self.trees.append(tree)

    def __getitem__(self, index):
        return self.trees[index]

    def __len__(self):
        return len(self.trees)

# --------------------------
# Cell 3: 3D Visualization functions
# --------------------------
def draw(ax, p, color):
    center = p[0:3]
    lengths = p[3:6]
    dir_1 = p[6:9] / LA.norm(p[6:9])
    dir_2 = p[9:12] / LA.norm(p[9:12])
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3 / LA.norm(dir_3)
    d1, d2, d3 = 0.5*lengths[0]*dir_1, 0.5*lengths[1]*dir_2, 0.5*lengths[2]*dir_3
    cornerpoints = np.array([
        center - d1 - d2 - d3, center - d1 + d2 - d3, center + d1 - d2 - d3, center + d1 + d2 - d3,
        center - d1 - d2 + d3, center - d1 + d2 + d3, center + d1 - d2 + d3, center + d1 + d2 + d3
    ])
    edges = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        ax.plot(*zip(cornerpoints[e[0]], cornerpoints[e[1]]), c=color)

def showGenshape(genshape):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.7,0.7); ax.set_ylim(-0.7,0.7); ax.set_zlim(-0.7,0.7)
    cmap = plt.get_cmap('jet_r')
    for jj, box in enumerate(genshape):
        p = box.squeeze(0).numpy() if isinstance(box, torch.Tensor) else box
        draw(ax, p, cmap(float(jj)/len(genshape)))
    plt.show()

# --------------------------
# Cell 4: Decode structure
# --------------------------
def vrrotvec2mat(rotvector):
    s, c, t = math.sin(rotvector[3]), math.cos(rotvector[3]), 1 - math.cos(rotvector[3])
    x, y, z = rotvector[0], rotvector[1], rotvector[2]
    m = torch.FloatTensor([
        [t*x*x+c, t*x*y-s*z, t*x*z+s*y],
        [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
        [t*x*z-s*y, t*y*z+s*x, t*z*z+c]
    ])
    return m

def decode_structure(root):
    syms = [torch.ones(8).mul(10)]
    stack = [root]
    boxes = []
    while stack:
        node = stack.pop()
        node_type = node.node_type.value
        if node_type == 1:  # ADJ
            stack.append(node.left)
            stack.append(node.right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        if node_type == 2:  # SYM
            stack.append(node.left)
            syms.pop()
            syms.append(node.sym.squeeze(0))
        if node_type == 0:  # BOX
            reBox = node.box
            reBoxes = [reBox]
            s = syms.pop()
            l1, l2, l3 = abs(s[0]+1), abs(s[0]), abs(s[0]-1)
            # Symmetry/Translation/Reflection operations
            if l1 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                f1 = torch.cat([sList[1], sList[2], sList[3]]); f1 = f1/torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1/s[7].item())
                for i in range(folds-1):
                    rotvector = torch.cat([f1, sList[7]*2*3.1415*(i+1)])
                    rotm = vrrotvec2mat(rotvector)
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = rotm.matmul(center - f2) + f2
                    newbox = torch.cat([newcenter, dir0, rotm.matmul(dir1), rotm.matmul(dir2)])
                    reBoxes.append(newbox)
            boxes.extend(reBoxes)
    return boxes

# --------------------------
# Cell 5: Load and visualize one shape
# --------------------------
dataset = GRASSDataset('dropbox/Chair', models_num=1)
tree = dataset[0]
boxes = decode_structure(tree.root)
showGenshape(boxes)


# In[16]:


import k3d
import numpy as np
import torch

def showGenshape_k3d(genshape):
    plot = k3d.plot()
    for jj, box in enumerate(genshape):
        p = box.squeeze(0).numpy() if isinstance(box, torch.Tensor) else box
        center = p[0:3]
        lengths = p[3:6]
        dir1 = p[6:9] / np.linalg.norm(p[6:9])
        dir2 = p[9:12] / np.linalg.norm(p[9:12])
        dir3 = np.cross(dir1, dir2)
        dir3 = dir3 / np.linalg.norm(dir3)
        d1, d2, d3 = 0.5*lengths[0]*dir1, 0.5*lengths[1]*dir2, 0.5*lengths[2]*dir3

        # 8 corners of the box
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

        # edges as pairs of corner indices
        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7)
        ]

        for start, end in edges:
            line_pts = np.vstack([corners[start], corners[end]]).astype(np.float32)
            line = k3d.line(line_pts, shader='thick', width=0.01)
            plot += line

    plot.display()


# In[17]:


dataset = GRASSDataset('dropbox/Bag', models_num=1)
tree = dataset[0]
boxes = decode_structure(tree.root)
showGenshape_k3d(boxes)


# In[26]:


import k3d
import numpy as np
import torch

# Assign colors to node types
NODE_COLORS = {
    0: 0x1f77b4,  # BOX - blue
    1: 0xff7f0e,  # ADJ - orange
    2: 0x2ca02c   # SYM - green
}

def showGenshape_k3d_colored(boxes, node_types=None, labels=None):
    """
    boxes: list of torch tensors or numpy arrays (decoded boxes)
    node_types: list of same length as boxes, values in [0,1,2] (optional)
    labels: list of integers for part labels (optional)
    """
    plot = k3d.plot()
    for i, box in enumerate(boxes):
        p = box.squeeze(0).numpy() if isinstance(box, torch.Tensor) else box
        center = p[0:3]
        lengths = p[3:6]
        dir1 = p[6:9] / np.linalg.norm(p[6:9])
        dir2 = p[9:12] / np.linalg.norm(p[9:12])
        dir3 = np.cross(dir1, dir2)
        dir3 = dir3 / np.linalg.norm(dir3)
        d1, d2, d3 = 0.5*lengths[0]*dir1, 0.5*lengths[1]*dir2, 0.5*lengths[2]*dir3

        # 8 corners of the box
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

        # Decide color
        if labels is not None:
            # Map label to a color using a hash
            color = int(labels[i] * 1234567) % 0xFFFFFF
        elif node_types is not None:
            color = NODE_COLORS.get(node_types[i], 0x888888)  # default grey
        else:
            color = 0x1f77b4  # default blue

        # edges as pairs of corner indices
        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7)
        ]

        for start, end in edges:
            line_pts = np.vstack([corners[start], corners[end]]).astype(np.float32)
            line = k3d.line(line_pts, shader='thick', width=0.01, color=color)
            plot += line

    plot.display()


# In[27]:


def decode_structure_with_labels(root):
    syms = [torch.ones(8).mul(10)]
    stack = [root]
    boxes = []
    labels = []

    while stack:
        node = stack.pop()
        node_type = node.node_type.value
        if node_type == 1:  # ADJ
            stack.append(node.left)
            stack.append(node.right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        elif node_type == 2:  # SYM
            stack.append(node.left)
            syms.pop()
            syms.append(node.sym.squeeze(0))
        elif node_type == 0:  # BOX
            reBox = node.box
            reBoxes = [reBox]
            s = syms.pop()
            boxes.extend(reBoxes)
            labels.extend([node.label.item() for _ in reBoxes])  # store label
    return boxes, labels


# In[28]:


boxes, node_types = decode_structure_with_types(tree.root)
showGenshape_k3d_colored(boxes, node_types=node_types)


# In[22]:


import matplotlib.pyplot as plt
import k3d
import numpy as np
import torch

def showGenshape_k3d_labels(boxes, labels):
    plot = k3d.plot()
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    cmap = plt.get_cmap('tab20')  # 20 distinct colors

    # Map each label to a k3d color
    label_to_color = {}
    for i, label in enumerate(unique_labels):
        r, g, b, _ = cmap(i / n_labels)  # ignore alpha
        color = (int(r*255) << 16) + (int(g*255) << 8) + int(b*255)
        label_to_color[label] = color

    for i, box in enumerate(boxes):
        p = box.squeeze(0).numpy() if isinstance(box, torch.Tensor) else box
        center = p[0:3]
        lengths = p[3:6]
        dir1 = p[6:9] / np.linalg.norm(p[6:9])
        dir2 = p[9:12] / np.linalg.norm(p[9:12])
        dir3 = np.cross(dir1, dir2)
        dir3 /= np.linalg.norm(dir3)
        d1, d2, d3 = 0.5*lengths[0]*dir1, 0.5*lengths[1]*dir2, 0.5*lengths[2]*dir3

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

        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7)
        ]

        color = label_to_color[labels[i]]  # get color for this label

        for start, end in edges:
            line_pts = np.vstack([corners[start], corners[end]]).astype(np.float32)
            plot += k3d.line(line_pts, shader='thick', width=0.01, color=color)

    plot.display()


# In[23]:


boxes, labels = decode_structure_with_labels(tree.root)
showGenshape_k3d_labels(boxes, labels)


# In[ ]:




