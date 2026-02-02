#!/usr/bin/env python
# coding: utf-8

# In[23]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[24]:


## 1. Setup Paths & Configuration

import sys
import os
from pathlib import Path

# Add source directory to path
current_path = Path.cwd()
base_project_dir = current_path.parent
src_dir = base_project_dir / "src"

if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Define key directories
dataset_directory = src_dir / "abstractionsshapecoder" / "dataset"
saved_directory = src_dir / "abstractionsshapecoder" / "saved"

# --- AE Model Paths ---
saved_models_L1_AE_dir = saved_directory / "models_L1_AE"
saved_models_L2_AE_dir = saved_directory / "models_L2_AE"
saved_models_L1_AE_dir.mkdir(parents=True, exist_ok=True)
saved_models_L2_AE_dir.mkdir(parents=True, exist_ok=True)

# --- PCA Model Paths ---
saved_models_L1_PCA_dir = saved_directory / "models_L1_PCA"
saved_models_L2_PCA_dir = saved_directory / "models_L2_PCA"
saved_models_L1_PCA_dir.mkdir(parents=True, exist_ok=True)
saved_models_L2_PCA_dir.mkdir(parents=True, exist_ok=True)

print(f"Base project directory: {base_project_dir}")
print(f"Source directory: {src_dir}")
print(f"L1 AE Models directory: {saved_models_L1_AE_dir}")
print(f"L2 AE Models directory: {saved_models_L2_AE_dir}")
print(f"L1 PCA Models directory: {saved_models_L1_PCA_dir}")
print(f"L2 PCA Models directory: {saved_models_L2_PCA_dir}")


# In[25]:


import pickle
from pathlib import Path
from abstractionsshapecoder.debug_utils import debug_info, debug_success, debug_error

def load_processed_datasets():
    """
    Loads all hierarchical datasets (L0, L1-VAE, L1-AE, L2-AE) from the saved directory.
    """
    SAVED_DIR = Path.cwd().parent / "src" / "abstractionsshapecoder" / "saved"

    # Define file map: {Variable_Name: Filename}
    dataset_files = {
        "shapes_l0": "all_dsl_shapes.pkl",
        "shapes_l1_vae": "all_abstracted_shapes_L1_VAE.pkl",
        "shapes_l1_ae": "all_abstracted_shapes_L1_AE.pkl",
        "shapes_l2_ae": "all_abstracted_shapes_L2_AE.pkl"
    }

    loaded_datasets = {}

    for var_name, filename in dataset_files.items():
        file_path = SAVED_DIR / filename
        if file_path.exists():
            debug_info(f"Loading {var_name} from {filename}...")
            with open(file_path, "rb") as f:
                loaded_datasets[var_name] = pickle.load(f)
            debug_success(f"Successfully loaded {len(loaded_datasets[var_name])} shapes into {var_name}.")
        else:
            debug_error(f"Missing file: {filename}. Ensure previous training cells were run.")
            loaded_datasets[var_name] = {}

    return loaded_datasets

# Execute the load
datasets = load_processed_datasets()

# Extract into global namespace for easier access in subsequent cells
shapes_l0 = datasets.get("shapes_l0")


# In[44]:


print(shapes_l0["2741_0_343"]["dsl"])


# In[45]:


shapes_l0["172_0_0"]


# In[46]:


shapes_l0["172_0_0"]["dsl"]


# In[47]:


print(shapes_l0["172_0_0"]["dsl"])


# In[48]:


from __future__ import annotations
from egglog import *

class Shape(Expr):
    def __init__(self, name: StringLike) -> None: ...

    # Union as an operator for readability
    def __add__(self, other: Shape) -> Shape: ...

@function
def cuboid(w: f64Like, h: f64Like, d: f64Like) -> Shape: ...

@function
def translate(s: Shape, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...

@function
def rotate(s: Shape, x: f64Like, y: f64Like, z: f64Like, w: f64Like) -> Shape: ...


# In[49]:


def to_egglog(node):
    """Recursively converts DSL Python objects to egglog expressions."""
    from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union

    if isinstance(node, Cuboid):
        return cuboid(node.size[0], node.size[1], node.size[2])

    elif isinstance(node, Translate):
        child = to_egglog(node.child)
        return translate(child, node.vector[0], node.vector[1], node.vector[2])

    elif isinstance(node, Rotate):
        child = to_egglog(node.child)
        return rotate(child, node.quaternion[0], node.quaternion[1], 
                      node.quaternion[2], node.quaternion[3])

    elif isinstance(node, Union):
        return to_egglog(node.left) + to_egglog(node.right)

    raise TypeError(f"Unknown node type: {type(node)}")

# usage:
egraph = EGraph()
egg_shape = egraph.let("my_shape", to_egglog(shapes_l0["2741_0_343"]["dsl"]))


# In[50]:


print(egg_shape)


# In[51]:


# Extract the expression from the e-graph
extracted_shape = egraph.extract(egg_shape)
print(extracted_shape)


# In[52]:


# This returns a tuple: (Expression, Cost)
extracted_shape, cost = egraph.extract(egg_shape, include_cost=True)
print(f"Cost: {cost}")
print(f"Shape: {extracted_shape}")


# In[53]:


import black

# Convert the expression to a string and use black to format it like code
pretty_shape = black.format_str(str(extracted_shape), mode=black.FileMode())
print(pretty_shape)


# In[55]:


from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union

# Define the shared addition behavior
def dsl_add(self, other):
    return Union(self, other)

# Patch all your classes so they support the '+' operator
for cls in [Cuboid, Translate, Rotate, Union]:
    cls.__add__ = dsl_add

def from_egglog_string(egg_expr):
    expr_str = str(egg_expr)

    # We don't even need "__add__" in the namespace now 
    # because the '+' symbol in the string will trigger the methods we patched above.
    namespace = {
        "cuboid": lambda w, h, d: Cuboid(size=[w, h, d]),
        "translate": lambda child, x, y, z: Translate(child, vector=[x, y, z]),
        "rotate": lambda child, x, y, z, w: Rotate(child, quaternion=[x, y, z, w]),
    }

    return eval(expr_str, {"__builtins__": {}}, namespace)

# --- Usage ---
extracted = egraph.extract(egg_shape)
restored_dsl = from_egglog_string(extracted)

print("--- Successfully Restored DSL Tree ---")
print(restored_dsl)

# --- Usage ---
extracted = egraph.extract(egg_shape)
restored_dsl = from_egglog_string(extracted)

print("--- Successfully Restored DSL Tree ---")
print(restored_dsl)


# In[ ]:




