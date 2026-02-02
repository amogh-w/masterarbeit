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


# # egglog schema

# In[56]:


from __future__ import annotations
from egglog import *
from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union

# 1. Monkey-patch DSL classes to support '+' operator for string evaluation
def dsl_add(self, other):
    return Union(self, other)

for cls in [Cuboid, Translate, Rotate, Union]:
    cls.__add__ = dsl_add

# 2. Define Egglog Sorts and Functions
class Shape(Expr):
    def __add__(self, other: Shape) -> Shape: ...

@function
def cuboid(w: f64Like, h: f64Like, d: f64Like) -> Shape: ...

@function
def translate(s: Shape, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...

@function
def rotate(s: Shape, x: f64Like, y: f64Like, z: f64Like, w: f64Like) -> Shape: ...

print("Egglog schema and DSL patching complete.")


# # Bidirectional Conversion Functions

# In[66]:


def to_egglog(node) -> Shape:
    """Recursively converts DSL Python objects to egglog expressions with validation."""
    from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union

    try:
        if isinstance(node, Cuboid):
            return cuboid(*node.size[:3])

        elif isinstance(node, Translate):
            child = to_egglog(node.child)
            return translate(child, *node.vector[:3])

        elif isinstance(node, Rotate):
            child = to_egglog(node.child)
            # Ensure we have all 4 quaternion components
            return rotate(child, *node.quaternion[:4])

        elif isinstance(node, Union):
            return to_egglog(node.left) + to_egglog(node.right)

    except (AttributeError, IndexError) as e:
        raise ValueError(f"Malformed DSL node {type(node).__name__}: {e}")

    raise TypeError(f"Unsupported node type: {type(node)}")

def from_egglog_string(egg_expr):
    """Safely converts an egglog expression back to DSL objects."""
    expr_str = str(egg_expr)

    # Mapping table for the eval namespace
    namespace = {
        "cuboid": lambda w, h, d: Cuboid(size=[w, h, d]),
        "translate": lambda child, x, y, z: Translate(child, vector=[x, y, z]),
        "rotate": lambda child, x, y, z, w: Rotate(child, quaternion=[x, y, z, w]),
    }

    try:
        # Use a restricted environment to prevent arbitrary code execution
        return eval(expr_str, {"__builtins__": {}}, namespace)
    except NameError as e:
        raise ValueError(f"Egglog expression contains unknown function: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to restore DSL from string: {e}")

print("Conversion functions defined.")


# # Optimization Rules

# In[58]:


egraph = EGraph()

@egraph.register
def geometric_rules(s: Shape, x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64):
    # Rule 1: Combine nested translations
    # T(T(shape, v1), v2) => T(shape, v1 + v2)
    yield rewrite(translate(translate(s, x1, y1, z1), x2, y2, z2)).to(
        translate(s, x1 + x2, y1 + y2, z1 + z2)
    )

    # Rule 2: Identity translation
    yield rewrite(translate(s, 0.0, 0.0, 0.0)).to(s)

print("Optimization rules registered.")


# # Execution

# In[70]:


def process_shape(shape_id: str, data_store: dict):
    egraph = EGraph()

    # 1. Safe Access
    entry = data_store.get(shape_id)
    if not entry or "dsl" not in entry:
        print(f"Error: ID {shape_id} not found.")
        return None

    try:
        # 2. Conversion & Extraction
        egg_shape = egraph.let("my_shape", to_egglog(entry["dsl"]))

        # Optimization run (Optional: add egraph.run(10) here if you have rules)

        egraph.run(10)

        extracted, cost = egraph.extract(egg_shape, include_cost=True)
        print(f"Extracted {shape_id} with cost: {cost}")

        # 3. Restoration
        return from_egglog_string(extracted)

    except Exception as e:
        print(f"Pipeline failed for {shape_id}: {e}")
        return None

# --- Usage ---
restored_dsl = process_shape("2741_0_343", shapes_l0)


# In[69]:


print(restored_dsl)

