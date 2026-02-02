#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


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


# In[4]:


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


# In[5]:


print(shapes_l0["2741_0_343"]["dsl"])


# In[6]:


shapes_l0["172_0_0"]


# In[7]:


shapes_l0["172_0_0"]["dsl"]


# In[8]:


print(shapes_l0["172_0_0"]["dsl"])


# In[ ]:





# # egglog schema

# In[32]:


from __future__ import annotations
from egglog import *

class Shape(Expr):
    def __add__(self, other: Shape) -> Shape: ...

@function
def cuboid(w: f64Like, h: f64Like, d: f64Like) -> Shape: ...

@function
def translate(s: Shape, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...

@function
def rotate(s: Shape, x: f64Like, y: f64Like, z: f64Like, w: f64Like) -> Shape: ...

@function
def sym_ref(s: Shape, axis: f64Like) -> Shape: ... # Axis as float ID: 0=X, 1=Y, 2=Z

@function
def sym_trans(s: Shape, axis: f64Like, count: f64Like, spacing: f64Like) -> Shape: ...

@function
def inv(a: f64Like) -> f64: ...

@function
def add(a: f64Like, b: f64Like) -> f64: ...


# # Bidirectional Conversion Functions

# In[44]:


def to_egglog(node) -> Shape:
    from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union, SymRef, SymTrans

    try:
        if isinstance(node, Cuboid):
            return cuboid(*node.size[:3])

        elif isinstance(node, Translate):
            return translate(to_egglog(node.child), *node.vector[:3])

        elif isinstance(node, Rotate):
            # Pass all 4 quaternion components as floats
            return rotate(to_egglog(node.child), *node.quaternion[:4])

        elif isinstance(node, Union):
            return to_egglog(node.left) + to_egglog(node.right)

        elif isinstance(node, SymRef):
            # Map "AX"->0.0, "AY"->1.0, "AZ"->2.0
            ax_id = {"AX": 0.0, "AY": 1.0, "AZ": 2.0}[node.axis]
            return sym_ref(to_egglog(node.child), ax_id)

        elif isinstance(node, SymTrans):
            ax_id = {"AX": 0.0, "AY": 1.0, "AZ": 2.0}[node.axis]
            return sym_trans(to_egglog(node.child), ax_id, node.count, node.spacing)

    except (AttributeError, IndexError, KeyError) as e:
        raise ValueError(f"Malformed DSL node {type(node).__name__}: {e}")
    raise TypeError(f"Unsupported node type: {type(node)}")

import re

import re

def from_egglog_string(egg_expr):
    from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union, SymRef, SymTrans
    expr_str = str(egg_expr).strip()

    def with_union(obj):
        if not hasattr(obj, "__add__"):
            obj.__class__.__add__ = lambda self, other: Union(self, other)
        return obj

    namespace = {
        "cuboid": lambda w, h, d: with_union(Cuboid(size=[w, h, d])),
        "translate": lambda child, x, y, z: with_union(Translate(child, vector=[x, y, z])),
        "rotate": lambda child, x, y, z, w: with_union(Rotate(child, quaternion=[x, y, z, w])),
        "sym_ref": lambda child, ax: with_union(SymRef(child, {0.0:"AX", 1.0:"AY", 2.0:"AZ"}[ax])),
        "sym_trans": lambda child, ax, c, s: with_union(SymTrans(child, {0.0:"AX", 1.0:"AY", 2.0:"AZ"}[ax], c, s)),
        "add": lambda a, b: a + b,
        "inv": lambda a: -a,
    }

    try:
        # 1. Extract all variable assignments first
        # Look for lines like: _f64_1 = 0.004...
        # We use a non-greedy check to find lines starting with an underscore and an equal sign
        assignments = re.findall(r"^(_\w+)\s*=\s*(.*)$", expr_str, re.MULTILINE)

        # 2. Identify the main body
        # The main body is the part of the string that DOES NOT start with an assignment
        body_parts = []
        for line in expr_str.splitlines():
            if not re.match(r"^_\w+\s*=", line.strip()):
                body_parts.append(line)

        # 3. Evaluate assignments in order
        for var_name, var_val in assignments:
            namespace[var_name] = eval(var_val.strip(), {"__builtins__": {}}, namespace)

        # 4. Evaluate the full remaining body as one single expression
        final_body = "\n".join(body_parts).strip()
        return eval(final_body, {"__builtins__": {}}, namespace)

    except Exception as e:
        raise RuntimeError(f"Failed to restore DSL. Error: {e}\nFull String:\n{expr_str}")


# # Optimization Rules

# In[45]:


sem_3dl_rules = ruleset()

@sem_3dl_rules.register
def rules(
    s: Shape, s2: Shape, s3: Shape,
    x: f64, y: f64, z: f64, w: f64,
    x1: f64, y1: f64, z1: f64, w1: f64,
    x2: f64, y2: f64, z2: f64, w2: f64,
    # Parameters for Symmetry/Math
    a: f64, v: f64, v1: f64, v2: f64,
    count: f64, spacing: f64,
    # Cuboid dims
    wd: f64, ht: f64, dp: f64
):
    # --- 1. Constant Folding & Math ---
    yield rewrite(add(x, y)).to(x + y)
    yield rewrite(inv(x)).to(-x)
    yield rewrite(inv(inv(x))).to(x)     # s3dl_inv_cancel
    yield rewrite(add(x, 0.0)).to(x)      # zero_add_p2
    yield rewrite(add(0.0, x)).to(x)      # zero_add_p1

    # --- 2. Symmetry Flips (s3dl_sym_axis_flip) ---
    yield rewrite(sym_ref(translate(s, x, y, z), 0.0)).to(sym_ref(translate(s, inv(x), y, z), 0.0))
    yield rewrite(sym_ref(translate(s, x, y, z), 1.0)).to(sym_ref(translate(s, x, inv(y), z), 1.0))
    yield rewrite(sym_ref(translate(s, x, y, z), 2.0)).to(sym_ref(translate(s, x, y, inv(z)), 2.0))

    # --- 3. Symmetry Definitions (Detect Symmetry Pattern) ---
    # Union(Move(Cuboid, v), Move(Cuboid, flip_v)) -> SymRef
    yield rewrite(translate(cuboid(wd, ht, dp), x1, y1, z1) + translate(cuboid(wd, ht, dp), inv(x1), y1, z1)).to(
        sym_ref(translate(cuboid(wd, ht, dp), x1, y1, z1), 0.0)
    )
    yield rewrite(translate(cuboid(wd, ht, dp), x1, y1, z1) + translate(cuboid(wd, ht, dp), x1, inv(y1), z1)).to(
        sym_ref(translate(cuboid(wd, ht, dp), x1, y1, z1), 1.0)
    )
    yield rewrite(translate(cuboid(wd, ht, dp), x1, y1, z1) + translate(cuboid(wd, ht, dp), x1, y1, inv(z1))).to(
        sym_ref(translate(cuboid(wd, ht, dp), x1, y1, z1), 2.0)
    )

    # --- 4. Symmetry Expansion (s3dl_sym_axis_def_rev) ---
    yield rewrite(sym_ref(translate(s, x, y, z), 0.0)).to(translate(s, x, y, z) + translate(s, inv(x), y, z))
    yield rewrite(sym_ref(translate(s, x, y, z), 1.0)).to(translate(s, x, y, z) + translate(s, x, inv(y), z))
    yield rewrite(sym_ref(translate(s, x, y, z), 2.0)).to(translate(s, x, y, z) + translate(s, x, y, inv(z)))

    # --- 5. Translations & Moves ---
    # s3dl_move_comb: Combine nested moves
    yield rewrite(translate(translate(s, x1, y1, z1), x2, y2, z2)).to(
        translate(s, add(x1, x2), add(y1, y2), add(z1, z2))
    )
    # s3dl_zero_mov: Remove no-op
    yield rewrite(translate(s, 0.0, 0.0, 0.0)).to(s)
    # s3dl_move_over_union_fwd/bwd
    yield rewrite(translate(s, x, y, z) + translate(s2, x, y, z)).to(translate(s + s2, x, y, z))
    yield rewrite(translate(s + s2, x, y, z)).to(translate(s, x, y, z) + translate(s2, x, y, z))

    # --- 6. Rotations ---
    # s3dl_rot_comb: Combine rotations (Assuming same quat direction for simplified combine)
    yield rewrite(rotate(rotate(s, x, y, z, w), x1, y1, z1, w1)).to(rotate(s, x, y, z, w)) # logic for combination
    # s3dl_zero_rot
    yield rewrite(rotate(s, 0.0, 0.0, 0.0, 1.0)).to(s)
    # s3dl_rot_over_union_fwd/bwd
    yield rewrite(rotate(s, x, y, z, w) + rotate(s2, x, y, z, w)).to(rotate(s + s2, x, y, z, w))
    yield rewrite(rotate(s + s2, x, y, z, w)).to(rotate(s, x, y, z, w) + rotate(s2, x, y, z, w))
    # s3dl_rot_over_move
    yield rewrite(translate(rotate(s, x, y, z, w), x1, y1, z1)).to(rotate(translate(s, x1, y1, z1), x, y, z, w))

    # --- 7. Union & Symmetry Logic ---
    # s3dl_union_comm & s3dl_union_over
    yield rewrite(s + s2).to(s2 + s)
    yield rewrite(s + (s2 + s3)).to((s + s2) + s3)
    # s3dl_union_over_sym
    yield rewrite(sym_ref(s + s2, a)).to(sym_ref(s, a) + sym_ref(s2, a))
    # s3dl_symt_move: Translational group integration
    yield rewrite(translate(sym_trans(translate(s, x1, y1, z1), a, count, spacing), x2, y2, z2)).to(
        sym_trans(translate(s, add(x1, x2), add(y1, y2), add(z1, z2)), a, count, spacing)
    )


# # Execution

# In[46]:


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


# In[47]:


print(restored_dsl)


# In[50]:


import numpy as np

def verify_pipeline(data_store: dict, limit: int = None):
    """
    Runs the full pipeline on the dataset and verifies geometric consistency.
    """
    results = {"success": 0, "failed": 0, "mismatched": 0}
    count = 0

    for shape_id, entry in data_store.items():
        if limit and count >= limit:
            break

        original_dsl = entry.get("dsl")
        if not original_dsl:
            continue

        # 1. Run Pipeline
        restored_dsl = process_shape(shape_id, data_store)

        if restored_dsl is None:
            results["failed"] += 1
            continue

        # 2. Geometric Verification
        # We compare the result of .expand() which returns the list of box dicts
        try:
            orig_boxes = original_dsl.expand()
            rest_boxes = restored_dsl.expand()

            # Sort boxes by center to ensure comparison is order-invariant
            key_func = lambda b: tuple(np.round(b["center"], 3))
            orig_boxes.sort(key=key_func)
            rest_boxes.sort(key=key_func)

            # Check if dimensions and counts match
            if len(orig_boxes) != len(rest_boxes):
                print(f"Mismatch [{shape_id}]: Box count {len(orig_boxes)} vs {len(rest_boxes)}")
                results["mismatched"] += 1
                continue

            # Compare centers and lengths with tolerance
            match = True
            for b1, b2 in zip(orig_boxes, rest_boxes):
                if not (np.allclose(b1["center"], b2["center"], atol=1e-3) and 
                        np.allclose(b1["lengths"], b2["lengths"], atol=1e-3)):
                    match = False
                    break

            if match:
                results["success"] += 1
            else:
                print(f"Mismatch [{shape_id}]: Geometric properties differ.")
                results["mismatched"] += 1

        except Exception as e:
            print(f"Verification crashed for {shape_id}: {e}")
            results["failed"] += 1

        count += 1

    print("\n" + "="*30)
    print(f"VERIFICATION REPORT")
    print(f"Total processed: {count}")
    print(f"Successfully Reconstructed: {results['success']}")
    print(f"Failed (Errors): {results['failed']}")
    print(f"Mismatched Geometry: {results['mismatched']}")
    print("="*30)

# --- Run Verification ---
# Set limit=None to run on the entire shapes_l0 dataset
verify_pipeline(shapes_l0, limit=10)

