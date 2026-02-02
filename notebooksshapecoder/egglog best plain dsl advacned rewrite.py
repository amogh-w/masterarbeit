#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


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


# In[3]:


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


# In[4]:


print(shapes_l0["2741_0_343"]["dsl"])


# In[5]:


shapes_l0["172_0_0"]


# In[6]:


shapes_l0["172_0_0"]["dsl"]


# In[7]:


print(shapes_l0["172_0_0"]["dsl"])


# In[ ]:





# # egglog schema

# In[8]:


from __future__ import annotations
from egglog import *

class Shape(Expr):
    @method(cost=5)
    def __add__(self, other: Shape) -> Shape: ...

@function(cost=1)
def cuboid(w: f64Like, h: f64Like, d: f64Like) -> Shape: ...

@function(cost=1)
def translate(s: Shape, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...

@function(cost=1)
def rotate(s: Shape, x: f64Like, y: f64Like, z: f64Like, w: f64Like) -> Shape: ...

@function(cost=1)
def sym_ref(s: Shape, axis: f64Like) -> Shape: ... 

@function(cost=1)
def sym_trans(s: Shape, axis: f64Like, count: f64Like, spacing: f64Like) -> Shape: ...


# # Bidirectional Conversion Functions

# In[17]:


def to_egglog(node) -> Shape:
    from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union, SymRef, SymTrans
    try:
        if isinstance(node, Cuboid):
            return cuboid(*node.size[:3])
        elif isinstance(node, Translate):
            return translate(to_egglog(node.child), *node.vector[:3])
        elif isinstance(node, Rotate):
            return rotate(to_egglog(node.child), *node.quaternion[:4])
        elif isinstance(node, Union):
            return to_egglog(node.left) + to_egglog(node.right)
        elif isinstance(node, SymRef):
            ax_id = {"AX": 0.0, "AY": 1.0, "AZ": 2.0}[node.axis]
            return sym_ref(to_egglog(node.child), ax_id)
        elif isinstance(node, SymTrans):
            ax_id = {"AX": 0.0, "AY": 1.0, "AZ": 2.0}[node.axis]
            return sym_trans(to_egglog(node.child), ax_id, node.count, node.spacing)
    except Exception as e:
        raise ValueError(f"Conversion to egglog failed: {e}")
    raise TypeError(f"Unsupported node type: {type(node)}")

import re

def from_egglog_string(egg_expr):
    from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union, SymRef, SymTrans
    expr_str = str(egg_expr).strip()

    # Define the union logic once
    def dsl_add(self, other):
        return Union(self, other)

    # Patch all classes so they support the '+' operator during eval
    for cls in [Cuboid, Translate, Rotate, Union, SymRef, SymTrans]:
        cls.__add__ = dsl_add

    namespace = {
        "cuboid": lambda w, h, d: Cuboid(size=[w, h, d]),
        "translate": lambda child, x, y, z: Translate(child, vector=[x, y, z]),
        "rotate": lambda child, x, y, z, w: Rotate(child, quaternion=[x, y, z, w]),
        "sym_ref": lambda child, ax: SymRef(child, {0.0:"AX", 1.0:"AY", 2.0:"AZ"}[ax]),
        "sym_trans": lambda child, ax, c, s: SymTrans(child, {0.0:"AX", 1.0:"AY", 2.0:"AZ"}[ax], c, s),
        # 'add' in egglog string might appear as a standalone function call for math
        "add": lambda a, b: a + b, 
        "inv": lambda a: -a,
    }

    try:
        # Extract variables (_f64_1 = ...)
        assignments = re.findall(r"^(_\w+)\s*=\s*(.*)$", expr_str, re.MULTILINE)

        # Extract the main expression body
        body_lines = []
        for line in expr_str.splitlines():
            if not re.match(r"^_\w+\s*=", line.strip()):
                body_lines.append(line)

        # Evaluate assignments to populate the namespace with intermediate variables
        for var_name, var_val in assignments:
            # We allow the namespace to reference itself
            namespace[var_name] = eval(var_val.strip(), {"__builtins__": {}}, namespace)

        final_body = "\n".join(body_lines).strip()

        # Final evaluation of the reconstructed DSL tree
        return eval(final_body, {"__builtins__": {}}, namespace)

    except Exception as e:
        raise RuntimeError(f"Failed to restore DSL. Error: {e}\nFull String:\n{expr_str}")


# # Optimization Rules

# In[18]:


sem_3dl_rules = ruleset()

@sem_3dl_rules.register
def _3dl_rules_definition(
    s: Shape, s2: Shape, s3: Shape,
    x: f64, y: f64, z: f64, w: f64,
    x1: f64, y1: f64, z1: f64, w1: f64,
    x2: f64, y2: f64, z2: f64, w2: f64,
    a: f64, count: f64, spacing: f64
):
    # --- 1. Symmetry Detection ---
    # Use native f64 negation. Egglog folds these constants automatically.
    yield rewrite(translate(s, x1, y1, z1) + translate(s, f64(-1.0) * x1, y1, z1)).to(
        sym_ref(translate(s, x1, y1, z1), 0.0)
    )
    yield rewrite(translate(s, x1, y1, z1) + translate(s, x1, f64(-1.0) * y1, z1)).to(
        sym_ref(translate(s, x1, y1, z1), 1.0)
    )
    yield rewrite(translate(s, x1, y1, z1) + translate(s, x1, y1, f64(-1.0) * z1)).to(
        sym_ref(translate(s, x1, y1, z1), 2.0)
    )

    # --- 2. Symmetry Flips (Normalization) ---
    yield rewrite(sym_ref(translate(s, x, y, z), 0.0)).to(sym_ref(translate(s, f64(-1.0) * x, y, z), 0.0))
    yield rewrite(sym_ref(translate(s, x, y, z), 1.0)).to(sym_ref(translate(s, x, f64(-1.0) * y, z), 1.0))
    yield rewrite(sym_ref(translate(s, x, y, z), 2.0)).to(sym_ref(translate(s, x, y, f64(-1.0) * z), 2.0))

    # --- 3. Translation & Move Simplification ---
    # Combine nested translations using native '+'
    yield rewrite(translate(translate(s, x1, y1, z1), x2, y2, z2)).to(
        translate(s, x1 + x2, y1 + y2, z1 + z2)
    )
    # Remove no-op moves
    yield rewrite(translate(s, 0.0, 0.0, 0.0)).to(s)
    # Factor out common translations
    yield rewrite(translate(s, x, y, z) + translate(s2, x, y, z)).to(translate(s + s2, x, y, z))

    # --- 4. Rotation Simplification ---
    yield rewrite(rotate(s, 0.0, 0.0, 0.0, 1.0)).to(s)
    yield rewrite(rotate(s, x, y, z, w) + rotate(s2, x, y, z, w)).to(rotate(s + s2, x, y, z, w))

    # --- 5. Union & Symmetry Properties ---
    yield rewrite(s + s2).to(s2 + s)
    yield rewrite((s + s2) + s3).to(s + (s2 + s3))
    yield rewrite(sym_ref(s + s2, a)).to(sym_ref(s, a) + sym_ref(s2, a))

    # wow
    # Add this to your ruleset definition in Cell 10
    yield rewrite(rotate(s, x, y, z, w) + rotate(s, x, y, z, f64(-1.0) * w)).to(
        sym_ref(rotate(s, x, y, z, w), 0.0) # Simplified logic for testing
    )


# # Execution

# In[19]:


def process_shape(shape_id: str, data_store: dict):
    egraph = EGraph()

    # We don't use egraph.register(sem_3dl_rules) here because it causes the TypeError.
    # Instead, we define which ruleset to use directly in the .run() call.

    entry = data_store.get(shape_id)
    if not entry or "dsl" not in entry:
        print(f"Error: ID {shape_id} not found.")
        return None

    try:
        # 1. Conversion: Use egraph.let to put the shape into the e-graph
        egg_shape = egraph.let("my_shape", to_egglog(entry["dsl"]))

        # 2. Optimization: Pass the ruleset into run()
        # We use saturate() to run the rules until no more changes occur.
        # This is where sem_3dl_rules is actually 'registered' and executed.
        egraph.run(sem_3dl_rules.saturate())

        # 3. Extraction: Get the best version based on our cost model
        extracted, cost = egraph.extract(egg_shape, include_cost=True)
        print(f"Extracted {shape_id} with cost: {cost}")

        # 4. Restoration: Convert back to your original Python DSL objects
        return from_egglog_string(extracted)

    except Exception as e:
        print(f"Pipeline failed for {shape_id}: {e}")
        # Helpful for debugging inside notebooks
        import traceback
        traceback.print_exc()
        return None

# --- Usage ---
restored_dsl = process_shape("2741_0_343", shapes_l0)


# In[20]:


print(restored_dsl)


# In[21]:


def validate_all_shapes(data_store, limit=None):
    results = []
    count = 0

    # Iterate through the dictionary (limit for testing)
    for shape_id, entry in data_store.items():
        if limit and count >= limit:
            break

        print(f"[{count+1}] Validating {shape_id}...", end="\r")

        # 1. Capture Original
        original_dsl = entry["dsl"]

        # 2. Process through Egglog
        optimized_dsl = process_shape(shape_id, data_store)

        if optimized_dsl:
            results.append({
                "id": shape_id,
                "status": "Success",
                "original": original_dsl,
                "optimized": optimized_dsl
            })
        else:
            results.append({
                "id": shape_id,
                "status": "Failed",
                "original": original_dsl,
                "optimized": None
            })
        count += 1

    return results

# Run on first 50 shapes to verify logic
validation_report = validate_all_shapes(shapes_l0, limit=3)

# Print a summary of differences
for res in validation_report:
    if res["status"] == "Success":
        # Check if egglog actually found a better version
        if str(res["original"]) != str(res["optimized"]):
            print(f"✅ Shape {res['id']} OPTIMIZED:")
            print(f"   FROM: {res['original']}")
            print(f"   TO:   {res['optimized']}\n")


# In[22]:


def print_all_variants(shape_id: str, data_store: dict, limit=10):
    egraph = EGraph()
    entry = data_store.get(shape_id)

    if not entry:
        print(f"ID {shape_id} not found.")
        return

    # 1. Convert and insert into E-Graph
    egg_shape = egraph.let("my_shape", to_egglog(entry["dsl"]))

    # 2. Run the optimization rules to generate variants
    egraph.run(sem_3dl_rules.saturate())

    # 3. Extract multiple variants
    # This finds distinct trees within the same E-Class
    variants = egraph.extract_multiple(egg_shape, limit)

    print(f"=== Found {len(variants)} variants for {shape_id} ===\n")

    for i, var in enumerate(variants):
        try:
            # Reconstruct into your DSL objects for pretty printing
            dsl_version = from_egglog_string(var)
            print(f"VARIANT #{i}:")
            print(dsl_version)
            print("-" * 30)
        except Exception as e:
            print(f"VARIANT #{i} (Raw Egglog):")
            print(var)
            print(f"(Restoration failed: {e})")
            print("-" * 30)

# Example Usage
print_all_variants("2741_0_343", shapes_l0, limit=3)


# In[27]:


def count_dsl_stats(node):
    """
    Recursively walks the restored Python DSL tree to count:
    1. Nodes: Every object instance (Cuboid, Translate, etc.)
    2. Params: Every numerical value used (sizes, vectors, angles)
    """
    node_count = 1
    param_count = 0

    if hasattr(node, "serialize"):
        # Your serialize returns: (Class, (params_list, children_list))
        _, (params, children) = node.serialize()

        # Count parameters (e.g., [w, h, d] is 3 params)
        if params:
            if isinstance(params, (list, tuple)):
                param_count += len(params)
            else:
                param_count += 1

        # Recurse into children nodes
        for child in children:
            if hasattr(child, "serialize"):
                c_nodes, c_params = count_dsl_stats(child)
                node_count += c_nodes
                param_count += c_params

    return node_count, param_count

def analyze_variants_to_dsl(shape_id, data_store, limit=5):
    egraph = EGraph()
    entry = data_store.get(shape_id)

    # 1. To Egglog
    egg_shape = egraph.let("my_shape", to_egglog(entry["dsl"]))

    # 2. Optimize
    egraph.run(sem_3dl_rules.saturate())

    # 3. Extract variants
    variants = egraph.extract_multiple(egg_shape, limit)

    print(f"=== DSL Metric Analysis for {shape_id} ===\n")
    print(f"{'Idx':<4} | {'Egg Cost':<10} | {'DSL Nodes':<10} | {'DSL Params':<10}")
    print("-" * 50)

    for i, var_expr in enumerate(variants):
        # We extract the specific cost of this variant
        _, cost = egraph.extract(var_expr, include_cost=True)

        try:
            # CONVERT BACK TO DSL OBJECTS
            restored_dsl = from_egglog_string(var_expr)
            print(restored_dsl)

            # CALCULATE METRICS ON PYTHON OBJECTS
            nodes, params = count_dsl_stats(restored_dsl)

            print(f"{i:<4} | {cost:<10} | {nodes:<10} | {params:<10}")
        except Exception as e:
            print(f"{i:<4} | {cost:<10} | ERROR: Restoration failed")

# Run it
analyze_variants_to_dsl("2741_0_343", shapes_l0, limit=2)


# In[ ]:




