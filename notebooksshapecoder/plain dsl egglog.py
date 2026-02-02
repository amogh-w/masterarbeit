#!/usr/bin/env python
# coding: utf-8

# In[17]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[18]:


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


# In[19]:


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


shapes_l0["172_0_0"]


# In[21]:


shapes_l0["172_0_0"]["dsl"]


# In[22]:


print(shapes_l0["172_0_0"]["dsl"])


# In[23]:


from __future__ import annotations
from egglog import *

class Shape(Expr):
    # Represents your Cuboid
    def __init__(self, w: f64Like, h: f64Like, d: f64Like, label: i64Like) -> None: ...

    # Represents your Translate
    def translate(self, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...

    # Represents your Rotate (quaternion)
    def rotate(self, x: f64Like, y: f64Like, z: f64Like, w: f64Like) -> Shape: ...

    # Represents your Union
    def __add__(self, other: Shape) -> Shape: ...

# We need to import your DSL classes to check types
from abstractionsshapecoder.dsl_nodes import Cuboid, Translate, Rotate, Union


# In[24]:


def to_egglog(node) -> Shape:
    if isinstance(node, Cuboid):
        return Shape(node.size[0], node.size[1], node.size[2], node.label)

    elif isinstance(node, Translate):
        # vector is [x, y, z]
        return to_egglog(node.child).translate(*node.vector)

    elif isinstance(node, Rotate):
        # quaternion is [x, y, z, w]
        return to_egglog(node.child).rotate(*node.quaternion)

    elif isinstance(node, Union):
        return to_egglog(node.left) + to_egglog(node.right)

    raise TypeError(f"Unknown DSL node type: {type(node)}")


# In[25]:


# Pick a shape from your loaded pickle
target_shape_obj = shapes_l0["172_0_0"]["dsl"]

# Initialize EGraph
egraph = EGraph()

# Convert and 'let' it in the EGraph
# This makes it the starting point for optimization
expr = egraph.let("my_shape", to_egglog(target_shape_obj))

# Now you can register rules and run them
# e.g., egraph.run(10)


# In[26]:


egraph.run(10)


# In[27]:


# A box of size [1, 2, 1]
part = Cuboid(size=[1, 2, 1])

# Apply a translation of zero
step1 = Translate(child=part, vector=[0, 0, 0])

# Now move it up by 10, then down by 10
step2 = Translate(child=step1, vector=[0, 10, 0])
final_shape = Translate(child=step2, vector=[0, -10, 0])


# In[28]:


final_shape


# In[29]:


# Two different boxes
leg1 = Cuboid(size=[1, 5, 1])
leg2 = Cuboid(size=[1, 5, 1])

# A union of the two legs
combined_legs = Union(left=leg1, right=leg2)

# Move both legs together to the right by 5
final_shape = Translate(child=combined_legs, vector=[5, 0, 0])


# In[30]:


from __future__ import annotations
from egglog import *

# ==========================================
# 1. DEFINE THE SCHEMA & CONVERTERS
# ==========================================
class Shape(Expr):
    def __init__(self, w: f64Like, h: f64Like, d: f64Like) -> None: ...
    def translate(self, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...
    def __add__(self, other: Shape) -> Shape: ... # Union

# Permanent fix for the "ConvertError" - allows using ints where f64 are expected
converter(int, f64, lambda i: f64(float(i)))
converter(i64, f64, f64.from_i64)

egraph = EGraph()

# ==========================================
# 2. DEFINE THE OPTIMIZATION RULES
# ==========================================
@egraph.register
def _rules(s: Shape, s1: Shape, s2: Shape, 
           x1: f64, y1: f64, z1: f64, 
           x2: f64, y2: f64, z2: f64):

    # Rule A: Fold successive translations T(T(s, v1), v2) -> T(s, v1 + v2)
    yield rewrite(s.translate(x1, y1, z1).translate(x2, y2, z2)).to(
        s.translate(x1 + x2, y1 + y2, z1 + z2)
    )

    # Rule B: Remove Identity Translation T(s, 0) -> s
    yield rewrite(s.translate(0.0, 0.0, 0.0)).to(s)

    # Rule C: Push Translate through Union T(s1 + s2, v) -> T(s1, v) + T(s2, v)
    yield rewrite((s1 + s2).translate(x1, y1, z1)).to(
        s1.translate(x1, y1, z1) + s2.translate(x1, y1, z1)
    )

# ==========================================
# 3. RUN EXPERIMENT 1: Identity & Redundancy
# ==========================================
print("--- Running Experiment 1 ---")
part = Shape(1, 2, 1)

# Program: Move 0, then move up 10, then move down 10
exp1_input = egraph.let("exp1_input", 
    part.translate(0, 0, 0)
        .translate(0, 10, 0)
        .translate(0, -10, 0)
)

egraph.run(10)

# Extraction finds the simplest version
exp1_output = egraph.extract(exp1_input)
print(f"Result: {exp1_output}") 
# Expected: Shape(1.0, 2.0, 1.0)

# ==========================================
# 4. RUN EXPERIMENT 2: Tree Transformation
# ==========================================
print("\n--- Running Experiment 2 ---")
leg1 = Shape(1, 5, 1)
leg2 = Shape(1, 5, 1)

# Input: One translation applied to a Union
exp2_input = egraph.let("exp2_input", (leg1 + leg2).translate(5, 0, 0))

egraph.run(10)

# Check if egglog considers the "Pushed" version equal to the input
pushed_version = leg1.translate(5, 0, 0) + leg2.translate(5, 0, 0)
egraph.check(exp2_input == pushed_version)

print("Check Successful: egglog treats T(A+B) and T(A)+T(B) as identical.")

# Extract the simplest form (egglog will pick the one with lower cost)
exp2_output = egraph.extract(exp2_input)
print(f"Extracted Version: {exp2_output}")


# In[31]:


# Program a "Chair" manually
leg_primitive = Shape(1, 5, 1)

leg_left = leg_primitive.translate(-2, 0, 0)
leg_right = leg_primitive.translate(2, 0, 0)

chair = egraph.let("chair", leg_left + leg_right)


# In[32]:


print("\n--- Running Experiment 3: De-duplication ---")
egraph = EGraph() # Fresh graph

# Define the chair
leg_primitive = Shape(1, 5, 1)
chair = egraph.let("chair", leg_primitive.translate(-2, 0, 0) + leg_primitive.translate(2, 0, 0))

# We haven't run any rules yet, but egglog already knows 
# that the 'leg_primitive' used in both places is the same E-Class.

# Let's extract multiple 'variants' of the chair
variants = egraph.extract_multiple(chair, 5)
print(f"Number of discovered variants: {len(variants)}")

# If we had a rule that said T(s, x) + T(s, -x) is a 'SymmetricPair(s, x)'
# egglog would find that abstraction for us automatically.


# In[33]:


# 1. Add SymmetricPair to our Shape sort
class Shape(Expr):
    def __init__(self, w: f64Like, h: f64Like, d: f64Like) -> None: ...
    def translate(self, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...
    def __add__(self, other: Shape) -> Shape: ...

    # NEW: Abstraction node
    @classmethod
    def symmetric_x(cls, child: Shape, offset: f64Like) -> Shape: ...

# Fix converters again for this cell
converter(int, f64, lambda i: f64(float(i)))
converter(i64, f64, f64.from_i64)

egraph = EGraph()

@egraph.register
def _abstraction_rules(s: Shape, x: f64):
    # Rule: T(s, x, 0, 0) + T(s, -x, 0, 0) => SymmetricPair(s, x)
    # We use 'birewrite' because they are mathematically the same
    yield birewrite(s.translate(x, 0, 0) + s.translate(-x, 0, 0)).to(
        Shape.symmetric_x(s, x)
    )

# 2. Define the Chair
leg = Shape(1, 5, 1)
chair = egraph.let("chair", leg.translate(2, 0, 0) + leg.translate(-2, 0, 0))

print("--- Running Experiment 4: Abstraction ---")
egraph.run(10)

# 3. Now let's see how many variants exist
variants = egraph.extract_multiple(chair, 10)
print(f"Number of discovered variants: {len(variants)}")

for i, v in enumerate(variants):
    print(f"Variant {i}: {v}")


# In[34]:


leg = Shape(1, 5, 1)

# Move right 10, then left 8 (Net move: 2)
leg_a = leg.translate(10, 0, 0).translate(-8, 0, 0)

# Move left 2
leg_b = leg.translate(-2, 0, 0)

complex_chair = egraph.let("complex_chair", leg_a + leg_b)


# In[35]:


print("\n--- Running Experiment 5: Combined Logic ---")
egraph = EGraph()

# Define all rules together
@egraph.register
def _combined_rules(s: Shape, x: f64, x1: f64, x2: f64):
    # Geometry Rules
    yield rewrite(s.translate(x1, 0, 0).translate(x2, 0, 0)).to(s.translate(x1 + x2, 0, 0))
    yield rewrite(s.translate(0.0, 0.0, 0.0)).to(s)

    # Abstraction Rule
    yield birewrite(s.translate(x, 0, 0) + s.translate(-x, 0, 0)).to(Shape.symmetric_x(s, x))

# Setup the "Messy" input
leg = Shape(1, 5, 1)
leg_a = leg.translate(10, 0, 0).translate(-8, 0, 0)
leg_b = leg.translate(-2, 0, 0)
complex_chair = egraph.let("complex_chair", leg_a + leg_b)

egraph.run(20)

# Extract the "Simplest" version
best_version = egraph.extract(complex_chair)
print(f"Messy Input simplified to: {best_version}")


# In[36]:


from scipy.spatial.transform import Rotation as R

# 1. Expand our Shape sort with Rotation logic
class Shape(Expr):
    def __init__(self, w: f64Like, h: f64Like, d: f64Like) -> None: ...
    def rotate(self, x: f64Like, y: f64Like, z: f64Like, w: f64Like) -> Shape: ...

# A helper function in the E-Graph to calculate combined rotations
@function
def combine_quat(x1: f64Like, y1: f64Like, z1: f64Like, w1: f64Like,
                 x2: f64Like, y2: f64Like, z2: f64Like, w2: f64Like) -> f64: ...
# Note: For a real implementation, you'd return a new object or 4 floats.
# For this learning experiment, let's focus on the pattern:

egraph = EGraph()

@egraph.register
def _rotation_rules(s: Shape, 
                    x1: f64, y1: f64, z1: f64, w1: f64,
                    x2: f64, y2: f64, z2: f64, w2: f64):

    # Pattern: Apply one rotation, then another.
    # We want to represent that these CAN be merged.
    yield rewrite(s.rotate(x1, y1, z1, w1).rotate(x2, y2, z2, w2)).to(
        # In a real setup, we'd use a function that computes the new quat
        # For now, let's visualize the "folding" logic
        s.rotate(x1 + x2, y1 + y2, z1 + z2, w1 + w2) # Simplified "math" for demo
    )

    # Identity Rotation: Rotating by [0,0,0,1] does nothing
    yield rewrite(s.rotate(0.0, 0.0, 0.0, 1.0)).to(s)

# --- RUNNING THE EXPERIMENT ---
print("--- Running Experiment 6: Rotation Folding ---")
# Create a cube
cube = Shape(1.0, 1.0, 1.0)

# Rotate 90 degrees, then -90 degrees (should cancel out)
# Using simplified components for the demo
messy_rotation = cube.rotate(0.0, 0.0, 0.0, 1.0).rotate(0.707, 0.0, 0.0, 0.707).rotate(-0.707, 0.0, 0.0, -0.707)

expr = egraph.let("rotated_cube", messy_rotation)
egraph.run(10)

best_version = egraph.extract(expr)
print(f"Result: {best_version}")


# In[37]:


from __future__ import annotations
from egglog import *
import numpy as np

# ==========================================
# 1. FINAL SCHEMA & COST MODEL
# ==========================================
class Shape(Expr):
    # Primitives
    def __init__(self, w: f64Like, h: f64Like, d: f64Like, label: i64Like) -> None: ...

    # Standard Ops (Higher cost to encourage abstraction)
    @method(cost=10)
    def translate(self, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...

    @method(cost=10)
    def rotate(self, x: f64Like, y: f64Like, z: f64Like, w: f64Like) -> Shape: ...

    @method(cost=20)
    def __add__(self, other: Shape) -> Shape: ...

    # Abstractions (Very low cost = The Goal)
    @method(cost=1)
    @classmethod
    def grid_x(cls, child: Shape, count: i64Like, spacing: f64Like) -> Shape: ...

    @method(cost=1)
    @classmethod
    def symmetric_x(cls, child: Shape, offset: f64Like) -> Shape: ...

# Setup automatic type conversion
converter(int, f64, lambda i: f64(float(i)))
converter(i64, f64, f64.from_i64)

egraph = EGraph()

# ==========================================
# 2. DISCOVERY & OPTIMIZATION RULES
# ==========================================
@egraph.register
def _rules(s: Shape, s1: Shape, s2: Shape, x: f64, x1: f64, x2: f64):
    # Flattening
    yield rewrite(s.translate(x1, 0, 0).translate(x2, 0, 0)).to(s.translate(x1 + x2, 0, 0))
    yield rewrite(s.translate(0, 0, 0)).to(s)

    # Abstraction Discovery
    # Pattern: Mirroring
    yield birewrite(s.translate(x, 0, 0) + s.translate(-x, 0, 0)).to(Shape.symmetric_x(s, x))

    # Pattern: 3-item Grid (e.g., three chair slats or legs)
    yield rewrite((s + s.translate(x, 0, 0)) + s.translate(x * 2, 0, 0)).to(Shape.grid_x(s, 3, x))

# ==========================================
# 3. BRIDGE TO YOUR DATA
# ==========================================
def to_egglog(node) -> Shape:
    if isinstance(node, Cuboid):
        return Shape(node.size[0], node.size[1], node.size[2], node.label)
    elif isinstance(node, Translate):
        return to_egglog(node.child).translate(node.vector[0], node.vector[1], node.vector[2])
    elif isinstance(node, Rotate):
        return to_egglog(node.child).rotate(*node.quaternion)
    elif isinstance(node, Union):
        return to_egglog(node.left) + to_egglog(node.right)

# ==========================================
# 4. EXECUTION ON REAL PICKLE DATA
# ==========================================
# Replace "172_0_0" with any valid ID from your shapes_l0
try:
    target_data = shapes_l0["172_0_0"]["dsl"]
    print(f"Loading shape from dataset...")

    input_expr = egraph.let("original_shape", to_egglog(target_data))

    print("Running Equality Saturation...")
    egraph.run(20)

    optimized = egraph.extract(input_expr)
    print("\n--- RESULTS ---")
    print(f"Original: (Check your shapes_l0['172_0_0']['dsl'])")
    print(f"Egglog Optimized: {optimized}")

except Exception as e:
    print(f"Error processing real data: {e}")


# In[38]:


from __future__ import annotations
from egglog import *

class Num(Expr):
    def __init__(self, name: StringLike) -> None: ...
    # We define symbolic math so egglog can still represent "n * spacing"
    def __mul__(self, other: Num) -> Num: ...

class Shape(Expr):
    def __init__(self, w: Num, h: Num, d: Num, label: i64Like) -> None: ...

    @method(cost=10)
    def translate(self, x: Num, y: Num, z: Num) -> Shape: ...

    @method(cost=20)
    def __add__(self, other: Shape) -> Shape: ...

    @method(cost=1)
    @classmethod
    def grid_x(cls, child: Shape, count: i64Like, spacing: Num) -> Shape: ...

egraph = EGraph()


# In[39]:


class SymbolicValueMap:
    def __init__(self, precision=2):
        self.map = {0.0: Num("z")} # Pre-map zero to 'z'
        self.precision = precision

    def get_sym(self, val):
        v = round(float(val), self.precision)
        if v not in self.map:
            # Create a new symbol like 'v1', 'v2', etc.
            self.map[v] = Num(f"v{len(self.map)}")
        return self.map[v]

sym_map = SymbolicValueMap()

def to_egglog_symbolic(node) -> Shape:
    if isinstance(node, Cuboid):
        return Shape(
            sym_map.get_sym(node.size[0]), 
            sym_map.get_sym(node.size[1]), 
            sym_map.get_sym(node.size[2]), 
            node.label
        )
    elif isinstance(node, Translate):
        return to_egglog_symbolic(node.child).translate(
            sym_map.get_sym(node.vector[0]),
            sym_map.get_sym(node.vector[1]),
            sym_map.get_sym(node.vector[2])
        )
    elif isinstance(node, Union):
        return to_egglog_symbolic(node.left) + to_egglog_symbolic(node.right)


# In[40]:


@egraph.register
def _structural_rules(s: Shape, x: Num, z: Num, n: i64):
    # Ensure 'z' is recognized as the zero-symbol we defined in the map
    z = Num("z")

    # Base Case: s + s.T(x) -> Grid(s, 2, x)
    # This works whenever two identical shapes are offset by the same symbol 'x'
    yield rewrite(
        s + s.translate(x, z, z)
    ).to(
        Shape.grid_x(s, 2, x)
    )

    # Inductive Case: 
    # Because we are symbolic, we can define that the next item 
    # is at some offset 'off'. A more advanced rule would check if off == n*x.
    yield rewrite(
        Shape.grid_x(s, n, x) + s.translate(Num("v_ANY"), z, z)
    ).to(
        Shape.grid_x(s, n + 1, x)
    )

@egraph.register
def _associative_rules(a: Shape, b: Shape, c: Shape):
    yield rewrite(a + b).to(b + a)
    yield rewrite((a + b) + c).to(a + (b + c))


# In[41]:


target_data = shapes_l0["172_0_0"]["dsl"]
# Use the symbolic bridge
input_expr = egraph.let("sym_shape", to_egglog_symbolic(target_data))

egraph.run(40)

result = egraph.extract(input_expr)
print("--- SYMBOLIC ABSTRACTED RESULT ---")
print(result)

# To see what the symbols actually represent:
print("\nSymbol Legend:")
for val, sym in sym_map.map.items():
    print(f"{sym} = {val}")


# In[42]:


from collections import Counter

# 1. Initialize a fresh Graph and Value Map
egraph = EGraph()
sym_map = SymbolicValueMap(precision=2) # Using the symbolic map from before
shape_expressions = {}

print(f"Converting {len(shapes_l0)} shapes to symbolic egglog expressions...")

# 2. Bridge EVERY shape into the same E-Graph
for shape_id, data in shapes_l0.items():
    try:
        dsl_obj = data["dsl"]
        # Convert to symbolic and store in the graph
        expr = egraph.let(f"shape_{shape_id}", to_egglog_symbolic(dsl_obj))
        shape_expressions[shape_id] = expr
    except Exception as e:
        continue

# 3. Run Optimization/Abstraction Rules
# This merges identical parts into the same e-classes
print("Running Equality Saturation to merge shared parts...")
egraph.run(20)

# 4. Extract the "Best" version of every shape
# While extracting, we will keep track of which sub-shapes appear most often
all_subshapes = []

print("Extracting and analyzing sub-structures...")
for shape_id, expr in shape_expressions.items():
    optimized = egraph.extract(expr)

    # We use a simple string representation to count unique parts
    # In a real research setup, you'd traverse the 'optimized' tree
    parts = str(optimized).split(" + ") 
    all_subshapes.extend(parts)

# 5. Show Results: The most frequent "Parts" in your dataset
common_library = Counter(all_subshapes).most_common(10)

print("\n--- DISCOVERED LIBRARY PARTS (TOP 10) ---")
for i, (part, count) in enumerate(common_library):
    print(f"Rank {i+1} (Used {count} times):")
    print(f"  {part}\n")


# In[ ]:


import random

# Pick 5 random IDs from your dataset
random_ids = random.sample(list(shapes_l0.keys()), 5)

for i, shape_id in enumerate(random_ids):
    print(f"--- SHAPE {i+1}: {shape_id} ---")

    # 1. Generate the Input Expression
    dsl_obj = shapes_l0[shape_id]["dsl"]
    input_expr = to_egglog_symbolic(dsl_obj)
    egraph.let(f"input_{shape_id}", input_expr)

    # 2. Run the Engine to saturate equalities
    egraph.run(20)

    # 3. Get the Canonical (Best) version
    canonical = egraph.extract(input_expr)

    print("CANONICAL REPRESENTATIVE:")
    print(canonical)
    print("-" * 30)


# In[46]:


from __future__ import annotations
from egglog import *

class Shape(Expr):
    # Defining a cube uses 4 params (w, h, d, label). Cost = 5
    @method(cost=5)
    def __init__(self, w: Num, h: Num, d: Num, label: i64Like) -> None: ...

    # Translating uses 3 params. Cost = 10 (to discourage nested moves)
    @method(cost=10)
    def translate(self, x: Num, y: Num, z: Num) -> Shape: ...

    # Unions are very "expensive" for description length. Cost = 20
    @method(cost=20)
    def __add__(self, other: Shape) -> Shape: ...

    # --- ABSTRACTIONS ---
    # A Grid defines MANY objects with only 2 params (count, spacing). 
    # We give it a Cost of 1 so Egglog LOVES picking this.
    @method(cost=1)
    @classmethod
    def grid_x(cls, child: Shape, count: i64Like, spacing: Num) -> Shape: ...

    @method(cost=1)
    @classmethod
    def symmetric_x(cls, child: Shape, offset: Num) -> Shape: ...


# In[47]:


import random

# Select IDs
sample_ids = random.sample(list(shapes_l0.keys()), 5)

for sid in sample_ids:
    egraph = EGraph() # Fresh graph for each to see isolated canonicals

    # 1. Convert to symbolic
    raw_dsl = shapes_l0[sid]["dsl"]
    input_expr = to_egglog_symbolic(raw_dsl)
    expr_name = egraph.let(f"shape_{sid}", input_expr)

    # 2. Register discovery rules (same as we defined before)
    @egraph.register
    def discovery_rules(s: Shape, x: Num, n: i64):
        z = Num("z")
        # Base Case Symmetry
        yield birewrite(s.translate(x, z, z) + s.translate(Num(f"-{x}"), z, z)).to(
            Shape.symmetric_x(s, x)
        )
        # Base Case Grid
        yield rewrite(s + s.translate(x, z, z)).to(Shape.grid_x(s, 2, x))

    # 3. Run and Extract
    egraph.run(30)
    canonical = egraph.extract(expr_name)

    print(f"\nID: {sid}")
    print(f"Canonical Representative (Cheapest Version):")
    print(canonical)
    print("-" * 50)


# In[48]:


def analyze_expression(expr):
    """
    Recursively counts total nodes and parameters in an egglog expression.
    """
    expr_str = str(expr)

    # 1. Parameter Count (Look for Num and i64 patterns)
    # We count every 'Num("v...")', 'Num("z")', and integer literal
    num_params = expr_str.count('Num(') + expr_str.count(', 0)') # '0' is the label param

    # 2. Node Count (Total building blocks)
    node_count = expr_str.count('Shape') + expr_str.count('translate') + expr_str.count(' + ')

    # 3. Calculate Cost based on your Blueprint
    # (Matching your @method(cost=...) decorators)
    cost = (expr_str.count('Shape') * 5 + 
            expr_str.count('translate') * 10 + 
            expr_str.count(' + ') * 20 +
            expr_str.count('grid_x') * 1 +
            expr_str.count('symmetric_x') * 1)

    return cost, num_params, node_count

# --- RUNNING ANALYSIS ON 5 SAMPLES ---
for sid in sample_ids:
    egraph = EGraph()
    input_expr = to_egglog_symbolic(shapes_l0[sid]["dsl"])
    expr_name = egraph.let(f"shape_{sid}", input_expr)

    # Register rules...
    egraph.run(30)

    # Extract Canonical
    canonical = egraph.extract(expr_name)

    # Analysis
    c_cost, c_params, c_nodes = analyze_expression(canonical)
    i_cost, i_params, i_nodes = analyze_expression(input_expr)

    print(f"\nID: {sid}")
    print(f"  [Original ] Cost: {i_cost:<4} | Params: {i_params:<3} | Nodes: {i_nodes}")
    print(f"  [Canonical] Cost: {c_cost:<4} | Params: {c_params:<3} | Nodes: {c_nodes}")
    print(f"  REDUCTION: {((i_cost - c_cost) / i_cost * 100):.1f}%")


# In[50]:


@egraph.register
def _aggressive_discovery(s: Shape, s1: Shape, s2: Shape, s3: Shape, 
                          x1: Num, x2: Num, z: Num):
    z = Num("z")

    # 1. FORCED ASSOCIATIVITY & COMMUTATIVITY
    # This ensures (A + (B + C)) is seen as ((A + B) + C)
    yield rewrite(s1 + (s2 + s3)).to((s1 + s2) + s3)
    yield rewrite((s1 + s2) + s3).to(s1 + (s2 + s3))
    yield rewrite(s1 + s2).to(s2 + s1)

    # 2. PATTERN: Symmetric Pair Discovery
    # Matches Shape.T(x) + Shape.T(-x)
    # Note: We use a symbolic '-x' concept or just match two offsets
    yield rewrite(s.translate(x1, z, z) + s.translate(x2, z, z)).to(
        Shape.symmetric_x(s, x1) # Simplified for symbolic matching
    )


# In[51]:


def get_metrics(expr):
    s = str(expr)
    # Count unique symbols v1, v2... and z
    import re
    params = len(set(re.findall(r'Num\(".*?"\)', s)))

    nodes = s.count('Shape') + s.count('translate') + s.count(' + ') + \
            s.count('grid_x') + s.count('symmetric_x')

    # Calculate weighted cost
    cost = (s.count('Shape') * 5 + 
            s.count('translate') * 10 + 
            s.count(' + ') * 20 +
            s.count('grid_x') * 1 + 
            s.count('symmetric_x') * 1)

    return cost, params, nodes

# --- FINAL DEMO ---
for sid in sample_ids:
    egraph = EGraph()
    # 1. Input
    raw_dsl = shapes_l0[sid]["dsl"]
    input_expr = to_egglog_symbolic(raw_dsl)
    input_name = egraph.let("input", input_expr)

    # 2. Run with Aggressive Discovery
    egraph.run(40)

    # 3. Extract
    canonical = egraph.extract(input_name)

    # 4. Report
    i_c, i_p, i_n = get_metrics(input_expr)
    c_c, c_p, c_n = get_metrics(canonical)

    print(f"\nID: {sid}")
    print(f"  [Original ] Cost: {i_c:<4} | Params: {i_p:<3} | Nodes: {i_n}")
    print(f"  [Canonical] Cost: {c_c:<4} | Params: {c_p:<3} | Nodes: {c_n}")

    # Check for the win
    if c_c < i_c:
        print(f"  🎉 SUCCESS: Reduced complexity by {i_c - c_c} points!")
    else:
        print(f"  - No structural abstraction found yet.")

