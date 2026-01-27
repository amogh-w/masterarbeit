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


# In[ ]:





# In[3]:


# Import your new tools
import pickle

from tqdm import tqdm
from abstractionsshapecoder.shape_parser import ShapeParser
from abstractionsshapecoder.dsl_utils import collect_singleton_and_pair_data

# --- Configuration ---
# In the new codebase, data comes from a single text file, not JSONs
INPUT_FILE = src_dir / "abstractionsshapecoder" / "prog_data" / "PN_chair.txt"
SAVED_DIR = src_dir / "abstractionsshapecoder" / "saved"
PICKLE_FILE = SAVED_DIR / "all_dsl_shapes.pkl"

# Set a limit for testing (set to None for all shapes)
CHAIR_LIMIT = None 

def load_dsl_dataset():
    # 1. Ensure saved directory exists
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    full_dsl_shapes = {}

    # 2. Check for cached Pickle
    if PICKLE_FILE.exists():
        print(f"[INFO] Loading cached dataset from {PICKLE_FILE}...")
        with open(PICKLE_FILE, "rb") as f:
            full_dsl_shapes = pickle.load(f)
        print(f"[SUCCESS] Loaded {len(full_dsl_shapes)} shapes from cache.")

    else:
        print(f"[INFO] Cache not found. Parsing from {INPUT_FILE}...")

        if not INPUT_FILE.exists():
            print(f"[ERROR] Input file not found: {INPUT_FILE}")
            return {}

        parser = ShapeParser()

        with open(INPUT_FILE, 'r') as f:
            lines = f.readlines()

        print(f"[INFO] Found {len(lines)} lines/programs.")

        # 3. Parse Text Lines -> DSL Objects
        for line in tqdm(lines, desc="Parsing Shapes"):
            line = line.strip()
            if not line: continue

            try:
                # Split "ID : ProgramString"
                shape_id, prog_text = line.split(':', 1)
                shape_id = shape_id.strip()

                # Parse into DSL Node Tree
                dsl_obj = parser.parse(prog_text)

                if dsl_obj:
                    # Calculate Params immediately (Singleton/Pair data)
                    # We pass [dsl_obj] as a list because the util expects a list
                    s_data, p_data = collect_singleton_and_pair_data([dsl_obj])

                    # Store in the standard structure for your pipeline
                    full_dsl_shapes[shape_id] = {
                        "dsl": dsl_obj,
                        "singleton_params": s_data,
                        "pair_params": p_data,
                        "program_text": prog_text.strip() # Optional: keep raw text
                    }

            except Exception as e:
                # print(f"Skipping line due to error: {e}")
                pass

        # 4. Save to Pickle
        print(f"[INFO] Saving {len(full_dsl_shapes)} processed shapes to pickle...")
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(full_dsl_shapes, f)
        print("[SUCCESS] Dataset saved.")

    # 5. Apply Limit (if requested)
    all_dsl_shapes = full_dsl_shapes
    if CHAIR_LIMIT is not None and len(full_dsl_shapes) > CHAIR_LIMIT:
        print(f"[INFO] Applying limit: {CHAIR_LIMIT}")
        # Slice dictionary
        all_dsl_shapes = dict(list(full_dsl_shapes.items())[:CHAIR_LIMIT])

    print(f"[RESULT] Working with {len(all_dsl_shapes)} shapes.")
    return all_dsl_shapes


# In[4]:


shapes = load_dsl_dataset()

# Verify one entry
if shapes:
    first_key = list(shapes.keys())[0]
    print(f"\nExample Entry ({first_key}):")
    print(shapes[first_key]["dsl"])


# In[5]:


len(shapes)


# In[6]:


## 5. Extract L1 Structures & Parameters (Common for both AE and PCA)

from collections import defaultdict
from abstractionsshapecoder.debug_utils import debug_success
from abstractionssymh.debug_utils import debug_info


debug_info("Building L1 detailed dictionaries for singletons and pairs...")
combined_singletons_detailed_L1 = defaultdict(list)
combined_pairs_detailed_L1 = defaultdict(list)

for filename, data in tqdm(shapes.items(), desc="Aggregating L1 Parameters"):
    # SINGLETON parameters
    for pattern_name, param_lists in data["singleton_params"].items():
        if "Box" in pattern_name: continue # Skip Box nodes
        for param_list in param_lists or []:
            combined_singletons_detailed_L1[pattern_name].append({
                'file': filename, 'params': param_list
            })
    # PAIR parameters
    for pattern_name, param_lists in data["pair_params"].items():
        if "Box" in pattern_name: continue # Skip Box nodes
        for param_list in param_lists or []:
            combined_pairs_detailed_L1[pattern_name].append({
                'file': filename, 'params': param_list
            })

debug_success(f"Aggregated all L1 parameters.")


# In[7]:


combined_singletons_detailed_L1


# In[8]:


## 6. Prepare L1 Training Data (Common for both AE and PCA)

debug_info("--- Preparing L1 data for model training ---")

training_singleton_params_L1 = {}
for pattern_name, records in combined_singletons_detailed_L1.items():
    if records:
        training_singleton_params_L1[pattern_name] = [rec['params'] for rec in records]

training_pair_params_L1 = {}
for pattern_name, records in combined_pairs_detailed_L1.items():
    if records:
        training_pair_params_L1[pattern_name] = [rec['params'] for rec in records]

debug_success(f"L1 Data flattened for training.")
print(f"Found {len(training_singleton_params_L1)} L1 singleton patterns to train.")
print(f"Found {len(training_pair_params_L1)} L1 pair patterns to train.")


# In[9]:


training_singleton_params_L1.keys()


# In[10]:


training_pair_params_L1.keys()


# In[12]:


## 7.   VAE Pipeline: Train/Load L1 Models

# --- Setup VAE Directories (Local to this cell) ---
import torch
from abstractionssymh.abstraction_utils import DEVICE, find_abstractions, make_safe_filename
from abstractionssymh.debug_utils import debug_error


saved_models_L1_VAE_dir = saved_directory / "models_L1_VAE"
saved_models_L1_VAE_dir.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
ABSTRACTION_METHOD_VAE = 'vae'
debug_info(f"--- STARTING VARIATIONAL AUTOENCODER (VAE) L1 PIPELINE ---")
print(f"Saving VAE models to: {saved_models_L1_VAE_dir}")

# Check for existing models
models_exist_L1_VAE = any(saved_models_L1_VAE_dir.glob('*.pth'))
singleton_models_L1_VAE = {}
pair_models_L1_VAE = {}

if models_exist_L1_VAE:
    debug_info(f"--- L1 VAE models found. Loading from {saved_models_L1_VAE_dir} ---")

    # Load L1 VAE Singleton Models
    for name in training_singleton_params_L1.keys():
        save_file = saved_models_L1_VAE_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            try:
                # Load the full model object
                model = torch.load(save_file, map_location=DEVICE, weights_only=False)
                model.eval()
                singleton_models_L1_VAE[name] = model
            except Exception as e:
                debug_error(f"Failed to load L1 VAE model '{name}': {e}")

    # Load L1 VAE Pair Models
    for name in training_pair_params_L1.keys():
        save_file = saved_models_L1_VAE_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            try:
                model = torch.load(save_file, map_location=DEVICE, weights_only=False)
                model.eval()
                pair_models_L1_VAE[name] = model
            except Exception as e:
                debug_error(f"Failed to load L1 VAE model '{name}': {e}")
else:
    debug_info(f"--- No L1 VAE models found. Starting training... ---")

    # Train Singletons
    singleton_models_L1_VAE = find_abstractions(
        training_singleton_params_L1, 
        method=ABSTRACTION_METHOD_VAE,
        structure_type="SINGLETONS_L1_VAE", 
        min_examples=50, 
        epochs=30, # VAEs often benefit from slightly longer training
        save_dir=saved_models_L1_VAE_dir,
        plot_error_distribution=True,
        error_threshold=0.1
    )

    # Train Pairs
    pair_models_L1_VAE = find_abstractions(
        training_pair_params_L1, 
        method=ABSTRACTION_METHOD_VAE,
        structure_type="PAIRS_L1_VAE", 
        min_examples=50, 
        epochs=30,
        save_dir=saved_models_L1_VAE_dir,
        plot_error_distribution=True,
        error_threshold=0.1
    )

    # Save L1 VAE Models
    for name, model in singleton_models_L1_VAE.items():
        torch.save(model, saved_models_L1_VAE_dir / make_safe_filename(name, suffix="pth"))
    for name, model in pair_models_L1_VAE.items():
        torch.save(model, saved_models_L1_VAE_dir / make_safe_filename(name, suffix="pth"))
    debug_success(f"Saved L1 VAE models to {saved_models_L1_VAE_dir}")

debug_success(f"--- L1 VAE Workflow complete. {len(singleton_models_L1_VAE)} singleton and {len(pair_models_L1_VAE)} pair models ready. ---")


# In[14]:


from abstractionssymh.abstraction_utils import integrate_abstractions


debug_info("--- Creating new L1-VAE Abstracted Dataset ---")

all_abstracted_shapes_L1_VAE = {}
pickle_file_L1_VAE = saved_directory / "all_abstracted_shapes_L1_VAE.pkl"

if pickle_file_L1_VAE.exists():
    with open(pickle_file_L1_VAE, "rb") as f:
        all_abstracted_shapes_L1_VAE = pickle.load(f)
    debug_success(f"Loaded {len(all_abstracted_shapes_L1_VAE)} L1-VAE abstracted shapes.")
else:
    for filename, data in tqdm(shapes.items(), desc="Integrating L1-VAE Abstractions"):

        # We integrate using the VAE models trained in Step 7
        abstracted_dsl = integrate_abstractions(
            data["dsl"],
            singleton_models_L1_VAE,
            pair_models_L1_VAE,
            # IMPORTANT: VAEs are probabilistic and often have higher MSE than standard AEs.
            # We relax the threshold from 0.02 to 0.08 (or 0.10) to allow abstractions to apply.
            error_threshold=0.5, 
            detailed_debug=False
        )

        # Collect parameters (These are now latent Mean vectors)
        l1_singletons, l1_pairs = collect_singleton_and_pair_data([abstracted_dsl])

        all_abstracted_shapes_L1_VAE[filename] = {
            "dsl": abstracted_dsl,
            "singleton_params": l1_singletons,
            "pair_params": l1_pairs,
            "original_dsl": data["dsl"]
        }

    with open(pickle_file_L1_VAE, "wb") as f:
        pickle.dump(all_abstracted_shapes_L1_VAE, f)
    debug_success(f"Created and saved {len(all_abstracted_shapes_L1_VAE)} L1-VAE shapes.")


# In[16]:


# In[14]:
## 14. Imports for Visualization

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from abstractionsshapecoder.dsl_utils import Abstraction

# Set a nice style for the plots
sns.set_theme(style="whitegrid", palette="muted")
print("Plotting libraries imported.")

# In[12]:
## 12. Helper Functions for Statistics (Updated)

debug_info("Defining statistics helper functions...")

def _get_children_aware(node):
    """Helper to get children from either an Abstraction or a DSL node."""
    if isinstance(node, Abstraction):
        return node.children
    elif hasattr(node, "serialize"):
        _, (_, children_from_serialize) = node.serialize()
        # Filter for node-like objects
        return [c for c in children_from_serialize if hasattr(c, "serialize") or isinstance(c, Abstraction)]
    return []

def count_nodes(node):
    """Recursively counts all nodes in a tree (Abstraction-aware)."""
    if not node: 
        return 0
    count = 1  # Count the node itself
    for child in _get_children_aware(node):
        count += count_nodes(child)
    return count

def get_depth(node):
    """Recursively finds the max depth of a tree (Abstraction-aware)."""
    if not node:
        return 0
    children = _get_children_aware(node)
    if not children:
        return 1  # Leaf node
    return 1 + max(get_depth(child) for child in children)

def serialize_structure(node):
    """Creates a canonical string 'fingerprint' of a tree's structure (Abstraction-aware)."""
    if not node:
        return ""

    # Get the node's name
    if isinstance(node, Abstraction):
        name = f"Abs({node.pattern_name})"
    elif hasattr(node, "serialize"):
        name = type(node).__name__
    else:
        return "Unknown" # e.g., a primitive parameter

    # Get children structures
    children = _get_children_aware(node)
    if not children:
        return name

    child_structs = [serialize_structure(c) for c in children]
    child_structs.sort()  # Sort to make representation canonical (e.g., Union(A,B) == Union(B,A))
    return f"{name}({','.join(child_structs)})"

def get_abstraction_types(node, unique_types):
    """
    Recursively finds all unique abstraction node types (e.g., 'Abs(Scale)')
    and adds them to the provided set.
    """
    if not node:
        return

    if isinstance(node, Abstraction):
        name = f"Abs({node.pattern_name})"
        unique_types.add(name)

    for child in _get_children_aware(node):
        get_abstraction_types(child, unique_types)

debug_success("Statistics helpers defined.")


# In[20]:


# In[13]:
## 13. Calculate & Display Detailed Comparative Statistics

from abstractionsshapecoder.dsl_utils import find_all_subtrees
from abstractionssymh.abstraction_utils import Abstraction
import numpy as np

def calculate_dataset_stats(dataset_dict, name):
    """
    Runs all statistics calculations for a given dataset and returns a stats dict.
    """
    stats = {
        "name": name,
        "total_shapes": 0,
        "node_counts": [],
        "depths": [],
        "abstraction_nodes_per_shape": [],
        "structure_counts": defaultdict(int),
        "unique_abstraction_types": set()
    }

    if not dataset_dict:
        print(f"\n---   Statistics for {name} ---")
        print("Dataset is empty. Cannot calculate stats.")
        return None

    stats["total_shapes"] = len(dataset_dict)

    for data in tqdm(dataset_dict.values(), desc=f"Analyzing {name}"):
        tree = data["dsl"]

        stats["node_counts"].append(count_nodes(tree))
        stats["depths"].append(get_depth(tree))
        stats["structure_counts"][serialize_structure(tree)] += 1

        # Abstraction-specific stats
        if name != "L0 (Original) Dataset":
             # A more robust way for L1+
             abs_nodes_count = sum(1 for n in find_all_subtrees(tree) if isinstance(n, Abstraction))
             stats["abstraction_nodes_per_shape"].append(abs_nodes_count)
             get_abstraction_types(tree, stats["unique_abstraction_types"])
        else:
            stats["abstraction_nodes_per_shape"].append(0)

    # --- Compile & Print Report ---
    print("\n" + "="*80)
    print(f"---   Statistics for {name} ---")
    print("="*80)

    # Calculate aggregate numbers
    node_counts_np = np.array(stats["node_counts"])
    depths_np = np.array(stats["depths"])
    abs_nodes_np = np.array(stats["abstraction_nodes_per_shape"])

    print(f"Total Shapes:                     {stats['total_shapes']}")
    print(f"Total Unique Structures:          {len(stats['structure_counts'])}")
    print(f"Total Abstraction Nodes (Sum):    {np.sum(abs_nodes_np)}")

    print("\n--- Nodes Per Shape ---")
    print(f"  Mean (± std):                 {np.mean(node_counts_np):.2f} (± {np.std(node_counts_np):.2f})")
    print(f"  Median:                       {np.median(node_counts_np):.0f}")
    print(f"  Min / Max:                    {np.min(node_counts_np)} / {np.max(node_counts_np)}")

    print("\n--- Depth Per Shape ---")
    print(f"  Mean (± std):                 {np.mean(depths_np):.2f} (± {np.std(depths_np):.2f})")
    print(f"  Median:                       {np.median(depths_np):.0f}")
    print(f"  Min / Max:                    {np.min(depths_np)} / {np.max(depths_np)}")

    print("\n--- Abstraction Nodes Per Shape ---")
    print(f"  Mean (± std):                 {np.mean(abs_nodes_np):.2f} (± {np.std(abs_nodes_np):.2f})")
    print(f"  Median:                       {np.median(abs_nodes_np):.0f}")
    print(f"  Min / Max:                    {np.min(abs_nodes_np)} / {np.max(abs_nodes_np)}")

    # --- NEW SECTION TO PRINT ABSTRACTION TYPES ---
    print(f"\nTotal Unique Abstraction Types:   {len(stats['unique_abstraction_types'])}")
    if stats['unique_abstraction_types']:
        print("--- Unique Abstraction Types Found ---")
        # Sort for consistent output
        sorted_types = sorted(list(stats['unique_abstraction_types']))
        for i, abs_type in enumerate(sorted_types):
            print(f"  {i+1}. {abs_type}")
    # --- END NEW SECTION ---

    print("\n--- Top 3 Most Common Structures ---")
    sorted_structures = sorted(stats['structure_counts'].items(), key=lambda item: item[1], reverse=True)
    for i, (structure, count) in enumerate(sorted_structures[:3]):
        percentage = (count / stats['total_shapes']) * 100
        print(f"{i+1}. (Count: {count}, {percentage:.1f}%)")
        print(f"   {structure[:150] + '...' if len(structure) > 150 else structure}\n")

    return stats

# --- Run Analysis on all three datasets ---
stats_L0 = calculate_dataset_stats(shapes, "L0 (Original) Dataset")
stats_L1 = calculate_dataset_stats(all_abstracted_shapes_L1_VAE, "L1 Abstracted Dataset")
# stats_L2 = calculate_dataset_stats(all_abstracted_shapes_L1, "L2 Abstracted Dataset")

# Filter out None in case a dataset was empty
all_stats = [s for s in [stats_L0, stats_L1] if s]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


debug_info("--- Extracting L2-AE Parameters ---")
combined_singletons_detailed_L2_AE = defaultdict(list)
combined_pairs_detailed_L2_AE = defaultdict(list)

for filename, data in tqdm(all_abstracted_shapes_L1_VAE.items(), desc="Aggregating L2-AE Params"):
    for p_name, p_lists in data["singleton_params"].items():
        if "Box" in p_name: continue
        for p_list in p_lists or []:
            combined_singletons_detailed_L2_AE[p_name].append({'params': p_list})
    for p_name, p_lists in data["pair_params"].items():
        if "Box" in p_name: continue
        for p_list in p_lists or []:
            combined_pairs_detailed_L2_AE[p_name].append({'params': p_list})

# Prepare L2-AE Training Data
training_singleton_params_L2_AE = {
    k: [r['params'] for r in v] for k, v in combined_singletons_detailed_L2_AE.items() if v
}
training_pair_params_L2_AE = {
    k: [r['params'] for r in v] for k, v in combined_pairs_detailed_L2_AE.items() if v
}

debug_success(f"Found {len(training_singleton_params_L2_AE)} L2-AE singleton and {len(training_pair_params_L2_AE)} L2-AE pair patterns.")


# In[32]:


## 10.   AUTOENCODER Pipeline: Train/Load L2-AE Models

debug_info("--- Starting L2-AE Abstraction Pipeline ---")
models_exist_L2_AE = any(saved_models_L2_AE_dir.glob('*.pth'))
singleton_models_L2_AE = {}
pair_models_L2_AE = {}

if models_exist_L2_AE:
    debug_info(f"--- L2 AE models found. Loading from {saved_models_L2_AE_dir} ---")
    for name in training_singleton_params_L2_AE.keys():
        save_file = saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            # --- FIX: Added weights_only=False ---
            model = torch.load(save_file, map_location=DEVICE, weights_only=False); model.eval()
            singleton_models_L2_AE[name] = model
    for name in training_pair_params_L2_AE.keys():
        save_file = saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            # --- FIX: Added weights_only=False ---
            model = torch.load(save_file, map_location=DEVICE, weights_only=False); model.eval()
            pair_models_L2_AE[name] = model
else:
    debug_info("--- No L2 AE models found. Starting training... ---")
    singleton_models_L2_AE = find_abstractions(
        training_singleton_params_L2_AE, method='ae', structure_type="SINGLETONS_L2_AE", min_examples=50, epochs=20, save_dir=saved_models_L2_AE_dir,
        plot_error_distribution=True
    )
    pair_models_L2_AE = find_abstractions(
        training_pair_params_L2_AE, method='ae', structure_type="PAIRS_L2_AE", min_examples=50, epochs=20, save_dir=saved_models_L2_AE_dir,
        plot_error_distribution=True
    )
    for name, model in singleton_models_L2_AE.items():
        torch.save(model, saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth"))
    for name, model in pair_models_L2_AE.items():
        torch.save(model, saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth"))
    debug_success(f"Saved L2 AE models to {saved_models_L2_AE_dir}")

debug_success(f"--- L2 AE Workflow complete. {len(singleton_models_L2_AE)} singleton and {len(pair_models_L2_AE)} pair models ready. ---")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## 2. Imports

import pickle
import random
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Project-specific imports
from abstractionsshapecoder.debug_utils import debug_info, debug_error, debug_success
from abstractionsshapecoder.data_loader import parse_json_to_dsl
from abstractionsshapecoder.plot_utils import plot_dsl_with_k3d, plot_dsl_grid
from abstractionsshapecoder.dsl_utils import collect_singleton_and_pair_data
from abstractionsshapecoder.abstraction_utils import (
    find_abstractions, 
    integrate_abstractions, 
    expand_l1_to_l0,
    expand_l2_to_l1,
    Abstraction,
    Autoencoder, 
    PCAModel,
    DEVICE,
    make_safe_filename
)
from abstractionssymh.abstraction_compare_utils import (
    get_point_cloud_from_dsl,
    calculate_chamfer_distance
)
from abstractionssymh.dsl_nodes import Box # Used for type checking

print(f"All libraries imported. Using device: {DEVICE}")


# In[4]:


## 3. Load L0 Chair Dataset

# Set a limit on the number of chairs to load for faster testing.
# Set to None to load all chairs.
CHAIR_LIMIT = None

pickle_file = saved_directory / "all_dsl_shapes.pkl"
all_dsl_shapes = {} # This will be our final, limited dictionary
full_dsl_shapes = {} # This will hold the complete dataset

if pickle_file.exists():
    debug_info(f"Loading L0 DSL shapes from pickle: {pickle_file}")
    with open(pickle_file, "rb") as f:
        full_dsl_shapes = pickle.load(f)
    debug_success(f"Loaded {len(full_dsl_shapes)} total shapes from pickle.")
else:
    debug_info(f"Pickle file not found. Generating new pickle from JSON files...")
    chair_directory = dataset_directory / "Chair"
    if not chair_directory.exists():
        debug_error(f"Chair dataset directory not found at: {chair_directory}")
        # Stop execution or handle error
    else:
        json_files = sorted(list(chair_directory.glob("*.json")))
        if not json_files:
            debug_error(f"No JSON files found in {chair_directory}")
            # Stop execution or handle error
        else:
            for json_file in tqdm(json_files, desc="Loading JSON files"):
                try:
                    json_content = json_file.read_text(encoding="utf-8")
                    dsl_obj = parse_json_to_dsl(json_content)
                    full_dsl_shapes[json_file.name] = {
                        "dsl": dsl_obj,
                        "singleton_params": {},
                        "pair_params": {},
                    }
                except Exception as e:
                    debug_error(f"Failed to load {json_file.name}: {e}")

            debug_info(f"Loaded {len(full_dsl_shapes)} DSL shapes.")
            debug_info("Collecting parameters for each shape...")
            for name, data in tqdm(full_dsl_shapes.items(), desc="Collecting parameters"):
                dsl_obj = data["dsl"]
                singletons, pairs = collect_singleton_and_pair_data([dsl_obj])
                data["singleton_params"] = singletons
                data["pair_params"] = pairs

            with open(pickle_file, "wb") as f:
                pickle.dump(full_dsl_shapes, f)
            debug_success(f"Saved all {len(full_dsl_shapes)} shapes to {pickle_file}")

# --- Apply CHAIR_LIMIT ---
if full_dsl_shapes:
    if CHAIR_LIMIT is not None and len(full_dsl_shapes) > CHAIR_LIMIT:
        debug_info(f"Limiting dataset to {CHAIR_LIMIT} chairs.")
        limited_items = list(full_dsl_shapes.items())[:CHAIR_LIMIT]
        all_dsl_shapes = dict(limited_items)
    else:
        debug_info(f"Using all {len(full_dsl_shapes)} loaded chairs.")
        all_dsl_shapes = full_dsl_shapes
    debug_success(f"Final L0 dataset size: {len(all_dsl_shapes)} shapes.")
else:
    debug_error("No L0 shapes were loaded or generated. Notebook cannot continue.")


# In[5]:


from collections import defaultdict
import pandas as pd
from tqdm.auto import tqdm
from abstractionssymh.dsl_nodes import (
    Box, Scale, Rotate, Translate, Union, 
    SymRef, SymRot, SymTrans
)

def get_structure_signature(node, include_labels=False):
    """
    Recursively generates a string signature of the tree structure,
    ignoring all continuous parameters.
    """
    # 1. Base Case: Box
    if isinstance(node, Box):
        if include_labels:
            return f"Box({node.label})"
        return "Box"

    # 2. Binary Case: Union
    # We sort the children signatures to ensure Union(A, B) == Union(B, A)
    if isinstance(node, Union):
        left_sig = get_structure_signature(node.left, include_labels)
        right_sig = get_structure_signature(node.right, include_labels)

        # Sort to canonicalize structure
        sigs = sorted([left_sig, right_sig])
        return f"Union({sigs[0]}, {sigs[1]})"

    # 3. Unary Cases (Transformations & Symmetries)
    node_type = type(node).__name__

    # Handle children
    # Most nodes have .child, but let's be robust
    if hasattr(node, 'child'):
        child_sig = get_structure_signature(node.child, include_labels)

        # Symmetry nodes often have discrete params (n_fold) that act like structure
        # We generally want to treat SymRot(4) different from SymRot(8)
        extra_info = ""
        if hasattr(node, 'n'): # SymRot, SymTrans
            extra_info = f",n={node.n}"

        return f"{node_type}{extra_info}({child_sig})"

    return "Unknown"


# In[6]:


## 24. Main Grouping Loop

debug_info("Starting Structural Analysis on DSL trees...")

structure_groups = defaultdict(list)
structure_stats = []

for chair_key, data in tqdm(all_dsl_shapes.items(), desc="Generating Signatures"):
    dsl_tree = data['dsl']

    # Generate signature
    signature = get_structure_signature(dsl_tree, include_labels=False)

    # Store
    structure_groups[signature].append(chair_key)

# --- Process Results into DataFrame ---

debug_info("Aggregating structural groups...")

for signature, chairs in structure_groups.items():
    # Calculate a rough "complexity" by counting open parenthesis
    complexity = signature.count('(')

    structure_stats.append({
        'signature': signature,
        'count': len(chairs),
        'complexity': complexity,
        'example_chair': chairs[0], # Pick one random example
        'all_chairs': chairs
    })

# Convert to DataFrame and Sort
df_structures = pd.DataFrame(structure_stats)
df_structures = df_structures.sort_values(by='count', ascending=False).reset_index(drop=True)

# --- Display Statistics ---
total_unique_structures = len(df_structures)
top_1_count = df_structures.iloc[0]['count']
top_10_sum = df_structures.head(10)['count'].sum()

print("\n" + "="*60)
print(f"STRUCTURAL GRAMMAR ANALYSIS")
print("="*60)
print(f"Total Shapes Analyzed:      {len(all_dsl_shapes)}")
print(f"Unique Structures Found:    {total_unique_structures}")
print(f"Most Common Structure:      {top_1_count} chairs share the exact same tree.")
print(f"Top 10 Structures Cover:    {top_10_sum} chairs ({(top_10_sum/len(all_dsl_shapes))*100:.1f}%)")
print("="*60 + "\n")

# Show Top 20 Most Common Structures
pd.set_option('display.max_colwidth', 100)
display(df_structures[['count', 'complexity', 'signature', 'example_chair']].head(20))


# In[7]:


df_structures


# In[8]:


## 25. Visualize the Top Structural Archetypes

top_n_to_visualize = 5

print(f"Visualizing the top {top_n_to_visualize} most common structural archetypes...")

archetype_dsls = []
archetype_names = []

for i in range(top_n_to_visualize):
    row = df_structures.iloc[i]
    chair_key = row['example_chair']
    count = row['count']

    # Load DSL
    dsl_obj = all_dsl_shapes[chair_key]['dsl']

    archetype_dsls.append(dsl_obj)
    archetype_names.append(f"Rank {i+1} (N={count})\n{chair_key}")

# Plot grid
plot_dsl_grid(
    archetype_dsls, 
    archetype_names, 
    grid_cols=5, 
    grid_title="Top Structural Archetypes (The 'Template' Chairs)"
)


# In[9]:


## 26. Finalize and Save Structure Buckets

import pickle

# This dictionary maps: Signature (str) -> List of Chair Keys (list[str])
structure_buckets = dict(structure_groups)

# Let's save this so we don't have to re-run the analysis every time
buckets_pickle_path = saved_directory / "structure_buckets.pkl"

with open(buckets_pickle_path, "wb") as f:
    pickle.dump(structure_buckets, f)

debug_success(f"Saved structure buckets to {buckets_pickle_path}")
print(f"Total Buckets: {len(structure_buckets)}")

# --- Helper function to get chairs by Rank ---
# Since signatures are long strings, it's easier to ask for "Rank 1" (most common)
def get_chairs_by_rank(rank_index):
    """
    Returns the list of chairs for the Nth most common structure.
    rank_index is 0-based (0 = Most Common).
    """
    if rank_index >= len(df_structures):
        print(f"Error: Rank {rank_index} out of bounds.")
        return [], None, 0

    row = df_structures.iloc[rank_index]
    sig = row['signature']
    chairs = row['all_chairs']
    count = row['count']

    return chairs, sig, count

# Example: Get details of the #1 most common bucket
top_chairs, top_sig, top_count = get_chairs_by_rank(0)

print(f"\nExample - Rank 0 Bucket:")
print(f"Signature: {top_sig[:100]}...") # Truncated for display
print(f"Contains {top_count} chairs.")
print(f"First 5 IDs: {top_chairs[:5]}")


# In[10]:


## 27. Visualize a Specific Bucket
import random

# Change this to see different groups!
# 0 = The most common group (943 chairs)
# 1 = The second most common group
TARGET_RANK = 3

chairs_in_bucket, signature, count = get_chairs_by_rank(TARGET_RANK)

print(f"Visualizing random sample from Rank {TARGET_RANK} (Count: {count})")
print(f"Structure Signature: {signature}")

# Pick 6 random chairs from this specific bucket
if len(chairs_in_bucket) > 6:
    sample_keys = random.sample(chairs_in_bucket, 6)
else:
    sample_keys = chairs_in_bucket

bucket_dsls = [all_dsl_shapes[k]['dsl'] for k in sample_keys]
bucket_names = [k for k in sample_keys]

plot_dsl_grid(
    bucket_dsls, 
    bucket_names, 
    grid_cols=3, 
    grid_title=f"Sample from Structure Rank {TARGET_RANK}"
)


# In[11]:


import matplotlib.pyplot as plt

# Get counts from the dataframe
counts = df_structures['count'].values

plt.figure(figsize=(10, 6))
plt.plot(counts, color='#4c72b0', linewidth=2)
plt.fill_between(range(len(counts)), counts, color='#4c72b0', alpha=0.3)

plt.title('Distribution of Structural Archetypes (The Long Tail)', fontsize=14)
plt.xlabel('Structure Rank (0 = Most Common)', fontsize=12)
plt.ylabel('Number of Chairs', fontsize=12)
plt.yscale('log') # Log scale helps see the tail better
plt.grid(True, linestyle='--', alpha=0.5)

# Add text for the "Head"
plt.axvline(x=10, color='r', linestyle='--')
plt.text(15, counts[0], 'Top 10 Buckets\n(42% of data)', color='r', verticalalignment='top')

plt.show()


# In[12]:


import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# 1. Setup the Output Area (where plots will appear)
output_area = widgets.Output()

# 2. Create the Slider
# max value is total_unique_structures - 1
rank_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(df_structures) - 1,
    step=1,
    description='Structure Rank:',
    continuous_update=False,  # Update only when mouse is released (prevents lag)
    layout=widgets.Layout(width='500px')
)

# 3. Create a Text Output for Stats
info_label = widgets.HTML(
    value=f"<b>Rank 0</b>: {df_structures.iloc[0]['count']} chairs<br>Signature: {df_structures.iloc[0]['signature'][:80]}..."
)

# 4. Define the Update Function
def on_rank_change(change):
    rank = change['new']

    # Get Data
    chairs, sig, count = get_chairs_by_rank(rank)

    # Update Label
    info_label.value = f"<b>Rank {rank}</b>: {count} chairs<br>Signature: {sig[:100]}..."

    # Update Plot
    with output_area:
        clear_output(wait=True)

        # Pick samples
        sample_size = min(5, len(chairs))
        sample_keys = random.sample(chairs, sample_size)

        dsl_objs = [all_dsl_shapes[k]['dsl'] for k in sample_keys]
        names = [k for k in sample_keys]

        # Plot using your existing utility
        # Note: We need to use matplotlib inline or similar for widgets to render nicely usually,
        # but since plot_dsl_grid creates a new figure, it should work fine.
        try:
            plot_dsl_grid(
                dsl_objs, 
                names, 
                grid_cols=min(sample_size, 5), 
                grid_title=f"Archetype Rank {rank} (Total N={count})"
            )
        except Exception as e:
            print(f"Error plotting rank {rank}: {e}")

# 5. Link Slider to Function
rank_slider.observe(on_rank_change, names='value')

# 6. Display Everything
display(widgets.VBox([rank_slider, info_label, output_area]))

# Trigger initial load
on_rank_change({'new': 0})


# In[13]:


## 32. Helper Function: Create Single-Rank Dataset

def create_dataset_for_rank(target_rank, all_shapes_dict, structure_df):
    """
    Creates a new dataset dictionary containing ONLY chairs that match
    the structural signature of the specified rank.

    Parameters
    ----------
    target_rank : int
        The 0-based rank index (e.g., 8).
    all_shapes_dict : dict
        The master dictionary (all_dsl_shapes).
    structure_df : pd.DataFrame
        The dataframe containing structure statistics (df_structures).

    Returns
    -------
    dict
        A dictionary subset of all_shapes_dict.
    """

    # 1. Validation
    if target_rank >= len(structure_df):
        print(f"[Error] Rank {target_rank} is out of bounds. Max Rank is {len(structure_df)-1}.")
        return {}

    # 2. Retrieve Structural Info
    row = structure_df.iloc[target_rank]
    signature = row['signature']
    chair_keys = row['all_chairs']
    count = row['count']

    print(f"--- Building Dataset for Rank {target_rank} ---")
    print(f"Structure Count: {count} chairs")
    print(f"Signature:       {signature[:80]}...") # Truncated for readability

    # 3. Build the Subset Dictionary
    rank_dataset = {}
    missing_count = 0

    for key in chair_keys:
        if key in all_shapes_dict:
            # Copy the reference to the data
            rank_dataset[key] = all_shapes_dict[key]
            # Tag it for convenience
            rank_dataset[key]['structure_rank'] = target_rank
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"[Warning] {missing_count} keys from structure analysis were missing in the master dict.")

    print(f"Success! Created dataset with {len(rank_dataset)} shapes.")
    return rank_dataset

# --- Usage Example: Rank 8 ---
rank_8_dataset = create_dataset_for_rank(8, all_dsl_shapes, df_structures)

# Verify by printing a random key
if rank_8_dataset:
    first_key = list(rank_8_dataset.keys())[0]
    print(f"\nExample Key: {first_key}")


# In[14]:


rank_8_dataset


# In[15]:


print(rank_8_dataset["Chair_1002.json"]["dsl"])


# In[16]:


from abstractionssymh.dsl_nodes import *
from abstractionssymh.plot_utils import plot_dsl_with_k3d

# 1. Reconstruct the object from your string
# (I formatted it into valid Python for you here)
rank_8_chair = Union(
    Union(
        Translate(child=Rotate(child=Scale(child=Box(label=1), lengths=[0.743, 0.198, 0.861]), quaternion=[0.0, 0.0, 0.0, 1.0]), center=[-0.018, -0.117, 0.114]),
        SymRef(
            child=Union(
                Translate(child=Rotate(child=Scale(child=Box(label=2), lengths=[0.074, 0.701, 0.074]), quaternion=[0.0, 0.0, 0.0, 1.0]), center=[-0.353, -0.500, 0.443]),
                Translate(child=Rotate(child=Scale(child=Box(label=2), lengths=[0.074, 0.701, 0.074]), quaternion=[0.0, -0.3826, -0.0, 0.9239]), center=[0.204, -0.500, -0.202])
            ),
            plane_normal=[1.0, 0.0, 0.0],
            point_on_plane=[-0.017, -0.500, 0.443]
        )
    ),
    Union(
        SymRef(
            child=Translate(child=Rotate(child=Scale(child=Box(label=0), lengths=[0.062, 0.319, 0.035]), quaternion=[0.0, 0.0, 0.0, 1.0]), center=[-0.210, -0.056, -0.245]),
            plane_normal=[1.0, -0.0, -0.0],
            point_on_plane=[-0.020, -0.056, -0.245]
        ),
        Union(
            Translate(child=Rotate(child=Scale(child=Box(label=0), lengths=[0.562, 0.643, 0.037]), quaternion=[0.0, 0.0, 0.0, 1.0]), center=[-0.014, 0.399, -0.205]),
            Translate(child=Rotate(child=Scale(child=Box(label=0), lengths=[0.676, 0.766, 0.106]), quaternion=[0.0, 0.0, 0.0, 1.0]), center=[-0.020, 0.399, -0.234])
        )
    )
)

# 2. Visualize with K3D to confirm the Backrest complexity
print("Visualizing Rank 8 Structure...")
plot_dsl_with_k3d(rank_8_chair)


# In[17]:


from abstractionssymh.dsl_nodes import *

def create_seat(scale, center):
    """
    Abstracts: Translate(Rotate(Scale(Box)))
    """
    # We hardcode rotation/label because they are constant for the seat
    return Translate(
        child=Rotate(
            child=Scale(child=Box(label=1), lengths=scale), 
            quaternion=[0,0,0,1]
        ), 
        center=center
    )

def create_4_legs(front_leg_scale, front_leg_pos, back_leg_scale, back_leg_pos, back_leg_rot):
    """
    Abstracts: SymRef(Union(FrontLeg, BackLeg))
    Returns a subtree representing 4 legs (2 source legs mirrored).
    """
    # 1. Build Left Side
    left_legs = Union(
        # Front Leg
        Translate(
            child=Rotate(
                child=Scale(child=Box(label=2), lengths=front_leg_scale),
                quaternion=[0,0,0,1]
            ),
            center=front_leg_pos
        ),
        # Back Leg (Note the custom rotation parameter)
        Translate(
            child=Rotate(
                child=Scale(child=Box(label=2), lengths=back_leg_scale),
                quaternion=back_leg_rot
            ),
            center=back_leg_pos
        )
    )

    # 2. Apply Symmetry to get Right Side
    return SymRef(
        child=left_legs,
        plane_normal=[1.0, 0.0, 0.0], # X-axis reflection
        point_on_plane=[-0.017, -0.5, 0.443] # Ideally, extract this too
    )

def create_rank8_backrest(side_scale, side_pos, panel_a_scale, panel_a_pos, panel_b_scale, panel_b_pos):
    """
    Abstracts: Union(SymRef(SideSupports), Union(PanelA, PanelB))
    This encapsulates the specific complexity of Rank 8 backrests.
    """
    # Part A: The Vertical Side Supports (Mirrored)
    side_supports = SymRef(
        child=Translate(
            child=Rotate(child=Scale(child=Box(label=0), lengths=side_scale), quaternion=[0,0,0,1]),
            center=side_pos
        ),
        plane_normal=[1.0, 0.0, 0.0],
        point_on_plane=[-0.02, -0.056, -0.245]
    )

    # Part B: The Central Panels (Stacked)
    center_panels = Union(
        Translate(
            child=Rotate(child=Scale(child=Box(label=0), lengths=panel_a_scale), quaternion=[0,0,0,1]),
            center=panel_a_pos
        ),
        Translate(
            child=Rotate(child=Scale(child=Box(label=0), lengths=panel_b_scale), quaternion=[0,0,0,1]),
            center=panel_b_pos
        )
    )

    return Union(side_supports, center_panels)


# In[18]:


def decompile_rank8_chair(dsl_root):
    """
    Extracts semantic parameters from a Rank 8 DSL tree based on known index positions.
    Returns a dictionary of parameters ready for the creator functions.
    """
    # 1. Flatten to get nodes in deterministic order
    all_nodes = []
    def recursive_flatten(node):
        if hasattr(node, 'child'):
            all_nodes.append(node)
            recursive_flatten(node.child)
        elif hasattr(node, 'left'): 
            all_nodes.append(node)
            recursive_flatten(node.left)
            recursive_flatten(node.right)
        else:
            all_nodes.append(node)
    recursive_flatten(dsl_root)

    # 2. Extract using known indices (validated in previous turn)
    # Nodes of interest are Scale (size), Translate (pos), Rotate (orientation)
    scale_nodes = [n for n in all_nodes if isinstance(n, Scale)]
    trans_nodes = [n for n in all_nodes if isinstance(n, Translate)]
    rot_nodes   = [n for n in all_nodes if isinstance(n, Rotate)]

    params = {
        # --- SEAT (Index 0) ---
        'seat_scale': scale_nodes[0].lengths,
        'seat_pos':   trans_nodes[0].center,

        # --- LEGS (Indices 1 & 2) ---
        'f_leg_scale': scale_nodes[1].lengths,
        'f_leg_pos':   trans_nodes[1].center,
        'b_leg_scale': scale_nodes[2].lengths,
        'b_leg_pos':   trans_nodes[2].center,
        'b_leg_rot':   rot_nodes[2].quaternion, # The angled back leg

        # --- BACKREST (Indices 3, 4, 5) ---
        'side_scale':    scale_nodes[3].lengths,
        'side_pos':      trans_nodes[3].center,
        'panel_a_scale': scale_nodes[4].lengths,
        'panel_a_pos':   trans_nodes[4].center,
        'panel_b_scale': scale_nodes[5].lengths,
        'panel_b_pos':   trans_nodes[5].center,
    }
    return params


# In[19]:


from abstractionssymh.plot_utils import plot_dsl_with_k3d

# 1. Get a raw Rank 8 chair
raw_chair = all_dsl_shapes[list(rank_8_dataset.keys())[0]]['dsl']

# 2. Decompile it (Get the "DNA")
params = decompile_rank8_chair(raw_chair)

print("Extracted Semantic Parameters:")
print(f"Seat Size: {params['seat_scale']}")
print(f"Back Leg Angle: {params['b_leg_rot']}")

# 3. Recompile using Abstractions
# Notice how readable this code is compared to the raw DSL!
my_seat = create_seat(params['seat_scale'], params['seat_pos'])

my_legs = create_4_legs(
    params['f_leg_scale'], params['f_leg_pos'],
    params['b_leg_scale'], params['b_leg_pos'], params['b_leg_rot']
)

my_back = create_rank8_backrest(
    params['side_scale'], params['side_pos'],
    params['panel_a_scale'], params['panel_a_pos'],
    params['panel_b_scale'], params['panel_b_pos']
)

# Combine them (Seat + Legs + Back)
# Note: Rank 8 structure was Union(Union(Seat, Legs), Back)
reconstructed_chair = Union(
    Union(my_seat, my_legs),
    my_back
)

# 4. Verify
print("\nVisualizing Reconstructed Chair...")
plot_dsl_with_k3d(reconstructed_chair)


# In[20]:


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 1. Initialize lists for each "Part"
seat_data = []
legs_data = []
back_data = []

print(f"Decompiling {len(rank_8_dataset)} Rank 8 chairs for training...")

for key, data in rank_8_dataset.items():
    # Use the decompiler we wrote in the previous step
    # (Ensure decompile_rank8_chair is defined in your notebook!)
    params = decompile_rank8_chair(data['dsl'])

    # --- SEAT VECTOR (6 dims) ---
    # [scale_x, scale_y, scale_z, pos_x, pos_y, pos_z]
    seat_vec = np.concatenate([params['seat_scale'], params['seat_pos']])
    seat_data.append(seat_vec)

    # --- LEGS VECTOR (19 dims) ---
    # Front Scale(3) + Pos(3) + Back Scale(3) + Pos(3) + Rot(4) + Plane(3)
    # Note: We add plane/point just in case, or keep it fixed if it never changes.
    # Let's stick to the variables that change:
    legs_vec = np.concatenate([
        params['f_leg_scale'], params['f_leg_pos'],
        params['b_leg_scale'], params['b_leg_pos'],
        params['b_leg_rot']
    ])
    legs_data.append(legs_vec)

    # --- BACKREST VECTOR (18 dims) ---
    # Side Scale(3)+Pos(3) + PanelA Scale(3)+Pos(3) + PanelB Scale(3)+Pos(3)
    back_vec = np.concatenate([
        params['side_scale'], params['side_pos'],
        params['panel_a_scale'], params['panel_a_pos'],
        params['panel_b_scale'], params['panel_b_pos']
    ])
    back_data.append(back_vec)

# Convert to Tensors
X_seat = torch.tensor(np.array(seat_data), dtype=torch.float32).to(DEVICE)
X_legs = torch.tensor(np.array(legs_data), dtype=torch.float32).to(DEVICE)
X_back = torch.tensor(np.array(back_data), dtype=torch.float32).to(DEVICE)

print(f"Training Data Ready:")
print(f" - Seat Tensor: {X_seat.shape} (Input Dim: 6)")
print(f" - Legs Tensor: {X_legs.shape} (Input Dim: 16)")
print(f" - Back Tensor: {X_back.shape} (Input Dim: 18)")


# In[21]:


from abstractionssymh.abstraction_utils import Autoencoder, train_autoencoder

# --- Configuration ---
LATENT_DIM_SEAT = 2   # Seat is simple, maybe just width/depth vary?
LATENT_DIM_LEGS = 4   # Legs have height, thickness, splay angle...
LATENT_DIM_BACK = 4   # Back has height, curvature, panel thickness...
EPOCHS = 50
BATCH_SIZE = 32

# --- Helper to Normalize Data ---
def normalize_and_loader(tensor):
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    std[std == 0] = 1.0 # Prevent divide by zero
    norm_tensor = (tensor - mean) / std
    dataset = TensorDataset(norm_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True), mean, std

# 1. Train Seat AE
print("\n--- Training Seat Autoencoder ---")
loader_seat, mean_seat, std_seat = normalize_and_loader(X_seat)
ae_seat = Autoencoder(input_dim=X_seat.shape[1], hidden_dim=LATENT_DIM_SEAT).to(DEVICE)
# Attach stats for later use
ae_seat.data_mean_ = mean_seat
ae_seat.data_std_ = std_seat
ae_seat = train_autoencoder(ae_seat, loader_seat, "Seat_AE", epochs=EPOCHS)

# 2. Train Legs AE
print("\n--- Training Legs Autoencoder ---")
loader_legs, mean_legs, std_legs = normalize_and_loader(X_legs)
ae_legs = Autoencoder(input_dim=X_legs.shape[1], hidden_dim=LATENT_DIM_LEGS).to(DEVICE)
ae_legs.data_mean_ = mean_legs
ae_legs.data_std_ = std_legs
ae_legs = train_autoencoder(ae_legs, loader_legs, "Legs_AE", epochs=EPOCHS)

# 3. Train Backrest AE
print("\n--- Training Backrest Autoencoder ---")
loader_back, mean_back, std_back = normalize_and_loader(X_back)
ae_back = Autoencoder(input_dim=X_back.shape[1], hidden_dim=LATENT_DIM_BACK).to(DEVICE)
ae_back.data_mean_ = mean_back
ae_back.data_std_ = std_back
ae_back = train_autoencoder(ae_back, loader_back, "Back_AE", epochs=EPOCHS)


# In[22]:


def generate_new_rank8_chair(seat_latent, legs_latent, back_latent):
    """
    Takes latent vectors, decodes them into semantic parameters, 
    and builds a DSL tree.
    """
    # 1. Decode Seat
    with torch.no_grad():
        z_seat = torch.tensor(seat_latent, dtype=torch.float32).to(DEVICE)
        recon_seat = ae_seat.decoder(z_seat)
        # Un-normalize
        seat_params = (recon_seat * ae_seat.data_std_) + ae_seat.data_mean_
        seat_params = seat_params.cpu().numpy()

    # 2. Decode Legs
    with torch.no_grad():
        z_legs = torch.tensor(legs_latent, dtype=torch.float32).to(DEVICE)
        recon_legs = ae_legs.decoder(z_legs)
        legs_params = (recon_legs * ae_legs.data_std_) + ae_legs.data_mean_
        legs_params = legs_params.cpu().numpy()

    # 3. Decode Back
    with torch.no_grad():
        z_back = torch.tensor(back_latent, dtype=torch.float32).to(DEVICE)
        recon_back = ae_back.decoder(z_back)
        back_params = (recon_back * ae_back.data_std_) + ae_back.data_mean_
        back_params = back_params.cpu().numpy()

    # 4. Reconstruct DSL using your Creator Functions
    # We have to slice the flat arrays back into [x,y,z] chunks

    my_seat = create_seat(
        scale=seat_params[0:3], 
        center=seat_params[3:6]
    )

    my_legs = create_4_legs(
        front_leg_scale=legs_params[0:3], front_leg_pos=legs_params[3:6],
        back_leg_scale=legs_params[6:9], back_leg_pos=legs_params[9:12], back_leg_rot=legs_params[12:16]
    )

    my_back = create_rank8_backrest(
        side_scale=back_params[0:3], side_pos=back_params[3:6],
        panel_a_scale=back_params[6:9], panel_a_pos=back_params[9:12],
        panel_b_scale=back_params[12:15], panel_b_pos=back_params[15:18]
    )

    return Union(Union(my_seat, my_legs), my_back)

# --- TEST: Generate a Random Chair ---
# Sample random latent vectors (standard normal distribution)
rand_seat = np.random.randn(LATENT_DIM_SEAT)
rand_legs = np.random.randn(LATENT_DIM_LEGS)
rand_back = np.random.randn(LATENT_DIM_BACK)

print("Generating chair from random latent codes...")
new_chair = generate_new_rank8_chair(rand_seat, rand_legs, rand_back)

plot_dsl_with_k3d(new_chair)


# In[23]:


import plotly.express as px
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from abstractionssymh.dsl_nodes import Scale, Box

# 1. Your Ground Truth Mapping
LABEL_COLORS = {
    0: 0xFF0000,  # red    (Backrest)
    1: 0x00FF00,  # green  (Seat)
    2: 0x0000FF,  # blue   (Leg)
    3: 0xFFFF00,  # yellow (Armrest)
    -1: 0x808080, # gray   (Unknown)
}

# Helper to get human names for the legend
LABEL_NAMES = {
    0: "Backrest",
    1: "Seat",
    2: "Leg",
    3: "Armrest",
    -1: "Unknown"
}

# Helper to convert integer color (0xFF0000) to CSS hex string ('#FF0000') for Plotly
def int_to_hex_str(c):
    return f"#{c:06x}"

print(f"Verifying Semantic Consistency for {len(rank_8_dataset)} Rank 8 chairs...")

plot_data = []
consistency_check = defaultdict(list)

for key, data in rank_8_dataset.items():
    dsl_root = data['dsl']

    # --- FLATTEN TREE ---
    all_nodes = []
    def recursive_flatten(node):
        if hasattr(node, 'child'):
            all_nodes.append(node); recursive_flatten(node.child)
        elif hasattr(node, 'left'): 
            all_nodes.append(node); recursive_flatten(node.left); recursive_flatten(node.right)
        elif hasattr(node, 'children'):
            all_nodes.append(node); 
            for c in node.children: recursive_flatten(c)
        else:
            all_nodes.append(node) # Leaf (Box)
    recursive_flatten(dsl_root)

    # Filter for Scale nodes (which define the Part Shape)
    scale_nodes = [n for n in all_nodes if isinstance(n, Scale)]

    for i, node in enumerate(scale_nodes):
        # --- DRILL DOWN TO FIND LABEL ---
        # Standard L0 structure: Scale -> Box
        # We check the immediate child.
        child = node.child
        label_id = -1

        if isinstance(child, Box):
            label_id = child.label

        # Map ID to Name and Color
        part_name = LABEL_NAMES.get(label_id, "Unknown")
        color_hex = int_to_hex_str(LABEL_COLORS.get(label_id, 0x808080))

        # Store for plotting
        sx, sy, sz = node.lengths
        plot_data.append({
            'Scale X': sx, 
            'Scale Y': sy, 
            'Scale Z': sz,
            'Part Label': part_name,
            'Color': color_hex,  # We store the exact color needed
            'Node Index': i,
            'Chair ID': key
        })

        # Store for consistency stats
        consistency_check[i].append(part_name)

# 2. Create DataFrame
df_verification = pd.DataFrame(plot_data)

print(f"Extracted {len(df_verification)} labeled parts.")

# 3. Plot
# We use 'Color' column directly for mapping to ensure it matches your dict
fig = px.scatter_3d(
    df_verification, 
    x='Scale X', y='Scale Y', z='Scale Z',
    color='Part Label',
    title='Ground Truth Labels vs. Scale Geometry (Rank 8)',
    hover_data=['Node Index', 'Chair ID'],
    opacity=0.6,
    # Explicitly map names to the hex strings we generated
    color_discrete_map={
        name: int_to_hex_str(LABEL_COLORS[lbl_id]) 
        for lbl_id, name in LABEL_NAMES.items()
    }
)

fig.update_traces(marker=dict(size=3))
fig.update_layout(
    scene=dict(
        xaxis_title='Scale X (Width)',
        yaxis_title='Scale Y (Height)',
        zaxis_title='Scale Z (Depth)',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)
fig.show()

# 4. --- THE CONSISTENCY REPORT ---
print("\n" + "="*60)
print("STRUCTURAL CONSISTENCY REPORT (Rank 8)")
print("="*60)
print(f"{'Tree Index':<12} | {'Most Common Label':<15} | {'Consistency'}")
print("-" * 60)

for idx in sorted(consistency_check.keys()):
    labels = consistency_check[idx]
    most_common = Counter(labels).most_common(1)[0]
    label_name = most_common[0]
    count = most_common[1]
    total = len(labels)
    percentage = (count / total) * 100

    print(f"Node #{idx:<5} | {label_name:<15} | {percentage:.1f}% ({count}/{total})")


# In[24]:


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 1. Initialize Training Buckets
# We group indices based on your visual clusters
# Index 0 = Seat
# Index 1, 2 = Legs (They are geometrically similar, so we group them to learn "Leg-ness")
# Index 3, 4, 5 = Backrest (We group them to learn "Backrest-ness")
data_buckets = {
    "Seat": [],
    "Legs": [],
    "Backrest": []
}

print(f"Sorting scale vectors for {len(rank_8_dataset)} chairs...")

for key, data in rank_8_dataset.items():
    dsl_root = data['dsl']

    # --- Flatten to get deterministic order ---
    all_nodes = []
    def recursive_flatten(node):
        if hasattr(node, 'child'):
            all_nodes.append(node); recursive_flatten(node.child)
        elif hasattr(node, 'left'): 
            all_nodes.append(node); recursive_flatten(node.left); recursive_flatten(node.right)
        elif hasattr(node, 'children'):
            all_nodes.append(node); 
            for c in node.children: recursive_flatten(c)
        else:
            all_nodes.append(node)
    recursive_flatten(dsl_root)

    # Filter for Scale nodes
    scale_nodes = [n for n in all_nodes if isinstance(n, Scale)]

    # --- Sort into Buckets ---
    for i, node in enumerate(scale_nodes):
        vec = node.lengths # [sx, sy, sz]

        if i == 0:
            data_buckets["Seat"].append(vec)
        elif i in [1, 2]:
            data_buckets["Legs"].append(vec)
        elif i in [3, 4, 5]:
            data_buckets["Backrest"].append(vec)

# 2. Convert to PyTorch Tensors
tensors = {}
for name, vec_list in data_buckets.items():
    # Convert to numpy first for speed, then tensor
    arr = np.array(vec_list, dtype=np.float32)
    t = torch.tensor(arr).to(DEVICE)
    tensors[name] = t
    print(f"Dataset '{name}': {t.shape[0]} samples (Shape: {t.shape[1]}D)")


# In[25]:


from abstractionssymh.abstraction_utils import Autoencoder, train_autoencoder

# Configuration
LATENT_DIM = 1  # Compress 3D scale -> 1D "Style Factor"
EPOCHS = 30
BATCH_SIZE = 64

trained_models = {}

# Helper to normalize
def make_loader(tensor):
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    std[std < 1e-5] = 1.0 # Safety for constant values
    norm = (tensor - mean) / std
    return DataLoader(TensorDataset(norm), batch_size=BATCH_SIZE, shuffle=True), mean, std

print("\n--- TRAINING SPECIALIST MODELS ---")

for part_name, tensor in tensors.items():
    print(f"Training {part_name} Model...")

    # 1. Prepare Data
    loader, mean, std = make_loader(tensor)

    # 2. Initialize Model
    model = Autoencoder(input_dim=3, hidden_dim=LATENT_DIM).to(DEVICE)
    model.data_mean_ = mean
    model.data_std_ = std

    # 3. Train
    # We suppress the plot for brevity, but you can enable it
    model = train_autoencoder(model, loader, f"{part_name}_Scale_AE", epochs=EPOCHS)

    trained_models[part_name] = model
    print(f"✅ {part_name} model trained.")


# In[26]:


def test_model_generation(part_name):
    model = trained_models[part_name]
    model.eval()

    print(f"\n--- Generating {part_name} Variations ---")
    print(f"{'Latent':<8} | {'Scale X':<8} {'Scale Y':<8} {'Scale Z':<8} | {'Interpretation'}")
    print("-" * 60)

    # Sample the latent space at -2, 0, +2 (Standard Deviations)
    test_points = [-2.0, 0.0, 2.0]

    for val in test_points:
        # 1. Create Latent Vector
        z = torch.tensor([val], dtype=torch.float32).to(DEVICE)

        # 2. Decode
        with torch.no_grad():
            recon_norm = model.decoder(z)
            # Un-normalize: (val * std) + mean
            recon_real = (recon_norm * model.data_std_) + model.data_mean_
            res = recon_real.cpu().numpy()

        # 3. Interpret
        # Simple heuristic: Y is usually height/length
        interp = "Average"
        if part_name == "Legs":
            interp = "Short/Thin" if res[1] < model.data_mean_[1] else "Tall/Thick"
        elif part_name == "Seat":
            interp = "Small" if res[0] < model.data_mean_[0] else "Large"

        print(f"{val:<8.1f} | {res[0]:<8.3f} {res[1]:<8.3f} {res[2]:<8.3f} | {interp}")

# Run Tests
test_model_generation("Legs")
test_model_generation("Seat")
test_model_generation("Backrest")


# In[27]:


import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# --- 1. The Matplotlib Renderer ---
def plot_chair_mpl(boxes):
    """
    Draws the chair boxes using standard Matplotlib.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rank 8 Semantic Colors
    # 1=Seat(Green), 2=Legs(Blue), 0=Back(Red)
    # Matplotlib expects normalized RGB (0-1)
    colors = {
        1: (0.2, 0.8, 0.2), # Green
        2: (0.2, 0.2, 0.8), # Blue
        0: (0.8, 0.2, 0.2)  # Red
    }

    for box in boxes:
        c = np.array(box["center"], dtype=float)
        l = np.asarray(box["lengths"], dtype=float).ravel()
        q = box["quaternion"]
        lbl = box.get("label_id", -1)

        # Get Corners
        rot = Rotation.from_quat(q).as_matrix()
        d1, d2, d3 = [col * length / 2 for col, length in zip(rot.T, l)]

        # 8 Corners
        corners = np.array([
            c-d1-d2-d3, c-d1+d2-d3, c+d1-d2-d3, c+d1+d2-d3,
            c-d1-d2+d3, c-d1+d2+d3, c+d1-d2+d3, c+d1+d2+d3
        ])

        # 6 Faces (indices of corners)
        faces_idx = [
            [corners[0], corners[1], corners[3], corners[2]], # Bottom
            [corners[4], corners[5], corners[7], corners[6]], # Top
            [corners[0], corners[1], corners[5], corners[4]], # Front
            [corners[2], corners[3], corners[7], corners[6]], # Back
            [corners[0], corners[2], corners[6], corners[4]], # Left
            [corners[1], corners[3], corners[7], corners[5]]  # Right
        ]

        # Create Mesh
        poly = Poly3DCollection(faces_idx, alpha=0.8)
        poly.set_facecolor(colors.get(lbl, 'gray'))
        poly.set_edgecolor('k') # Black edges make shape clearer
        poly.set_linewidth(0.5)
        ax.add_collection3d(poly)

    # Set fixed limits so the chair doesn't jump around
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Depth)") # In some coords Y is up, here Z is usually up
    ax.set_zlabel("Z (Height)")

    plt.tight_layout()
    plt.show()

# --- 2. Dashboard Logic ---

# The Output Widget acts as the canvas
plot_output = widgets.Output()

def update_view(change=None):
    s_val = seat_slider.value
    l_val = legs_slider.value
    b_val = back_slider.value

    with plot_output:
        # This clears the previous image so they don't stack
        clear_output(wait=True)

        try:
            # A. Generate
            # (Assuming generate_hybrid_chair is defined from previous steps)
            dsl = generate_hybrid_chair(s_val, l_val, b_val)
            boxes = dsl.expand()

            # B. Render
            plot_chair_mpl(boxes)

            print(f"Latents: Seat={s_val:.1f}, Legs={l_val:.1f}, Back={b_val:.1f}")

        except Exception as e:
            print(f"Error generating chair: {e}")

# --- 3. Controls ---
style = {'description_width': '100px'}
layout = {'width': '90%'}

seat_slider = widgets.FloatSlider(min=-3, max=3, step=0.2, value=0, description='Seat Style', style=style, layout=layout)
legs_slider = widgets.FloatSlider(min=-3, max=3, step=0.2, value=0, description='Leg Style', style=style, layout=layout)
back_slider = widgets.FloatSlider(min=-3, max=3, step=0.2, value=0, description='Back Style', style=style, layout=layout)

# Coloring handles to match the plot
seat_slider.style.handle_color = "#33CC33" # Green
legs_slider.style.handle_color = "#3333CC" # Blue
back_slider.style.handle_color = "#CC3333" # Red

# Link sliders
seat_slider.observe(update_view, names='value')
legs_slider.observe(update_view, names='value')
back_slider.observe(update_view, names='value')

# --- 4. Layout ---
controls = widgets.VBox([
    widgets.HTML("<h3><b>Generative Chair Dashboard</b></h3>"),
    widgets.Label("Use sliders to explore semantic latent spaces:"),
    back_slider,
    seat_slider,
    legs_slider
])

app = widgets.HBox([controls, plot_output])

display(app)

# Initial Draw
update_view()


# In[28]:


# 1. Retrain Legs AE with 3 Latent Dimensions
# We hope the AE separates the features (e.g. Height vs Thickness)
LATENT_DIM_LEGS_NEW = 3 

print(f"Retraining Legs Autoencoder with {LATENT_DIM_LEGS_NEW} dimensions...")

# Get data (assuming 'loader_legs' is still available from previous steps)
# If not, re-run the 'Prepare Training Data' cell
if 'loader_legs' in locals():
    ae_legs = Autoencoder(input_dim=X_legs.shape[1], hidden_dim=LATENT_DIM_LEGS_NEW).to(DEVICE)
    ae_legs.data_mean_ = mean_legs
    ae_legs.data_std_ = std_legs

    # Train slightly longer to ensure it learns the separation
    ae_legs = train_autoencoder(ae_legs, loader_legs, "Legs_AE_3D", epochs=100)

    # Update the global dictionary
    trained_models["Legs"] = ae_legs
    print("✅ Legs model updated to 3D latent space.")
else:
    print("Error: 'loader_legs' not found. Please re-run the training preparation step.")


# In[29]:


import k3d
import numpy as np
from abstractionssymh.dsl_nodes import Scale

def visualize_rank_indices(rank_idx):
    # Get a representative chair
    chairs, _, _ = get_chairs_by_rank(rank_idx)
    dsl_root = all_dsl_shapes[chairs[0]]['dsl']

    # Flatten
    all_nodes = []
    def recursive_flatten(node):
        if hasattr(node, 'child'):
            all_nodes.append(node); recursive_flatten(node.child)
        elif hasattr(node, 'left'): 
            all_nodes.append(node); recursive_flatten(node.left); recursive_flatten(node.right)
        elif hasattr(node, 'children'):
            all_nodes.append(node); 
            for c in node.children: recursive_flatten(c)
        else:
            all_nodes.append(node)
    recursive_flatten(dsl_root)

    # Filter for Scale nodes
    scale_nodes = [n for n in all_nodes if isinstance(n, Scale)]

    plot = k3d.plot(name=f"Rank {rank_idx} Index Map", height=400)
    colors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0x00FFFF, 0xFF00FF] 

    for idx, node in enumerate(scale_nodes):
        boxes = node.expand()
        color = colors[idx % len(colors)]
        for box in boxes:
            center = np.array(box["center"])
            # Plot Label
            plot += k3d.text(f"{idx}", position=center, color=0x000000, size=1.5, label_box=True)
            # Plot Dot
            plot += k3d.points([center], color=color, point_size=0.15)

    print(f"--- Rank {rank_idx} Index Map ---")
    plot.display()

# Run for Top 5 Ranks
for r in [0, 1, 2, 3, 4]:
    visualize_rank_indices(r)


# In[30]:


# --- USER CONFIGURATION ---
# Map: Rank_ID -> List of Node Indices that are LEGS
# Look at the plots above to fill this in!
LEG_INDICES = {
    0: [1, 2],       # We confirmed Rank 0 has legs at 1 and 2
    1: [1, 2],       # [GUESS] Likely similar to Rank 0? Check plot!
    2: [1, 2],       # [GUESS] Check plot!
    3: [1, 2],       # [GUESS] Check plot!
    4: [1, 2]        # [GUESS] Check plot!
}
# --------------------------

import pandas as pd

universal_legs_data = []

print("Building Universal Leg Dataset...")

for rank, leg_idxs in LEG_INDICES.items():
    # Get all chairs for this rank
    chairs, _, count = get_chairs_by_rank(rank)

    print(f"Processing Rank {rank} ({count} chairs)...")

    for chair_key in chairs:
        dsl = all_dsl_shapes[chair_key]['dsl']

        # Flatten Tree (Same logic as visualizer)
        all_nodes = []
        def recursive_flatten(node):
            if hasattr(node, 'child'):
                all_nodes.append(node); recursive_flatten(node.child)
            elif hasattr(node, 'left'): 
                all_nodes.append(node); recursive_flatten(node.left); recursive_flatten(node.right)
            elif hasattr(node, 'children'):
                all_nodes.append(node); 
                for c in node.children: recursive_flatten(c)
            else:
                all_nodes.append(node)
        recursive_flatten(dsl)

        # Extract Scale Nodes
        scale_nodes = [n for n in all_nodes if isinstance(n, Scale)]

        # Grab only the specific leg nodes
        for idx in leg_idxs:
            if idx < len(scale_nodes):
                vec = scale_nodes[idx].lengths
                universal_legs_data.append({
                    'SX': vec[0],
                    'SY': vec[1],
                    'SZ': vec[2],
                    'Source Rank': f"Rank {rank}",
                    'Chair ID': chair_key
                })

df_universal_legs = pd.DataFrame(universal_legs_data)
print(f"\nSuccess! Extracted {len(df_universal_legs)} leg samples.")


# In[31]:


import plotly.express as px

fig = px.scatter_3d(
    df_universal_legs, 
    x='SX', y='SY', z='SZ',
    color='Source Rank',
    title='The Universal Leg Cluster (Top 5 Ranks)',
    opacity=0.5,
    hover_data=['Chair ID']
)

fig.update_traces(marker=dict(size=3))
fig.update_layout(
    scene=dict(
        xaxis_title='Thickness X',
        yaxis_title='Height Y',
        zaxis_title='Thickness Z',
        aspectmode='data'
    )
)
fig.show()


# In[32]:


from sklearn.mixture import GaussianMixture
import plotly.express as px
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from abstractionssymh.abstraction_utils import Autoencoder, train_autoencoder

# 1. Prepare Data
# We use the DataFrame 'df_universal_legs' created in the previous step
X_legs_all = df_universal_legs[['SX', 'SY', 'SZ']].values

# 2. Determine Optimal Clusters (Quick Elbow Check)
# We assume there are likely 2 or 3 main leg types.
n_components = 3 
gmm = GaussianMixture(n_components=n_components, random_state=42)
labels = gmm.fit_predict(X_legs_all)

# 3. Visualize the Clusters
df_universal_legs['Leg Type'] = labels.astype(str)

print(f"Clustered {len(df_universal_legs)} legs into {n_components} types.")

fig = px.scatter_3d(
    df_universal_legs, 
    x='SX', y='SY', z='SZ',
    color='Leg Type', # Color by the NEW cluster ID, not the Rank
    symbol='Source Rank', # Use symbol to see which Ranks fall into which Type
    title='Discovered Leg Sub-Types (Universal)',
    opacity=0.6
)
fig.update_traces(marker=dict(size=3))
fig.show()


# In[33]:


leg_models = {}

print("\n--- Training Leg Specialist Models ---")

for cluster_id in range(n_components):
    # 1. Filter Data for this Cluster
    cluster_data = df_universal_legs[df_universal_legs['Leg Type'] == str(cluster_id)]
    vectors = cluster_data[['SX', 'SY', 'SZ']].values

    print(f"Training Model for Leg Type {cluster_id} ({len(vectors)} samples)...")

    # 2. Convert to Tensor
    tensor = torch.tensor(vectors, dtype=torch.float32).to(DEVICE)

    # 3. Create Loader
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    std[std < 1e-5] = 1.0
    norm_tensor = (tensor - mean) / std
    loader = DataLoader(TensorDataset(norm_tensor), batch_size=64, shuffle=True)

    # 4. Train AE (1D Latent is usually enough for simple shapes)
    ae = Autoencoder(input_dim=3, hidden_dim=1).to(DEVICE)
    ae.data_mean_ = mean
    ae.data_std_ = std

    ae = train_autoencoder(ae, loader, f"LegAE_Type{cluster_id}", epochs=30)

    # Store
    leg_models[cluster_id] = ae

print("All leg models trained.")


# In[34]:


def generate_smart_leg(leg_type_id, style_val):
    """
    Generates a leg scale vector.
    Args:
        leg_type_id (int): 0, 1, or 2 (The GMM Cluster ID)
        style_val (float): The latent slider value (-3 to +3)
    """
    if leg_type_id not in leg_models:
        print(f"Error: Unknown Leg Type {leg_type_id}")
        return [0.05, 0.4, 0.05] # Default fallback

    model = leg_models[leg_type_id]
    model.eval()

    z = torch.tensor([[style_val]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        recon = model.decoder(z)
        real = (recon * model.data_std_) + model.data_mean_

    return real.cpu().numpy()[0]

# --- TEST IT ---
print("\n--- Testing Leg Generators ---")
for i in range(n_components):
    # Generate a "Average" leg (Latent 0) for each type
    leg_vec = generate_smart_leg(i, 0.0)
    print(f"Type {i} Average: {leg_vec}")


# In[35]:


# 1. Add the cluster labels back to the dataframe so we can query by Type
df_universal_legs['Leg Type'] = labels.astype(str)

# 2. Find example chairs for each Leg Type
for type_id in range(n_components):
    # Get rows for this type
    subset = df_universal_legs[df_universal_legs['Leg Type'] == str(type_id)]

    if not subset.empty:
        # Pick a random example
        sample_row = subset.iloc[0] 
        chair_key = sample_row['Chair ID']
        dims = [sample_row['SX'], sample_row['SY'], sample_row['SZ']]

        print(f"\n--- Leg Type {type_id} Example ---")
        print(f"Chair: {chair_key}")
        print(f"Leg Scale: {dims}")

        # Visualize
        # We plot the whole chair to see the leg in context
        dsl = all_dsl_shapes[chair_key]['dsl']
        plot_dsl_with_k3d(dsl)


# In[36]:


def create_universal_leg(leg_type_idx, style_val, position):
    """
    Creates a leg based on the AI-discovered types.
    """
    # 1. Generate the Scale Vector using your trained AEs
    scale_vec = generate_smart_leg(leg_type_idx, style_val)

    # 2. Create the Geometry
    # Note: Type 1 (Block Leg) might need a different rotation if it's a panel!
    # For now, we assume standard vertical orientation.
    return Translate(
        child=Rotate(
            child=Scale(child=Box(label=2), lengths=scale_vec),
            quaternion=[0,0,0,1] # Default upright
        ),
        center=position
    )


# In[37]:


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from abstractionssymh.abstraction_utils import Autoencoder, train_autoencoder, DEVICE

# 1. Prepare Data (Assuming df_universal_legs exists from previous step)
# If not, we quickly rebuild the array from global extraction for robustness
if 'df_universal_legs' not in locals():
    print("Universal leg data not found. Please run 'Step 2' of the previous turn first.")
else:
    X_legs_all = df_universal_legs[['SX', 'SY', 'SZ']].values

    # 2. Cluster into 3 Types (Standard, Chunky, Spindle)
    n_leg_types = 3
    gmm_legs = GaussianMixture(n_components=n_leg_types, random_state=42)
    leg_labels = gmm_legs.fit_predict(X_legs_all)

    print(f"Clustered legs into {n_leg_types} geometric types.")

    # 3. Train 3 Specialist Autoencoders (2 Sliders each)
    leg_models = {}
    LATENT_DIM_LEGS = 2 # We give you 2 sliders per leg type

    print("\n--- Training Leg Models ---")
    for i in range(n_leg_types):
        # Filter data
        vecs = X_legs_all[leg_labels == i]
        tensor = torch.tensor(vecs, dtype=torch.float32).to(DEVICE)

        # Normalize
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0)
        std[std < 1e-5] = 1.0
        norm_tensor = (tensor - mean) / std

        loader = DataLoader(TensorDataset(norm_tensor), batch_size=64, shuffle=True)

        # Train
        ae = Autoencoder(input_dim=3, hidden_dim=LATENT_DIM_LEGS).to(DEVICE)
        ae.data_mean_ = mean
        ae.data_std_ = std

        # We use a shorter epoch count for speed, sufficient for simple vectors
        ae = train_autoencoder(ae, loader, f"LegType_{i}", epochs=30)
        leg_models[i] = ae

    print("✅ Leg Library Ready.")


# In[38]:


import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

# --- 1. Leg Plotter (Matplotlib) ---
def plot_single_leg(scale_vec):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a single box at origin
    center = np.array([0, 0, 0])
    l = scale_vec

    # Simple Box Geometry
    # We don't rotate it so you can see the raw dimensions aligned to axes
    d = l / 2
    corners = np.array([
        center-d, center+d*[-1,1,-1], center+d*[1,1,-1], center+d*[1,-1,-1],
        center+d*[-1,-1,1], center+d*[-1,1,1], center+d, center+d*[1,-1,1]
    ])

    faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]

    poly = Poly3DCollection(faces, alpha=0.8)
    poly.set_facecolor('#3333CC') # Blue
    poly.set_edgecolor('k')
    ax.add_collection3d(poly)

    # Fixed Limits to see growth clearly
    limit = 0.8
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.set_xlabel(f"Width X ({l[0]:.2f})")
    ax.set_ylabel(f"Height Y ({l[1]:.2f})") # In chairs, Y is usually Up
    ax.set_zlabel(f"Depth Z ({l[2]:.2f})")

    plt.tight_layout()
    plt.show()

# --- 2. Generator Logic ---
def generate_leg_mesh(type_id, s1, s2):
    if type_id not in leg_models: return [0.1, 0.5, 0.1]

    model = leg_models[type_id]
    model.eval()

    # 2D Latent Vector
    z = torch.tensor([[s1, s2]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        recon = model.decoder(z)
        real = (recon * model.data_std_) + model.data_mean_

    return real.cpu().numpy()[0]

# --- 3. Dashboard UI ---
out_leg = widgets.Output()

def update_leg_view(change=None):
    t_id = dd_type.value
    v1 = slider_1.value
    v2 = slider_2.value

    with out_leg:
        clear_output(wait=True)
        try:
            dims = generate_leg_mesh(t_id, v1, v2)
            plot_single_leg(dims)
        except Exception as e:
            print(e)

# Widgets
dd_type = widgets.Dropdown(options=[0, 1, 2], description='Leg Type:')
slider_1 = widgets.FloatSlider(min=-3, max=3, step=0.1, value=0, description='Dim 1')
slider_2 = widgets.FloatSlider(min=-3, max=3, step=0.1, value=0, description='Dim 2')

slider_1.style.handle_color = "blue"
slider_2.style.handle_color = "blue"

for w in [dd_type, slider_1, slider_2]:
    w.observe(update_leg_view, names='value')

ui = widgets.VBox([
    widgets.HTML("<h3><b>Universal Leg Factory</b></h3>"),
    widgets.Label("1. Select a Cluster (Type)"),
    dd_type,
    widgets.HTML("<hr>"),
    widgets.Label("2. Tweak Dimensions (AI)"),
    slider_1,
    slider_2
])

display(widgets.HBox([ui, out_leg]))
update_leg_view()


# In[39]:


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from abstractionssymh.abstraction_utils import Autoencoder, DEVICE

# 1. Extract Leg Scales (Rank 8)
leg_scales = []
for key, data in rank_8_dataset.items():
    dsl = data['dsl']
    # Flatten
    all_nodes = []
    def r_flat(n):
        if hasattr(n,'child'): all_nodes.append(n); r_flat(n.child)
        elif hasattr(n,'left'): all_nodes.append(n); r_flat(n.left); r_flat(n.right)
        elif hasattr(n,'children'): all_nodes.append(n); [r_flat(c) for c in n.children]
        else: all_nodes.append(n)
    r_flat(dsl)

    scales = [n for n in all_nodes if isinstance(n, Scale)]

    # In Rank 8, indices 1 & 2 are legs. Let's take both to get more data.
    if len(scales) > 2:
        leg_scales.append(scales[1].lengths)
        leg_scales.append(scales[2].lengths)

# Convert to Tensor
X_legs = torch.tensor(np.array(leg_scales), dtype=torch.float32).to(DEVICE)

# 2. Calculate Statistics for Normalization
# We need these to "guide" the network correctly
mean = X_legs.mean(dim=0)
std = X_legs.std(dim=0)
std[std == 0] = 1.0

# Normalized Data
X_legs_norm = (X_legs - mean) / std

print(f"Training on {len(X_legs)} legs.")
print(f"Average Leg Height: {mean[1]:.3f} (Std: {std[1]:.3f})")


# In[40]:


from torch.optim import AdamW
import torch.nn as nn

# --- CONFIGURATION ---
LATENT_DIM = 2
GUIDANCE_WEIGHT = 2.0  # How strict we are about Slider 0 being Height

# Initialize Model
ae_guided = Autoencoder(input_dim=3, hidden_dim=LATENT_DIM).to(DEVICE)
ae_guided.data_mean_ = mean
ae_guided.data_std_ = std

optimizer = AdamW(ae_guided.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("Starting Guided Training...")

# Training Loop
epochs = 50
for epoch in range(epochs):
    epoch_loss = 0

    # Shuffle indices manually since we aren't using DataLoader for this custom loop
    indices = torch.randperm(X_legs_norm.size(0))

    for i in range(0, len(indices), 64): # Batch size 64
        batch_idx = indices[i:i+64]
        batch = X_legs_norm[batch_idx]

        optimizer.zero_grad()

        # Forward Pass
        latent, reconstructed = ae_guided(batch)

        # --- LOSS CALCULATION ---
        # 1. Standard Reconstruction Loss (Make valid legs)
        recon_loss = loss_fn(reconstructed, batch)

        # 2. Semantic Guidance Loss (Force Latent[0] to match Normalized Height)
        # Note: batch[:, 1] is the normalized height (Scale Y)
        # We force latent[:, 0] to track it.
        guidance_loss = loss_fn(latent[:, 0], batch[:, 1])

        # Total Loss
        loss = recon_loss + (GUIDANCE_WEIGHT * guidance_loss)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

print("✅ Guided Model Trained. Slider 0 is now locked to Height.")


# In[ ]:


import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import numpy as np

# --- Visualization Helper ---
def plot_guided_leg(height_val, thickness_val):
    ae_guided.eval()

    # Construct Latent: [Height_Slider, Thickness_Slider]
    z = torch.tensor([[height_val, thickness_val]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        recon_norm = ae_guided.decoder(z)
        # Un-normalize
        dims = (recon_norm * ae_guided.data_std_ + ae_guided.data_mean_).cpu().numpy()[0]

    # Plot (Matplotlib)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw Box
    d = dims / 2
    center = np.array([0,0,0])

    # 8 Corners
    corners = np.array([
        center-d, center+d*[-1,1,-1], center+d*[1,1,-1], center+d*[1,-1,-1],
        center+d*[-1,-1,1], center+d*[-1,1,1], center+d, center+d*[1,-1,1]
    ])

    # 6 Faces (Indices)
    faces_indices = [
        [0,1,2,3], [4,5,6,7], [0,1,5,4], 
        [2,3,7,6], [0,3,7,4], [1,2,6,5]
    ]

    # --- THE FIX: Map Indices to Coordinates ---
    # We convert the list of indices into a list of actual (x,y,z) tuples
    verts = [[corners[i] for i in face] for face in faces_indices]

    poly = Poly3DCollection(verts, alpha=0.8)
    poly.set_facecolor('green')
    poly.set_edgecolor('k')
    ax.add_collection3d(poly)

    # Fixed Limits to see scaling effect
    limit = 0.6
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-0.2, 0.2)

    ax.set_xlabel(f"Width ({dims[0]:.2f})")
    ax.set_ylabel(f"Height ({dims[1]:.2f})")
    ax.set_zlabel(f"Depth ({dims[2]:.2f})")

    plt.tight_layout()
    plt.show()

# --- Dashboard ---
out_guided = widgets.Output()

def update_guided(change=None):
    h = slider_height.value
    t = slider_thick.value
    with out_guided:
        clear_output(wait=True)
        try:
            plot_guided_leg(h, t)
        except Exception as e:
            print(f"Plot Error: {e}")

# Sliders
slider_height = widgets.FloatSlider(min=-3, max=3, step=0.1, value=0, description='Height (Forced)')
slider_thick  = widgets.FloatSlider(min=-3, max=3, step=0.1, value=0, description='Thickness')

slider_height.style.handle_color = "green"
slider_thick.style.handle_color = "orange"

slider_height.observe(update_guided, names='value')
slider_thick.observe(update_guided, names='value')

ui = widgets.VBox([
    widgets.HTML("<h3><b>Semantically Guided Leg Generator</b></h3>"),
    slider_height,
    slider_thick
])

display(widgets.HBox([ui, out_guided]))
update_guided()


# In[42]:


import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. Prepare Data (Rank 8 Legs)
# Inputs:  [Height, Avg_Thickness]
# Targets: [Scale_X, Scale_Y, Scale_Z]
inputs = []
targets = []

print("Preparing Regression Data...")

for key, data in rank_8_dataset.items():
    dsl = data['dsl']
    # Flatten
    all_nodes = []
    def r_flat(n):
        if hasattr(n,'child'): all_nodes.append(n); r_flat(n.child)
        elif hasattr(n,'left'): all_nodes.append(n); r_flat(n.left); r_flat(n.right)
        elif hasattr(n,'children'): all_nodes.append(n); [r_flat(c) for c in n.children]
        else: all_nodes.append(n)
    r_flat(dsl)

    scales = [n for n in all_nodes if isinstance(n, Scale)]

    # Grab leg scales (Indices 1 & 2)
    for i in [1, 2]:
        if i < len(scales):
            vec = scales[i].lengths
            sx, sy, sz = vec

            # Define our Semantic Parameters
            height = sy
            thickness = (sx + sz) / 2.0 # Average width/depth

            inputs.append([height, thickness])
            targets.append([sx, sy, sz])

# Convert to Tensors
X_in = torch.tensor(np.array(inputs), dtype=torch.float32).to(DEVICE)
Y_out = torch.tensor(np.array(targets), dtype=torch.float32).to(DEVICE)

# Normalize Inputs (Crucial for slider feel)
in_mean = X_in.mean(dim=0)
in_std = X_in.std(dim=0)
X_in_norm = (X_in - in_mean) / in_std

# 2. Define the Regressor Model
# Simple MLP: Takes 2 inputs -> Predicts 3 outputs
class LegRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3) # Output: sx, sy, sz
        )
        # We attach stats to the model for easy usage later
        self.in_mean = in_mean
        self.in_std = in_std

    def forward(self, x):
        return self.net(x)

regressor = LegRegressor().to(DEVICE)
optimizer = AdamW(regressor.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 3. Train
print(f"Training Regressor on {len(X_in)} legs...")
loader = DataLoader(TensorDataset(X_in_norm, Y_out), batch_size=64, shuffle=True)

for epoch in range(100):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = regressor(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

print("✅ Leg Regressor Trained.")
print(f"  Input 0 Mean (Height):    {in_mean[0]:.3f}")
print(f"  Input 1 Mean (Thickness): {in_mean[1]:.3f}")


# In[43]:


import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Visualization Helper ---
def plot_regressed_leg(h_score, t_score):
    regressor.eval()

    # 1. Prepare Input Vector
    # We take the slider values (z-scores) and un-normalize them inside the input vector?
    # Actually, we trained on Normalized Inputs. So we pass the z-scores directly!
    # But wait, we normalized X_in before training. So yes, we pass [h_score, t_score].

    x = torch.tensor([[h_score, t_score]], dtype=torch.float32).to(DEVICE)

    # 2. Predict Shape
    with torch.no_grad():
        dims = regressor(x).cpu().numpy()[0]

    # 3. Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw Box
    d = dims / 2
    center = np.array([0,0,0])
    corners = np.array([
        center-d, center+d*[-1,1,-1], center+d*[1,1,-1], center+d*[1,-1,-1],
        center+d*[-1,-1,1], center+d*[-1,1,1], center+d, center+d*[1,-1,1]
    ])
    faces_indices = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]
    verts = [[corners[i] for i in face] for face in faces_indices]

    poly = Poly3DCollection(verts, alpha=0.8)
    poly.set_facecolor('#3333CC') # Blue
    poly.set_edgecolor('k')
    ax.add_collection3d(poly)

    # Limits
    limit = 0.6
    ax.set_xlim(-0.2, 0.2); ax.set_ylim(-limit, limit); ax.set_zlim(-0.2, 0.2)

    # Labels showing REAL world units
    ax.set_xlabel(f"Width ({dims[0]:.3f})")
    ax.set_ylabel(f"Height ({dims[1]:.3f})")
    ax.set_zlabel(f"Depth ({dims[2]:.3f})")
    ax.set_title(f"Input: H={h_score}, T={t_score}")

    plt.tight_layout()
    plt.show()

# --- Dashboard ---
out_reg = widgets.Output()

def update_reg(change=None):
    h = slider_h.value
    t = slider_t.value
    with out_reg:
        clear_output(wait=True)
        try:
            plot_regressed_leg(h, t)
        except Exception as e:
            print(e)

slider_h = widgets.FloatSlider(min=-3, max=3, step=0.1, value=0, description='Height')
slider_t = widgets.FloatSlider(min=-3, max=3, step=0.1, value=0, description='Thickness')

slider_h.style.handle_color = "blue"
slider_t.style.handle_color = "blue"

slider_h.observe(update_reg, names='value')
slider_t.observe(update_reg, names='value')

ui = widgets.VBox([
    widgets.HTML("<h3><b>Direct Leg Control (Regression)</b></h3>"),
    widgets.Label("Sliders control Standard Deviations from Mean"),
    slider_h,
    slider_t
])

display(widgets.HBox([ui, out_reg]))
update_reg()


# In[ ]:





# In[ ]:





# In[ ]:


# BRUH


# In[33]:


## 33. Extract Parameters for Rank 8

if 'rank_8_dataset' in locals() and rank_8_dataset:
    print(f"Extracting parameters for {len(rank_8_dataset)} chairs in Rank 8...")

    # 1. Get DSL trees
    rank_8_dsls = [data['dsl'] for data in rank_8_dataset.values()]

    # 2. Run Collection
    rank_8_singletons, rank_8_pairs = collect_singleton_and_pair_data(rank_8_dsls)

    # 3. Display Results
    print("\n" + "="*50)
    print("RANK 8 SINGLETONS")
    print("="*50)
    print(f"{'Node Type':<25} | {'Count':<10} | {'Per Chair'}")
    print("-" * 50)

    num_chairs = len(rank_8_dataset)
    for key, params in rank_8_singletons.items():
        count = len(params)
        per_chair = count / num_chairs
        print(f"{key:<25} | {count:<10} | {per_chair:.1f}")

    print("\n" + "="*50)
    print("RANK 8 PAIRS")
    print("="*50)
    for key, params in rank_8_pairs.items():
        count = len(params)
        per_chair = count / num_chairs
        print(f"{key:<25} | {count:<10} | {per_chair:.1f}")

    # Optional: Plot the 'Translate' to see the layout
    import plotly.express as px
    import pandas as pd
    import numpy as np

    if "Translate" in rank_8_singletons:
        vectors = np.array(rank_8_singletons["Translate"])
        df = pd.DataFrame(vectors, columns=['X', 'Y', 'Z'])

        fig = px.scatter_3d(
            df, x='X', y='Y', z='Z',
            title=f'Part Positions for Rank 8 (N={num_chairs})',
            opacity=0.7,
            color_discrete_sequence=['purple']
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

else:
    print("Error: 'rank_8_dataset' not found. Please run the previous cell to create it.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


## 29. Filter Dataset to Top 5 Archetypes

# Define which ranks you want to keep (0-based index)
# Rank 0 is the most common. Rank 4 is the 5th most common.
RANKS_TO_KEEP = [0, 1, 2, 3, 4]

print(f"Filtering dataset to keep only Structural Ranks: {RANKS_TO_KEEP}...")

filtered_dsl_shapes = {}
filtered_stats = []

for rank in RANKS_TO_KEEP:
    # 1. Get the chairs for this rank
    chairs, signature, count = get_chairs_by_rank(rank)

    print(f" -> Rank {rank}: Adding {count} chairs.")

    # 2. Add them to the new dictionary
    for chair_key in chairs:
        if chair_key in all_dsl_shapes:
            filtered_dsl_shapes[chair_key] = all_dsl_shapes[chair_key]

            # Keep track of metadata
            filtered_dsl_shapes[chair_key]['structure_rank'] = rank
            filtered_dsl_shapes[chair_key]['structure_signature'] = signature

# 3. Save the filtered dataset
filtered_pickle_path = saved_directory / "filtered_dsl_shapes_top5.pkl"

with open(filtered_pickle_path, "wb") as f:
    pickle.dump(filtered_dsl_shapes, f)

# 4. Display Summary
print("\n" + "="*50)
print("FILTERED DATASET SUMMARY")
print("="*50)
print(f"Original Count:   {len(all_dsl_shapes)}")
print(f"Filtered Count:   {len(filtered_dsl_shapes)}")
print(f"Retention Rate:   {len(filtered_dsl_shapes)/len(all_dsl_shapes):.1%}")
print(f"Saved to:         {filtered_pickle_path}")
print("="*50)

# Optional: Overwrite the main variable if you want to proceed with ONLY these
# all_dsl_shapes = filtered_dsl_shapes 
# print("Note: 'all_dsl_shapes' has NOT been overwritten yet. Use 'filtered_dsl_shapes' for next steps.")


# In[21]:


filtered_dsl_shapes


# In[22]:


## 30. Visualize Scale Parameters Colored by Rank

import plotly.express as px
import pandas as pd
import numpy as np

# 1. Collect Data with Rank Information
plot_data = []

print(f"Extracting Scale parameters from {len(filtered_dsl_shapes)} filtered chairs...")

for chair_key, data in tqdm(filtered_dsl_shapes.items(), desc="Processing Chairs"):
    # Retrieve the rank we stored earlier
    rank = data.get('structure_rank', 'Unknown')
    dsl = data['dsl']

    # Extract params for THIS specific chair only
    # We pass a list [dsl] because the function expects a list of trees
    singletons, _ = collect_singleton_and_pair_data([dsl])

    if "Scale" in singletons:
        for vec in singletons["Scale"]:
            # vec is [sx, sy, sz]
            plot_data.append({
                'SX': vec[0],
                'SY': vec[1],
                'SZ': vec[2],
                'Rank': f"Rank {rank}" # distinct string for legend
            })

# 2. Create DataFrame
df_scale_ranked = pd.DataFrame(plot_data)
# Sort by Rank so the legend is ordered
df_scale_ranked = df_scale_ranked.sort_values('Rank')

print(f"Extracted {len(df_scale_ranked)} total scale vectors.")

# 3. Interactive Plot
fig = px.scatter_3d(
    df_scale_ranked,
    x='SX',
    y='SY',
    z='SZ',
    color='Rank',
    title='Scale Parameters by Structure Rank (Top 5)',
    opacity=0.6,
    # Use a distinct color palette
    color_discrete_sequence=px.colors.qualitative.Bold
)

# 4. Refine Visuals
fig.update_traces(marker=dict(size=3))

fig.update_layout(
    scene=dict(
        xaxis_title='Scale X (Width)',
        yaxis_title='Scale Y (Height)',
        zaxis_title='Scale Z (Depth)',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend_title="Structure Type"
)

fig.show()


# In[23]:


## 31. Identify the "Unique" Parts
# Adjust these thresholds based on what you see in the Plotly graph!
# Example: Let's look for parts that are very long in X (width) but thin in Y/Z (Armrests?)
# OR distinctively different from the main leg cluster.

# --- USER CONFIGURATION (Look at your plot!) ---
# Example hypothesis: "What are the points with high X scale (> 0.5)?"
FILTER_CONDITION = "SX > 0.5 and SY < 0.2" 
# -----------------------------------------------

print(f"Searching for parts where: {FILTER_CONDITION} ...")

# Filter the dataframe we created in the previous step
distinct_parts = df_scale_ranked.query(FILTER_CONDITION)

if not distinct_parts.empty:
    print(f"Found {len(distinct_parts)} parts matching criteria.")

    # See which Ranks these belong to
    print("\nBreakdown by Rank:")
    print(distinct_parts['Rank'].value_counts())

    # Find a specific chair to visualize
    # We need to map back from the simplified dataframe to the chair key.
    # Since we didn't store the key in df_scale_ranked, we will re-scan quickly 
    # (or you can update the previous cell to store 'chair_key' in plot_data)

    # Let's do a quick re-scan for the first match to visualize
    found_key = None
    target_node = None

    for chair_key, data in filtered_dsl_shapes.items():
        dsl = data['dsl']
        singletons, _ = collect_singleton_and_pair_data([dsl])

        if "Scale" in singletons:
            for vec in singletons["Scale"]:
                # Check the condition manually
                sx, sy, sz = vec
                # Replicate the logic of your filter string roughly:
                if sx > 0.5 and sy < 0.2: 
                    found_key = chair_key
                    target_node = vec
                    break
        if found_key: break

    if found_key:
        print(f"\nVisualizing Chair: {found_key}")
        print(f"It has a part with Scale: {target_node}")
        plot_dsl_with_k3d(filtered_dsl_shapes[found_key]['dsl'])
    else:
        print("Could not retrieve exact source chair (logic check needed).")

else:
    print("No parts found in that region. Adjust your FILTER_CONDITION.")


# In[27]:


import k3d
import numpy as np
from scipy.spatial.transform import Rotation
from abstractionssymh.dsl_nodes import Scale
from abstractionssymh.abstraction_utils import Abstraction

def visualize_structure_indices(dsl_root, chair_title="Structure Analysis"):
    """
    Traverses the tree, finds every 'Scale' node (the definition of a part's shape),
    assigns it an ID based on traversal order, and plots it with that ID.
    """

    # 1. Flatten the tree to find nodes in deterministic order
    # We use a simple recursive finder
    all_nodes = []
    def recursive_flatten(node):
        if hasattr(node, 'child'):
            all_nodes.append(node)
            recursive_flatten(node.child)
        elif hasattr(node, 'left'): # Union
            all_nodes.append(node)
            recursive_flatten(node.left)
            recursive_flatten(node.right)
        elif hasattr(node, 'children'): # Abstraction
            all_nodes.append(node)
            for c in node.children:
                recursive_flatten(c)
        else: # Box or leaf
            all_nodes.append(node)

    recursive_flatten(dsl_root)

    # 2. Filter for "Part Definitions" (Scale nodes)
    # Scale is usually the best proxy for a "Part" because it defines the size.
    part_nodes = [n for n in all_nodes if isinstance(n, Scale)]

    print(f"Found {len(part_nodes)} distinct part definitions in this structure.")

    # 3. Setup Plot
    plot = k3d.plot(name=chair_title)

    # Distinct colors for IDs
    colors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0x00FFFF, 0xFF00FF, 0xFFFFFF]

    # 4. Plot each part with its ID
    for part_id, node in enumerate(part_nodes):
        # Expand ONLY this node to see its geometry
        # Note: We need to be careful. Expanding a node in isolation might place it 
        # at the origin if the parent Translate isn't included. 
        # However, for identification, shape matters more than position.
        # BUT, to make it readable, let's try to expand the whole tree 
        # and match the boxes to the node. 

        # SIMPLER APPROACH:
        # We assume the standard structure Translate -> Rotate -> Scale -> Box.
        # We will expand the node. If it's at 0,0,0, that's fine, we just need to see the shape.
        # To see it in context, we really want the full transformation.

        # Let's plot the FULL chair in Grey first for context
        full_boxes = dsl_root.expand()
        for box in full_boxes:
             # Plot ghost geometry
             pass # Skipping strict ghost geometry to keep scene clean, 
                  # instead we will plot the labeled parts brightly.

        # Expand the specific node (Local Frame)
        # This shows the shape of the part (e.g. Plate vs Stick)
        part_boxes = node.expand()

        color = colors[part_id % len(colors)]

        for box in part_boxes:
            center = np.array(box["center"], dtype=float)
            lengths = np.asarray(box["lengths"], dtype=float).ravel()

            # Draw the Mesh
            # (Simplified box generation for brevity - reuse your plotting logic if preferred)
            rotation_matrix = Rotation.from_quat(box["quaternion"]).as_matrix()
            d1, d2, d3 = [col * length / 2 for col, length in zip(rotation_matrix.T, lengths)]
            corners = np.array([
                center - d1 - d2 - d3, center - d1 + d2 - d3,
                center + d1 - d2 - d3, center + d1 + d2 - d3,
                center - d1 - d2 + d3, center - d1 + d2 + d3,
                center + d1 - d2 + d3, center + d1 + d2 + d3
            ], dtype=np.float32)
            faces = np.array([
                [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
                [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6]
            ], dtype=np.uint32)

            plot += k3d.mesh(corners, faces, color=color, opacity=1.0)

            # *** THE KEY PART: LABEL THE ID ***
            # We place the label slightly offset so you can read it
            plot += k3d.text(
                text=f"ID: {part_id}", 
                position=center, 
                color=0x000000, 
                size=1.0, 
                label_box=True
            )

    plot.display()

# --- Run on Rank 0 ---
chairs_rank0, _, _ = get_chairs_by_rank(0)
sample_chair = all_dsl_shapes[chairs_rank0[0]]['dsl']

visualize_structure_indices(sample_chair, "Rank 0 - Part ID Map")


# In[28]:


# HYPOTHETICAL EXAMPLE (You must verify with the plot above!)
# Replace these numbers with what you actually see in K3D
RANK_0_SEMANTIC_MAP = {
    0: "Seat",
    1: "Backrest",
    2: "Leg" 
}


# In[30]:


def extract_semantic_parts(dsl_tree, semantic_map):
    """
    Extracts params but labeled with semantic names (Seat, Leg)
    instead of generic indices.
    """
    extracted = {}

    # Flatten to get nodes in same order as visualization
    all_nodes = []
    def recursive_flatten(node):
        if hasattr(node, 'child'):
            all_nodes.append(node)
            recursive_flatten(node.child)
        elif hasattr(node, 'left'): 
            all_nodes.append(node)
            recursive_flatten(node.left)
            recursive_flatten(node.right)
        else:
            all_nodes.append(node)
    recursive_flatten(dsl_tree)

    part_nodes = [n for n in all_nodes if isinstance(n, Scale)]

    for index, node in enumerate(part_nodes):
        if index in semantic_map:
            name = semantic_map[index]
            extracted[name] = node.lengths # The scale vector

    return extracted

# Test it on a random chair from Rank 0
import random
random_key = random.choice(chairs_rank0)
dsl = all_dsl_shapes[random_key]['dsl']

parts = extract_semantic_parts(dsl, RANK_0_SEMANTIC_MAP)
print(f"Analysis of {random_key}:")
for part, dims in parts.items():
    print(f"  {part}: {dims}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


## 28. Global Parameter Extraction (Census)

print(f"Starting global extraction on {len(all_dsl_shapes)} shapes...")

# 1. Gather all DSL trees into a list
all_dsl_trees = [data['dsl'] for data in all_dsl_shapes.values()]

# 2. Run the collector utility
# This walks every tree and grabs every parameter vector
global_singletons, global_pairs = collect_singleton_and_pair_data(all_dsl_trees)

# 3. Display the "Census" Results
print("\n" + "="*50)
print("GLOBAL SINGLETONS (Node Types)")
print("="*50)
print(f"{'Node Type':<25} | {'Count':<10} | {'Param Dim'}")
print("-" * 50)

for key, params in global_singletons.items():
    # Check dimension of first item to know what we are dealing with
    dim = len(params[0]) if params else 0
    print(f"{key:<25} | {len(params):<10} | {dim}")

print("\n" + "="*50)
print("GLOBAL PAIRS (Parent-Child Relationships)")
print("="*50)
print(f"{'Pair Signature':<35} | {'Count':<10} | {'Combined Dim'}")
print("-" * 50)

for key, params in global_pairs.items():
    dim = len(params[0]) if params else 0
    print(f"{key:<35} | {len(params):<10} | {dim}")

debug_success("Global extraction complete.")


# In[7]:


import plotly.express as px
import pandas as pd
import numpy as np

# 1. Prepare Data
if 'global_singletons' in locals() and "Scale" in global_singletons:
    # Convert list of lists to numpy array
    scale_vectors = np.array(global_singletons["Scale"])

    # Create DataFrame for Plotly
    df_scale = pd.DataFrame(scale_vectors, columns=['SX', 'SY', 'SZ'])

    print(f"Plotting {len(df_scale)} scale vectors...")

    # 2. Create Interactive 3D Scatter Plot
    fig = px.scatter_3d(
        df_scale, 
        x='SX', 
        y='SY', 
        z='SZ',
        title='Global Distribution of Scale Parameters (Shape Primitives)',
        opacity=0.5,
        color_discrete_sequence=['green']  # Distinct color for Scale
    )

    # 3. Refine Visuals
    fig.update_traces(marker=dict(size=2)) # Smaller points for dense clouds

    fig.update_layout(
        scene=dict(
            xaxis_title='Scale X (Width)',
            yaxis_title='Scale Y (Height)',
            zaxis_title='Scale Z (Depth)',
            aspectmode='data' 
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

    # 4. Basic Stats (to guide your intuition)
    print("\n--- Scale Statistics ---")
    print(df_scale.describe().loc[['mean', 'std', 'min', 'max']])

else:
    print("Error: 'global_singletons' dictionary not found. Please run the global extraction cell first.")


# In[7]:


## 24. Structural Analysis (Grammar Discovery)

from abstractionssymh.dsl_nodes import (
    Box,
    Scale,
    Rotate,
    Translate,
    Union,
    SymRef,
    SymRot,
    SymTrans,
)

def get_structure_signature(node, include_labels=False):
    """
    Recursively generates a string signature of the tree structure,
    ignoring all continuous parameters.
    """
    # 1. Base Case: Box
    if isinstance(node, Box):
        if include_labels:
            return f"Box({node.label})"
        return "Box"

    # 2. Binary Case: Union
    # We sort the children signatures to ensure Union(A, B) == Union(B, A)
    if isinstance(node, Union):
        left_sig = get_structure_signature(node.left, include_labels)
        right_sig = get_structure_signature(node.right, include_labels)

        # Sort to canonicalize structure
        sigs = sorted([left_sig, right_sig])
        return f"Union({sigs[0]}, {sigs[1]})"

    # 3. Unary Cases (Transformations & Symmetries)
    node_type = type(node).__name__

    # Handle children
    # Most nodes have .child, but let's be robust
    if hasattr(node, 'child'):
        child_sig = get_structure_signature(node.child, include_labels)

        # Symmetry nodes often have discrete params (n_fold) that act like structure
        # We generally want to treat SymRot(4) different from SymRot(8)
        extra_info = ""
        if hasattr(node, 'n'): # SymRot, SymTrans
            extra_info = f",n={node.n}"

        return f"{node_type}{extra_info}({child_sig})"

    return "Unknown"

# --- Main Grouping Loop ---

debug_info("Starting Structural Analysis on DSL trees...")

structure_groups = defaultdict(list)
structure_stats = []

for chair_key, data in tqdm(all_dsl_shapes.items(), desc="Generating Signatures"):
    dsl_tree = data['dsl']

    # Generate signature
    signature = get_structure_signature(dsl_tree, include_labels=False)

    # Store
    structure_groups[signature].append(chair_key)

# --- Process Results into DataFrame ---

debug_info("Aggregating structural groups...")

for signature, chairs in structure_groups.items():
    # Calculate a rough "complexity" by counting open parenthesis
    complexity = signature.count('(')

    structure_stats.append({
        'signature': signature,
        'count': len(chairs),
        'complexity': complexity,
        'example_chair': chairs[0], # Pick one random example
        'all_chairs': chairs
    })

# Convert to DataFrame and Sort
df_structures = pd.DataFrame(structure_stats)
df_structures = df_structures.sort_values(by='count', ascending=False).reset_index(drop=True)

# --- Display Statistics ---
total_unique_structures = len(df_structures)
top_1_count = df_structures.iloc[0]['count']
top_10_sum = df_structures.head(10)['count'].sum()

print("\n" + "="*60)
print(f"STRUCTURAL GRAMMAR ANALYSIS")
print("="*60)
print(f"Total Shapes Analyzed:      {len(all_dsl_shapes)}")
print(f"Unique Structures Found:    {total_unique_structures}")
print(f"Most Common Structure:      {top_1_count} chairs share the exact same tree.")
print(f"Top 10 Structures Cover:    {top_10_sum} chairs ({(top_10_sum/len(all_dsl_shapes))*100:.1f}%)")
print("="*60 + "\n")

# Show Top 20 Most Common Structures
pd.set_option('display.max_colwidth', 100)
display(df_structures[['count', 'complexity', 'signature', 'example_chair']].head(20))


# In[8]:


## 25. Visualize the Top Structural Archetypes

top_n_to_visualize = 5

print(f"Visualizing the top {top_n_to_visualize} most common structural archetypes...")

archetype_dsls = []
archetype_names = []

for i in range(top_n_to_visualize):
    row = df_structures.iloc[i]
    chair_key = row['example_chair']
    count = row['count']

    # Load DSL
    dsl_obj = all_dsl_shapes[chair_key]['dsl']

    archetype_dsls.append(dsl_obj)
    archetype_names.append(f"Rank {i+1} (N={count})\n{chair_key}")

# Plot grid
plot_dsl_grid(
    archetype_dsls, 
    archetype_names, 
    grid_cols=5, 
    grid_title="Top Structural Archetypes (The 'Template' Chairs)"
)


# In[9]:


## 26. Finalize and Save Structure Buckets

# This dictionary maps: Signature (str) -> List of Chair Keys (list[str])
structure_buckets = dict(structure_groups)

# Let's save this so we don't have to re-run the analysis every time
buckets_pickle_path = saved_directory / "structure_buckets.pkl"

with open(buckets_pickle_path, "wb") as f:
    pickle.dump(structure_buckets, f)

debug_success(f"Saved structure buckets to {buckets_pickle_path}")
print(f"Total Buckets: {len(structure_buckets)}")

# --- Helper function to get chairs by Rank ---
# Since signatures are long strings, it's easier to ask for "Rank 1" (most common)
def get_chairs_by_rank(rank_index):
    """
    Returns the list of chairs for the Nth most common structure.
    rank_index is 0-based (0 = Most Common).
    """
    if rank_index >= len(df_structures):
        print(f"Error: Rank {rank_index} out of bounds.")
        return [], None

    row = df_structures.iloc[rank_index]
    sig = row['signature']
    chairs = row['all_chairs']
    count = row['count']

    return chairs, sig, count

# Example: Get details of the #1 most common bucket
top_chairs, top_sig, top_count = get_chairs_by_rank(0)

print(f"\nExample - Rank 0 Bucket:")
print(f"Signature: {top_sig[:100]}...") # Truncated for display
print(f"Contains {top_count} chairs.")
print(f"First 5 IDs: {top_chairs[:5]}")


# In[10]:


## 27. Visualize a Specific Bucket

# Change this to see different groups!
# 0 = The most common group (943 chairs)
# 1 = The second most common group
TARGET_RANK = 0 

chairs_in_bucket, signature, count = get_chairs_by_rank(TARGET_RANK)

print(f"Visualizing random sample from Rank {TARGET_RANK} (Count: {count})")
print(f"Structure Signature: {signature}")

# Pick 6 random chairs from this specific bucket
if len(chairs_in_bucket) > 6:
    sample_keys = random.sample(chairs_in_bucket, 6)
else:
    sample_keys = chairs_in_bucket

bucket_dsls = [all_dsl_shapes[k]['dsl'] for k in sample_keys]
bucket_names = [k for k in sample_keys]

plot_dsl_grid(
    bucket_dsls, 
    bucket_names, 
    grid_cols=3, 
    grid_title=f"Sample from Structure Rank {TARGET_RANK}"
)


# In[11]:


# 1. Get the data for the specific bucket (Rank 0)
target_rank = 0
chairs_in_bucket, signature, count = get_chairs_by_rank(target_rank)

print(f"Extracting parameters for {count} chairs in Rank {target_rank}...")

# 2. Collect Singletons and Pairs for ONLY this bucket
# We first get the DSL objects for these specific chairs
bucket_dsl_objects = [all_dsl_shapes[k]['dsl'] for k in chairs_in_bucket]

# Use the util function to extract params
bucket_singletons, bucket_pairs = collect_singleton_and_pair_data(bucket_dsl_objects)

# 3. Display what we found
print("\n--- Found Singletons (Node Types) ---")
for key, params in bucket_singletons.items():
    # params is a list of lists.
    # E.g. if there are 943 chairs, and each chair has 1 Seat, 1 Back, 4 Legs (abstracted),
    # The counts might vary depending on how many times that node appears in the tree.
    print(f"{key:<20} : {len(params)} samples found")

print("\n--- Found Pairs (Parent-Child Relations) ---")
for key, params in bucket_pairs.items():
    print(f"{key:<30} : {len(params)} samples found")

# 4. Plot a sample in K3D
print(f"\nVisualizing a random chair from this bucket in K3D...")
random_sample_key = random.choice(chairs_in_bucket)
sample_dsl = all_dsl_shapes[random_sample_key]['dsl']

plot_dsl_with_k3d(sample_dsl)


# In[15]:


import plotly.express as px
import pandas as pd
import numpy as np

# 1. Prepare Data
# Check if bucket_singletons exists
if 'bucket_singletons' in locals() and "Translate" in bucket_singletons:
    translate_vectors = np.array(bucket_singletons["Translate"])

    # Create DataFrame for Plotly
    df_trans = pd.DataFrame(translate_vectors, columns=['X', 'Y', 'Z'])

    print(f"Plotting {len(df_trans)} points with Plotly...")

    # 2. Create Interactive 3D Scatter Plot
    fig = px.scatter_3d(
        df_trans, 
        x='X', 
        y='Y', 
        z='Z',
        title='Distribution of Translate Parameters (Rank 0)',
        opacity=0.6,  # Helps see density in clusters
    )

    # 3. Refine Visuals
    fig.update_traces(marker=dict(size=3, color='red'))

    # Fix the aspect ratio so the chair doesn't look stretched
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Left/Right)',
            yaxis_title='Y (Up/Down)',
            zaxis_title='Z (Front/Back)',
            aspectmode='data' 
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
else:
    print("Error: 'bucket_singletons' dictionary not found. Please run the extraction cell first.")


# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 1. Prepare Data
if 'bucket_singletons' in locals() and "Translate" in bucket_singletons:
    translate_vectors = np.array(bucket_singletons["Translate"])

    n_components_range = range(1, 11)
    bic_scores = []
    aic_scores = []

    print(f"Running Elbow Method (BIC/AIC) for GMM on {len(translate_vectors)} points...")

    # 2. Loop through components to calculate BIC/AIC
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(translate_vectors)
        bic_scores.append(gmm.bic(translate_vectors))
        aic_scores.append(gmm.aic(translate_vectors))

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bic_scores, marker='o', label='BIC (Bayesian Info Criterion)', color='blue')
    plt.plot(n_components_range, aic_scores, marker='x', label='AIC (Akaike Info Criterion)', color='green')

    plt.title('Elbow Method for GMM (Lower Score is Better)')
    plt.xlabel('Number of Components (Clusters)')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(n_components_range)
    plt.show()

    # 4. Recommendation
    optimal_n = n_components_range[np.argmin(bic_scores)]
    print(f"\n--- Analysis Results ---")
    print(f"Minimum BIC occurs at n={optimal_n}")
    print("Look for the 'elbow' in the plot - the point where the score stops dropping significantly.")
    print("If n=4 is the elbow, it confirms our hypothesis (Seat, Back, 2 Legs).")

else:
    print("Error: 'bucket_singletons' dictionary not found. Please run the extraction cell first.")


# In[19]:


import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 1. Get Data for Rank 1 (The 2nd most common structure)
TARGET_RANK = 1
chairs_in_bucket, signature, count = get_chairs_by_rank(TARGET_RANK)

print(f"--- Analyzing Rank {TARGET_RANK} (Count: {count}) ---")
print(f"Signature: {signature[:100]}...") 

# 2. Extract Parameters
bucket_dsl_objects = [all_dsl_shapes[k]['dsl'] for k in chairs_in_bucket]
bucket_singletons, _ = collect_singleton_and_pair_data(bucket_dsl_objects)

if "Translate" in bucket_singletons:
    translate_vectors = np.array(bucket_singletons["Translate"])
    print(f"Plotting {len(translate_vectors)} translation points...")

    # --- PART A: 3D Scatter Plot ---
    df_trans = pd.DataFrame(translate_vectors, columns=['X', 'Y', 'Z'])

    fig = px.scatter_3d(
        df_trans, x='X', y='Y', z='Z',
        title=f'Translate Parameters (Rank {TARGET_RANK})',
        opacity=0.6,
        color_discrete_sequence=['blue'] # Different color for distinction
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

    # --- PART B: Elbow Method (Check for Clusters) ---
    print("\nRunning Elbow Method for Rank 1...")
    n_components_range = range(1, 11)
    bic_scores = []

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(translate_vectors)
        bic_scores.append(gmm.bic(translate_vectors))

    # Plot Elbow
    plt.figure(figsize=(8, 4))
    plt.plot(n_components_range, bic_scores, marker='o', color='purple')
    plt.title(f'Elbow Method (BIC) for Rank {TARGET_RANK}')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(n_components_range)
    plt.show()

    optimal_n = n_components_range[np.argmin(bic_scores)]
    print(f"Optimal clusters (BIC min): {optimal_n}")

else:
    print("No 'Translate' nodes found in this structure bucket.")


# In[20]:


import plotly.express as px
import pandas as pd
import numpy as np

# Initialize a list to hold dataframes
dfs = []

# Define the ranks to compare
ranks_to_plot = [0, 1]
colors = ['blue', 'red']  # Explicit colors for clarity

print("Extracting and combining data...")

for rank in ranks_to_plot:
    # 1. Get Chairs
    chairs, _, count = get_chairs_by_rank(rank)

    # 2. Extract DSLs & Parameters
    bucket_dsls = [all_dsl_shapes[k]['dsl'] for k in chairs]
    singletons, _ = collect_singleton_and_pair_data(bucket_dsls)

    # 3. Process Translate Vectors
    if "Translate" in singletons:
        vectors = np.array(singletons["Translate"])

        # Create temp DataFrame
        temp_df = pd.DataFrame(vectors, columns=['X', 'Y', 'Z'])
        temp_df['Rank'] = f"Rank {rank} (N={count})" # Label for legend

        dfs.append(temp_df)
    else:
        print(f"Warning: Rank {rank} has no Translate nodes.")

# 4. Combine and Plot
if dfs:
    df_combined = pd.concat(dfs, ignore_index=True)

    print(f"Plotting total {len(df_combined)} points...")

    fig = px.scatter_3d(
        df_combined,
        x='X', y='Y', z='Z',
        color='Rank',
        title='Comparison of Translate Parameters: Rank 0 vs Rank 1',
        opacity=0.5, # Low opacity helps see overlaps
        color_discrete_sequence=colors
    )

    fig.update_traces(marker=dict(size=3))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (Left/Right)',
            yaxis_title='Y (Up/Down)',
            zaxis_title='Z (Front/Back)',
            aspectmode='data'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.show()
else:
    print("No data found to plot.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


## 4. Display L0 Chairs (Sample)

if all_dsl_shapes:
    num_to_sample = min(9, len(all_dsl_shapes))
    random_keys = random.sample(list(all_dsl_shapes.keys()), num_to_sample)
    dsl_list = [all_dsl_shapes[key]['dsl'] for key in random_keys]
    name_list = [key.replace('.json', '') for key in random_keys]

    plot_dsl_grid(
        dsl_list,
        name_list,
        grid_cols=3,
        grid_title=f"Random Sample of {len(dsl_list)} L0 Chairs"
    )
else:
    debug_error("No DSL shapes loaded, cannot display.")


# In[6]:


## 5. Extract L1 Structures & Parameters (Common for both AE and PCA)

debug_info("Building L1 detailed dictionaries for singletons and pairs...")
combined_singletons_detailed_L1 = defaultdict(list)
combined_pairs_detailed_L1 = defaultdict(list)

for filename, data in tqdm(all_dsl_shapes.items(), desc="Aggregating L1 Parameters"):
    # SINGLETON parameters
    for pattern_name, param_lists in data["singleton_params"].items():
        if "Box" in pattern_name: continue # Skip Box nodes
        for param_list in param_lists or []:
            combined_singletons_detailed_L1[pattern_name].append({
                'file': filename, 'params': param_list
            })
    # PAIR parameters
    for pattern_name, param_lists in data["pair_params"].items():
        if "Box" in pattern_name: continue # Skip Box nodes
        for param_list in param_lists or []:
            combined_pairs_detailed_L1[pattern_name].append({
                'file': filename, 'params': param_list
            })

debug_success(f"Aggregated all L1 parameters.")


# In[7]:


## 6. Prepare L1 Training Data (Common for both AE and PCA)

debug_info("--- Preparing L1 data for model training ---")

training_singleton_params_L1 = {}
for pattern_name, records in combined_singletons_detailed_L1.items():
    if records:
        training_singleton_params_L1[pattern_name] = [rec['params'] for rec in records]

training_pair_params_L1 = {}
for pattern_name, records in combined_pairs_detailed_L1.items():
    if records:
        training_pair_params_L1[pattern_name] = [rec['params'] for rec in records]

debug_success(f"L1 Data flattened for training.")
print(f"Found {len(training_singleton_params_L1)} L1 singleton patterns to train.")
print(f"Found {len(training_pair_params_L1)} L1 pair patterns to train.")


# In[8]:


## 7.   AUTOENCODER Pipeline: Train/Load L1 Models

ABSTRACTION_METHOD_AE = 'ae'
debug_info(f"--- STARTING AUTOENCODER (AE) L1 PIPELINE ---")

models_exist_L1_AE = any(saved_models_L1_AE_dir.glob('*.pth'))
singleton_models_L1_AE = {}
pair_models_L1_AE = {}

if models_exist_L1_AE:
    debug_info(f"--- L1 AE models found. Loading from {saved_models_L1_AE_dir} ---")

    # Load L1 AE Singleton Models
    for name in training_singleton_params_L1.keys():
        save_file = saved_models_L1_AE_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            try:
                # --- FIX: Added weights_only=False ---
                model = torch.load(save_file, map_location=DEVICE, weights_only=False)
                model.eval()
                singleton_models_L1_AE[name] = model
            except Exception as e:
                debug_error(f"Failed to load L1 AE model '{name}': {e}")

    # Load L1 AE Pair Models
    for name in training_pair_params_L1.keys():
        save_file = saved_models_L1_AE_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            try:
                # --- FIX: Added weights_only=False ---
                model = torch.load(save_file, map_location=DEVICE, weights_only=False)
                model.eval()
                pair_models_L1_AE[name] = model
            except Exception as e:
                debug_error(f"Failed to load L1 AE model '{name}': {e}")
else:
    debug_info(f"--- No L1 AE models found. Starting training... ---")

    singleton_models_L1_AE = find_abstractions(
        training_singleton_params_L1, 
        method=ABSTRACTION_METHOD_AE,
        structure_type="SINGLETONS_L1_AE", 
        min_examples=50, 
        epochs=20,
        save_dir=saved_models_L1_AE_dir,
        plot_error_distribution=True
    )
    pair_models_L1_AE = find_abstractions(
        training_pair_params_L1, 
        method=ABSTRACTION_METHOD_AE,
        structure_type="PAIRS_L1_AE", 
        min_examples=50, 
        epochs=20,
        save_dir=saved_models_L1_AE_dir,
        plot_error_distribution=True
    )

    # Save L1 AE Models (CRITICAL: save the whole model, not just state_dict)
    for name, model in singleton_models_L1_AE.items():
        torch.save(model, saved_models_L1_AE_dir / make_safe_filename(name, suffix="pth"))
    for name, model in pair_models_L1_AE.items():
        torch.save(model, saved_models_L1_AE_dir / make_safe_filename(name, suffix="pth"))
    debug_success(f"Saved L1 AE models to {saved_models_L1_AE_dir}")

debug_success(f"--- L1 AE Workflow complete. {len(singleton_models_L1_AE)} singleton and {len(pair_models_L1_AE)} pair models ready. ---")


# In[9]:


## 8.   AUTOENCODER Pipeline: Create L1-AE Abstracted Dataset

debug_info("--- Creating new L1-AE Abstracted Dataset ---")
all_abstracted_shapes_L1_AE = {}
pickle_file_L1_AE = saved_directory / "all_abstracted_shapes_L1_AE.pkl"

if pickle_file_L1_AE.exists():
    with open(pickle_file_L1_AE, "rb") as f:
        all_abstracted_shapes_L1_AE = pickle.load(f)
    debug_success(f"Loaded {len(all_abstracted_shapes_L1_AE)} L1-AE abstracted shapes.")
else:
    for filename, data in tqdm(all_dsl_shapes.items(), desc="Integrating L1-AE Abstractions"):
        abstracted_dsl = integrate_abstractions(
            data["dsl"],
            singleton_models_L1_AE,
            pair_models_L1_AE,
            error_threshold=0.02,
            # detailed_debug=True
        )
        l1_singletons, l1_pairs = collect_singleton_and_pair_data([abstracted_dsl])
        all_abstracted_shapes_L1_AE[filename] = {
            "dsl": abstracted_dsl,
            "singleton_params": l1_singletons,
            "pair_params": l1_pairs,
            "original_dsl": data["dsl"]
        }
    with open(pickle_file_L1_AE, "wb") as f:
        pickle.dump(all_abstracted_shapes_L1_AE, f)
    debug_success(f"Created and saved {len(all_abstracted_shapes_L1_AE)} L1-AE shapes.")


# In[10]:


## 9.   AUTOENCODER Pipeline: Extract L2-AE Parameters

debug_info("--- Extracting L2-AE Parameters ---")
combined_singletons_detailed_L2_AE = defaultdict(list)
combined_pairs_detailed_L2_AE = defaultdict(list)

for filename, data in tqdm(all_abstracted_shapes_L1_AE.items(), desc="Aggregating L2-AE Params"):
    for p_name, p_lists in data["singleton_params"].items():
        if "Box" in p_name: continue
        for p_list in p_lists or []:
            combined_singletons_detailed_L2_AE[p_name].append({'params': p_list})
    for p_name, p_lists in data["pair_params"].items():
        if "Box" in p_name: continue
        for p_list in p_lists or []:
            combined_pairs_detailed_L2_AE[p_name].append({'params': p_list})

# Prepare L2-AE Training Data
training_singleton_params_L2_AE = {
    k: [r['params'] for r in v] for k, v in combined_singletons_detailed_L2_AE.items() if v
}
training_pair_params_L2_AE = {
    k: [r['params'] for r in v] for k, v in combined_pairs_detailed_L2_AE.items() if v
}

debug_success(f"Found {len(training_singleton_params_L2_AE)} L2-AE singleton and {len(training_pair_params_L2_AE)} L2-AE pair patterns.")


# In[11]:


## 10.   AUTOENCODER Pipeline: Train/Load L2-AE Models

debug_info("--- Starting L2-AE Abstraction Pipeline ---")
models_exist_L2_AE = any(saved_models_L2_AE_dir.glob('*.pth'))
singleton_models_L2_AE = {}
pair_models_L2_AE = {}

if models_exist_L2_AE:
    debug_info(f"--- L2 AE models found. Loading from {saved_models_L2_AE_dir} ---")
    for name in training_singleton_params_L2_AE.keys():
        save_file = saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            # --- FIX: Added weights_only=False ---
            model = torch.load(save_file, map_location=DEVICE, weights_only=False); model.eval()
            singleton_models_L2_AE[name] = model
    for name in training_pair_params_L2_AE.keys():
        save_file = saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            # --- FIX: Added weights_only=False ---
            model = torch.load(save_file, map_location=DEVICE, weights_only=False); model.eval()
            pair_models_L2_AE[name] = model
else:
    debug_info("--- No L2 AE models found. Starting training... ---")
    singleton_models_L2_AE = find_abstractions(
        training_singleton_params_L2_AE, method='ae', structure_type="SINGLETONS_L2_AE", min_examples=50, epochs=20, save_dir=saved_models_L2_AE_dir,
        plot_error_distribution=True
    )
    pair_models_L2_AE = find_abstractions(
        training_pair_params_L2_AE, method='ae', structure_type="PAIRS_L2_AE", min_examples=50, epochs=20, save_dir=saved_models_L2_AE_dir,
        plot_error_distribution=True
    )
    for name, model in singleton_models_L2_AE.items():
        torch.save(model, saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth"))
    for name, model in pair_models_L2_AE.items():
        torch.save(model, saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth"))
    debug_success(f"Saved L2 AE models to {saved_models_L2_AE_dir}")

debug_success(f"--- L2 AE Workflow complete. {len(singleton_models_L2_AE)} singleton and {len(pair_models_L2_AE)} pair models ready. ---")


# In[12]:


## 11.   AUTOENCODER Pipeline: Create L2-AE Abstracted Dataset

debug_info("--- Creating new L2-AE Abstracted Dataset ---")
all_abstracted_shapes_L2_AE = {}
pickle_file_L2_AE = saved_directory / "all_abstracted_shapes_L2_AE.pkl"

if pickle_file_L2_AE.exists():
    with open(pickle_file_L2_AE, "rb") as f:
        all_abstracted_shapes_L2_AE = pickle.load(f)
    debug_success(f"Loaded {len(all_abstracted_shapes_L2_AE)} L2-AE abstracted shapes.")
else:
    for filename, data in tqdm(all_abstracted_shapes_L1_AE.items(), desc="Integrating L2-AE Abstractions"):
        abstracted_dsl_L2 = integrate_abstractions(
            data["dsl"],
            singleton_models_L2_AE,
            pair_models_L2_AE,
            error_threshold=0.2
        )
        l2_singletons, l2_pairs = collect_singleton_and_pair_data([abstracted_dsl_L2])
        all_abstracted_shapes_L2_AE[filename] = {
            "dsl": abstracted_dsl_L2,
            "singleton_params": l2_singletons,
            "pair_params": l2_pairs,
            "original_dsl": data["original_dsl"]
        }
    with open(pickle_file_L2_AE, "wb") as f:
        pickle.dump(all_abstracted_shapes_L2_AE, f)
    debug_success(f"Created and saved {len(all_abstracted_shapes_L2_AE)} L2-AE shapes.")


# In[13]:


## 12.   PCA Pipeline: Train/Load L1 Models

ABSTRACTION_METHOD_PCA = 'pca'
debug_info(f"--- STARTING PCA L1 PIPELINE ---")

models_exist_L1_PCA = any(saved_models_L1_PCA_dir.glob('*.pth'))
singleton_models_L1_PCA = {}
pair_models_L1_PCA = {}

if models_exist_L1_PCA:
    debug_info(f"--- L1 PCA models found. Loading from {saved_models_L1_PCA_dir} ---")
    for name in training_singleton_params_L1.keys():
        save_file = saved_models_L1_PCA_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            try:
                # --- FIX: Added weights_only=False ---
                model = torch.load(save_file, map_location=DEVICE, weights_only=False)
                model.eval()
                singleton_models_L1_PCA[name] = model
            except Exception as e:
                debug_error(f"Failed to load L1 PCA model '{name}': {e}")
    for name in training_pair_params_L1.keys():
        save_file = saved_models_L1_PCA_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            try:
                # --- FIX: Added weights_only=False ---
                model = torch.load(save_file, map_location=DEVICE, weights_only=False)
                model.eval()
                pair_models_L1_PCA[name] = model
            except Exception as e:
                debug_error(f"Failed to load L1 PCA model '{name}': {e}")
else:
    debug_info(f"--- No L1 PCA models found. Starting fitting... ---")
    singleton_models_L1_PCA = find_abstractions(
        training_singleton_params_L1, 
        method=ABSTRACTION_METHOD_PCA,
        structure_type="SINGLETONS_L1_PCA", 
        min_examples=50, 
        retrain_iterations=1,
        error_threshold=0.01
    )
    pair_models_L1_PCA = find_abstractions(
        training_pair_params_L1, 
        method=ABSTRACTION_METHOD_PCA,
        structure_type="PAIRS_L1_PCA", 
        min_examples=50, 
        retrain_iterations=1,
        error_threshold=0.01
    )
    for name, model in singleton_models_L1_PCA.items():
        torch.save(model, saved_models_L1_PCA_dir / make_safe_filename(name, suffix="pth"))
    for name, model in pair_models_L1_PCA.items():
        torch.save(model, saved_models_L1_PCA_dir / make_safe_filename(name, suffix="pth"))
    debug_success(f"Saved L1 PCA models to {saved_models_L1_PCA_dir}")

debug_success(f"--- L1 PCA Workflow complete. {len(singleton_models_L1_PCA)} singleton and {len(pair_models_L1_PCA)} pair models ready. ---")


# In[14]:


## 13.   PCA Pipeline: Create L1-PCA Abstracted Dataset

debug_info("--- Creating new L1-PCA Abstracted Dataset ---")
all_abstracted_shapes_L1_PCA = {}
pickle_file_L1_PCA = saved_directory / "all_abstracted_shapes_L1_PCA.pkl"

if pickle_file_L1_PCA.exists():
    with open(pickle_file_L1_PCA, "rb") as f:
        all_abstracted_shapes_L1_PCA = pickle.load(f)
    debug_success(f"Loaded {len(all_abstracted_shapes_L1_PCA)} L1-PCA abstracted shapes.")
else:
    for filename, data in tqdm(all_dsl_shapes.items(), desc="Integrating L1-PCA Abstractions"):
        abstracted_dsl = integrate_abstractions(
            data["dsl"],
            singleton_models_L1_PCA,
            pair_models_L1_PCA,
            error_threshold=0.01
        )
        l1_singletons, l1_pairs = collect_singleton_and_pair_data([abstracted_dsl])
        all_abstracted_shapes_L1_PCA[filename] = {
            "dsl": abstracted_dsl,
            "singleton_params": l1_singletons,
            "pair_params": l1_pairs,
            "original_dsl": data["dsl"]
        }
    with open(pickle_file_L1_PCA, "wb") as f:
        pickle.dump(all_abstracted_shapes_L1_PCA, f)
    debug_success(f"Created and saved {len(all_abstracted_shapes_L1_PCA)} L1-PCA shapes.")


# In[15]:


## 14.   PCA Pipeline: Extract L2-PCA Parameters

debug_info("--- Extracting L2-PCA Parameters ---")
combined_singletons_detailed_L2_PCA = defaultdict(list)
combined_pairs_detailed_L2_PCA = defaultdict(list)

for filename, data in tqdm(all_abstracted_shapes_L1_PCA.items(), desc="Aggregating L2-PCA Params"):
    for p_name, p_lists in data["singleton_params"].items():
        if "Box" in p_name: continue
        for p_list in p_lists or []:
            combined_singletons_detailed_L2_PCA[p_name].append({'params': p_list})
    for p_name, p_lists in data["pair_params"].items():
        if "Box" in p_name: continue
        for p_list in p_lists or []:
            combined_pairs_detailed_L2_PCA[p_name].append({'params': p_list})

# Prepare L2-PCA Training Data
training_singleton_params_L2_PCA = {
    k: [r['params'] for r in v] for k, v in combined_singletons_detailed_L2_PCA.items() if v
}
training_pair_params_L2_PCA = {
    k: [r['params'] for r in v] for k, v in combined_pairs_detailed_L2_PCA.items() if v
}

debug_success(f"Found {len(training_singleton_params_L2_PCA)} L2-PCA singleton and {len(training_pair_params_L2_PCA)} L2-PCA pair patterns.")


# In[16]:


## 15.   PCA Pipeline: Train/Load L2-PCA Models

debug_info("--- Starting L2-PCA Abstraction Pipeline ---")
models_exist_L2_PCA = any(saved_models_L2_PCA_dir.glob('*.pth'))
singleton_models_L2_PCA = {}
pair_models_L2_PCA = {}

if models_exist_L2_PCA:
    debug_info(f"--- L2 PCA models found. Loading from {saved_models_L2_PCA_dir} ---")
    for name in training_singleton_params_L2_PCA.keys():
        save_file = saved_models_L2_PCA_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            # --- FIX: Added weights_only=False ---
            model = torch.load(save_file, map_location=DEVICE, weights_only=False); model.eval()
            singleton_models_L2_PCA[name] = model
    for name in training_pair_params_L2_PCA.keys():
        save_file = saved_models_L2_PCA_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            # --- FIX: Added weights_only=False ---
            model = torch.load(save_file, map_location=DEVICE, weights_only=False); model.eval()
            pair_models_L2_PCA[name] = model
else:
    debug_info("--- No L2 PCA models found. Starting fitting... ---")
    singleton_models_L2_PCA = find_abstractions(
        training_singleton_params_L2_PCA, method='pca', structure_type="SINGLETONS_L2_PCA", min_examples=50, error_threshold=0.01
    )
    pair_models_L2_PCA = find_abstractions(
        training_pair_params_L2_PCA, method='pca', structure_type="PAIRS_L2_PCA", min_examples=50, error_threshold=0.01
    )
    for name, model in singleton_models_L2_PCA.items():
        torch.save(model, saved_models_L2_PCA_dir / make_safe_filename(name, suffix="pth"))
    for name, model in pair_models_L2_PCA.items():
        torch.save(model, saved_models_L2_PCA_dir / make_safe_filename(name, suffix="pth"))
    debug_success(f"Saved L2 PCA models to {saved_models_L2_PCA_dir}")

debug_success(f"--- L2 PCA Workflow complete. {len(singleton_models_L2_PCA)} singleton and {len(pair_models_L2_PCA)} pair models ready. ---")


# In[17]:


## 16.   PCA Pipeline: Create L2-PCA Abstracted Dataset

debug_info("--- Creating new L2-PCA Abstracted Dataset ---")
all_abstracted_shapes_L2_PCA = {}
pickle_file_L2_PCA = saved_directory / "all_abstracted_shapes_L2_PCA.pkl"

if pickle_file_L2_PCA.exists():
    with open(pickle_file_L2_PCA, "rb") as f:
        all_abstracted_shapes_L2_PCA = pickle.load(f)
    debug_success(f"Loaded {len(all_abstracted_shapes_L2_PCA)} L2-PCA abstracted shapes.")
else:
    for filename, data in tqdm(all_abstracted_shapes_L1_PCA.items(), desc="Integrating L2-PCA Abstractions"):
        abstracted_dsl_L2 = integrate_abstractions(
            data["dsl"],
            singleton_models_L2_PCA,
            pair_models_L2_PCA,
            error_threshold=0.1
        )
        l2_singletons, l2_pairs = collect_singleton_and_pair_data([abstracted_dsl_L2])
        all_abstracted_shapes_L2_PCA[filename] = {
            "dsl": abstracted_dsl_L2,
            "singleton_params": l2_singletons,
            "pair_params": l2_pairs,
            "original_dsl": data["original_dsl"]
        }
    with open(pickle_file_L2_PCA, "wb") as f:
        pickle.dump(all_abstracted_shapes_L2_PCA, f)
    debug_success(f"Created and saved {len(all_abstracted_shapes_L2_PCA)} L2-PCA shapes.")


# In[18]:


## 17. Analysis Helper Functions

def analyze_single_tree(node):
    """Analyze a single DSL tree and return statistics"""
    if node is None:
        return {'total_nodes': 0, 'abstraction_nodes': 0, 'unique_patterns': set(), 'node_breakdown': {}}

    def _traverse_count(node, counts):
        """Recursive traversal to count nodes"""
        if not hasattr(node, "serialize") and not isinstance(node, Abstraction):
            return

        # --- THIS IS THE FIX ---
        # Count the current node, *whether it is an Abstraction or not*
        counts['total_nodes'] += 1
        # --- END FIX ---

        node_type = type(node).__name__
        if isinstance(node, Abstraction):
            node_type = f"Abs({node.pattern_name})"
            counts['abstraction_nodes'] += 1
            counts['unique_patterns'].add(node.pattern_name)
        # else:
            # The 'total_nodes' line above replaces this
            # counts['total_nodes'] += 1

        # Count in breakdown
        counts['node_breakdown'][node_type] = counts['node_breakdown'].get(node_type, 0) + 1

        # Get children
        if isinstance(node, Abstraction):
            children = node.children
        elif hasattr(node, "serialize"):
            _, (_, children) = node.serialize()
        else:
            children = []

        # Recursively count children
        for child in children:
            if hasattr(child, "serialize") or isinstance(child, Abstraction):
                _traverse_count(child, counts)

    counts = {
        'total_nodes': 0,
        'abstraction_nodes': 0,
        'unique_patterns': set(),
        'node_breakdown': {}
    }
    _traverse_count(node, counts)

    # Add a count for total *concrete* nodes
    counts['concrete_nodes'] = counts['total_nodes'] - counts['abstraction_nodes']
    return counts

def run_comparative_analysis(analysis_title, l0_data, l1_data, l2_data):
    """Comprehensive analysis comparing L0, L1, and L2 for ALL chairs"""
    print("=" * 80)
    print(f"COMPREHENSIVE ABSTRACTION ANALYSIS: {analysis_title}")
    print(f"Dataset Size: {len(l0_data)} chairs")
    print("=" * 80)

    total_stats = {
        'L0': {'total_nodes': 0, 'abstraction_nodes': 0, 'concrete_nodes': 0, 'unique_patterns': set(), 'node_breakdown': {}},
        'L1': {'total_nodes': 0, 'abstraction_nodes': 0, 'concrete_nodes': 0, 'unique_patterns': set(), 'node_breakdown': {}},
        'L2': {'total_nodes': 0, 'abstraction_nodes': 0, 'concrete_nodes': 0, 'unique_patterns': set(), 'node_breakdown': {}}
    }

    for chair_id in tqdm(l0_data.keys(), desc=f"Analyzing {analysis_title}"):
        try:
            l0_stats = analyze_single_tree(l0_data[chair_id]["dsl"])
            l1_stats = analyze_single_tree(l1_data[chair_id]["dsl"])
            l2_stats = analyze_single_tree(l2_data[chair_id]["dsl"])

            for level_stats, level_key in zip([l0_stats, l1_stats, l2_stats], ['L0', 'L1', 'L2']):
                total_stats[level_key]['total_nodes'] += level_stats['total_nodes']
                total_stats[level_key]['abstraction_nodes'] += level_stats['abstraction_nodes']
                total_stats[level_key]['concrete_nodes'] += level_stats['concrete_nodes']
                total_stats[level_key]['unique_patterns'].update(level_stats['unique_patterns'])
                for node_type, count in level_stats['node_breakdown'].items():
                    total_stats[level_key]['node_breakdown'][node_type] = \
                        total_stats[level_key]['node_breakdown'].get(node_type, 0) + count
        except Exception as e:
            print(f"Error analyzing chair {chair_id}: {e}")
            continue

    num_chairs = len(l0_data)
    analysis_results = {}

    for level in ['L0', 'L1', 'L2']:
        stats = total_stats[level]
        analysis_results[level] = {
            'total_nodes': stats['total_nodes'],
            'avg_nodes_per_chair': stats['total_nodes'] / num_chairs,
            'abstraction_nodes': stats['abstraction_nodes'],
            'concrete_nodes': stats['concrete_nodes'],
            'avg_abstraction_nodes_per_chair': stats['abstraction_nodes'] / num_chairs,
            'unique_patterns': len(stats['unique_patterns']),
            'unique_patterns_list': sorted(list(stats['unique_patterns'])),
            'node_breakdown': dict(sorted(stats['node_breakdown'].items(), key=lambda x: x[1], reverse=True)),
            'abstraction_ratio': stats['abstraction_nodes'] / stats['total_nodes'] if stats['total_nodes'] > 0 else 0
        }

    # Calculate reduction percentages
    try:
        analysis_results['L1']['node_reduction_vs_L0'] = (
            (analysis_results['L0']['total_nodes'] - analysis_results['L1']['total_nodes']) / 
            analysis_results['L0']['total_nodes']
        )
        analysis_results['L2']['node_reduction_vs_L0'] = (
            (analysis_results['L0']['total_nodes'] - analysis_results['L2']['total_nodes']) / 
            analysis_results['L0']['total_nodes']
        )
        analysis_results['L2']['node_reduction_vs_L1'] = (
            (analysis_results['L1']['total_nodes'] - analysis_results['L2']['total_nodes']) / 
            analysis_results['L1']['total_nodes']
        )
    except ZeroDivisionError:
        debug_error("Zero division during analysis, some data may be missing.")

    # --- Display Results ---
    print(f"\nDATASET OVERVIEW:")
    print(f"  • Total chairs analyzed: {num_chairs}")
    print(f"  • L0 (Original): {analysis_results['L0']['total_nodes']:,} total nodes")
    print(f"  • L1 (Abstracted): {analysis_results['L1']['total_nodes']:,} total nodes")
    print(f"  • L2 (Hierarchical): {analysis_results['L2']['total_nodes']:,} total nodes")

    print(f"\nNODE COUNT REDUCTION:")
    print(f"  • L1 vs L0: {analysis_results['L1']['node_reduction_vs_L0']:.1%}")
    print(f"  • L2 vs L0: {analysis_results['L2']['node_reduction_vs_L0']:.1%}")
    print(f"  • L2 vs L1: {analysis_results['L2']['node_reduction_vs_L1']:.1%}")

    print(f"\nAVG NODES PER CHAIR:")
    print(f"  • L0: {analysis_results['L0']['avg_nodes_per_chair']:.1f} nodes/chair")
    print(f"  • L1: {analysis_results['L1']['avg_nodes_per_chair']:.1f} nodes/chair")
    print(f"  • L2: {analysis_results['L2']['avg_nodes_per_chair']:.1f} nodes/chair")

    print(f"\nAVG ABSTRACTION NODES PER CHAIR:")
    print(f"  • L1: {analysis_results['L1']['avg_abstraction_nodes_per_chair']:.1f} abs_nodes/chair")
    print(f"  • L2: {analysis_results['L2']['avg_abstraction_nodes_per_chair']:.1f} abs_nodes/chair")

    for level in ['L0', 'L1', 'L2']:
        print(f"\n{level} - TOP 10 NODE TYPES (Total: {analysis_results[level]['total_nodes']:,}):")
        node_breakdown = analysis_results[level]['node_breakdown']
        top_nodes = list(node_breakdown.items())[:10]
        for node_type, count in top_nodes:
            percentage = (count / analysis_results[level]['total_nodes']) * 100
            print(f"  • {node_type:<30} {count:>7,} ({percentage:5.1f}%)")

    return analysis_results


# In[19]:


## 18. Run   AUTOENCODER Analysis

# Check if all required datasets are loaded
if 'all_dsl_shapes' in locals() and \
   'all_abstracted_shapes_L1_AE' in locals() and \
   'all_abstracted_shapes_L2_AE' in locals():

    analysis_results_AE = run_comparative_analysis(
        analysis_title="AUTOENCODER (AE) PIPELINE",
        l0_data=all_dsl_shapes,
        l1_data=all_abstracted_shapes_L1_AE,
        l2_data=all_abstracted_shapes_L2_AE
    )
else:
    debug_error("Cannot run AE analysis: L0, L1-AE, or L2-AE datasets are not loaded.")


# In[20]:


## 19. Run   PCA Analysis

# Check if all required datasets are loaded
if 'all_dsl_shapes' in locals() and \
   'all_abstracted_shapes_L1_PCA' in locals() and \
   'all_abstracted_shapes_L2_PCA' in locals():

    analysis_results_PCA = run_comparative_analysis(
        analysis_title="PCA PIPELINE",
        l0_data=all_dsl_shapes,
        l1_data=all_abstracted_shapes_L1_PCA,
        l2_data=all_abstracted_shapes_L2_PCA
    )
else:
    debug_error("Cannot run PCA analysis: L0, L1-PCA, or L2-PCA datasets are not loaded.")


# In[21]:


## 20. Final Visual & Geometric Comparison

def plot_full_comparison(chair_key):
    """
    Plots the original, L1-AE-expanded, L2-AE-expanded, L1-PCA-expanded,
    and L2-PCA-expanded versions of a single chair and calculates Chamfer distances.
    """

    print("=" * 80)
    print(f"RUNNING FULL COMPARISON FOR: {chair_key}")
    print("=" * 80)

    try:
        # --- 1. Get all DSL versions ---
        original_dsl = all_dsl_shapes[chair_key]["dsl"]
        l1_dsl_ae = all_abstracted_shapes_L1_AE[chair_key]["dsl"]
        l2_dsl_ae = all_abstracted_shapes_L2_AE[chair_key]["dsl"]
        l1_dsl_pca = all_abstracted_shapes_L1_PCA[chair_key]["dsl"]
        l2_dsl_pca = all_abstracted_shapes_L2_PCA[chair_key]["dsl"]

        # --- 2. Expand all abstracted trees back to L0 ---
        debug_info("Expanding AE trees...")
        l1_expanded_ae = expand_l1_to_l0(l1_dsl_ae, singleton_models_L1_AE, pair_models_L1_AE)
        l2_expanded_ae = expand_l1_to_l0(
            expand_l2_to_l1(l2_dsl_ae, singleton_models_L1_AE, pair_models_L1_AE, singleton_models_L2_AE, pair_models_L2_AE),
            singleton_models_L1_AE, pair_models_L1_AE
        )

        debug_info("Expanding PCA trees...")
        l1_expanded_pca = expand_l1_to_l0(l1_dsl_pca, singleton_models_L1_PCA, pair_models_L1_PCA)
        l2_expanded_pca = expand_l1_to_l0(
            expand_l2_to_l1(l2_dsl_pca, singleton_models_L1_PCA, pair_models_L1_PCA, singleton_models_L2_PCA, pair_models_L2_PCA),
            singleton_models_L1_PCA, pair_models_L1_PCA
        )
        debug_success("All trees expanded.")

        # --- 3. Generate Point Clouds ---
        debug_info("Generating point clouds...")
        pc_original = get_point_cloud_from_dsl(original_dsl, points_per_box=500)
        pc_l1_ae_exp = get_point_cloud_from_dsl(l1_expanded_ae, points_per_box=500)
        pc_l2_ae_exp = get_point_cloud_from_dsl(l2_expanded_ae, points_per_box=500)
        pc_l1_pca_exp = get_point_cloud_from_dsl(l1_expanded_pca, points_per_box=500)
        pc_l2_pca_exp = get_point_cloud_from_dsl(l2_expanded_pca, points_per_box=500)
        debug_success("Point clouds generated.")

        # --- 4. Calculate Chamfer Distances ---
        chamfer_l1_ae = calculate_chamfer_distance(pc_original, pc_l1_ae_exp)
        chamfer_l2_ae = calculate_chamfer_distance(pc_original, pc_l2_ae_exp)
        chamfer_l1_pca = calculate_chamfer_distance(pc_original, pc_l1_pca_exp)
        chamfer_l2_pca = calculate_chamfer_distance(pc_original, pc_l2_pca_exp)

        print("\n--- GEOMETRIC VERIFICATION (Chamfer Distance vs. Original) ---")
        print(f"  L1 AE Expanded:   {chamfer_l1_ae:.8f}")
        print(f"  L2 AE Expanded:   {chamfer_l2_ae:.8f}")
        print(f"  L1 PCA Expanded:  {chamfer_l1_pca:.8f}")
        print(f"  L2 PCA Expanded:  {chamfer_l2_pca:.8f}")

        # --- 5. Plot DSLs ---
        plot_dsl_grid(
            [original_dsl, l1_expanded_ae, l2_expanded_ae, l1_expanded_pca, l2_expanded_pca],
            [
                f"{chair_key} (Original L0)", 
                "L1-AE (Expanded)", 
                "L2-AE (Expanded)",
                "L1-PCA (Expanded)",
                "L2-PCA (Expanded)"
            ],
            grid_cols=3,
            grid_title=f"Full Expansion Comparison for {chair_key}"
        )

    except Exception as e:
        debug_error(f"Error processing chair {chair_key}: {e}")

# --- Run the comparison ---
# Use a chair key you know is interesting, or pick one at random
sample_key = "Chair_5689.json" 
if sample_key not in all_dsl_shapes:
    sample_key = random.choice(list(all_dsl_shapes.keys()))

plot_full_comparison(sample_key)


# In[22]:


# [Cell 31, at the end]

# --- Run the comparison ---

# 1. Get a sample of 10 chairs
if 'all_dsl_shapes' in locals() and all_dsl_shapes:
    num_to_sample = min(10, len(all_dsl_shapes))
    debug_info(f"Sampling {num_to_sample} chairs for full comparison...")

    try:
        sample_keys = random.sample(list(all_dsl_shapes.keys()), num_to_sample)
    except Exception as e:
        debug_error(f"Failed to get random sample: {e}")
        sample_keys = [] # Prevent crash

    # 2. Loop through each sampled chair and plot its comparison grid
    for key in sample_keys:
        plot_full_comparison(key)

else:
    debug_error("Cannot run comparison: `all_dsl_shapes` is not defined or is empty.")


# In[23]:


## 22. Full 2D Geometric Comparison (Heatmap)

from IPython.display import display
import pandas as pd
from tqdm.auto import tqdm
import itertools
import numpy as np

# --- Set a local limit for this 2D analysis ---
# Set to None to run on ALL shapes (if pickle doesn't exist)
ANALYSIS_LIMIT_2D = 1000
# ---

# --- Define save path for results ---
results_pickle_file = saved_directory / "chamfer_results_2d.pkl"
# ---

chamfer_results_2d = []

# --- NEW: Check if results already exist ---
if results_pickle_file.exists():
    debug_success(f"Loading pre-computed 2D chamfer results from: {results_pickle_file}")
    with open(results_pickle_file, "rb") as f:
        chamfer_results_2d = pickle.load(f)
else:
    # --- Run full computation if no file is found ---
    debug_error(
        f"Starting 2D geometric analysis. This will be 4x slower than the previous analysis."
    )
    if ANALYSIS_LIMIT_2D is None:
        debug_error("ANALYSIS_LIMIT_2D is None. This will run on all shapes and take a very long time.")

    # --- Check for the *base* dataset ---
    if 'all_dsl_shapes' in locals() and all_dsl_shapes:

        # Apply the local limit
        all_keys = all_dsl_shapes.keys()
        if ANALYSIS_LIMIT_2D is not None:
            keys_to_process = list(itertools.islice(all_keys, ANALYSIS_LIMIT_2D))
            debug_info(f"Applying local ANALYSIS_LIMIT: Processing {len(keys_to_process)} shapes.")
        else:
            keys_to_process = list(all_keys)
            debug_info(f"No local limit: Processing all {len(keys_to_process)} shapes.")

        # Loop through the *limited* set of keys
        for chair_key in tqdm(keys_to_process, desc="Calculating 2D Chamfer Matrix"):

            result_row = {"chair_key": chair_key}
            pc_original = None

            try:
                # 1. Get Original DSL and Point Cloud (Once)
                original_dsl = all_dsl_shapes[chair_key]["dsl"]
                pc_original = get_point_cloud_from_dsl(original_dsl, points_per_box=200)

                # 2. Process AE Models
                try:
                    l1_dsl_ae = all_abstracted_shapes_L1_AE[chair_key]["dsl"]
                    l2_dsl_ae = all_abstracted_shapes_L2_AE[chair_key]["dsl"]

                    l1_expanded_ae = expand_l1_to_l0(l1_dsl_ae, singleton_models_L1_AE, pair_models_L1_AE)
                    l2_expanded_ae = expand_l1_to_l0(
                        expand_l2_to_l1(l2_dsl_ae, singleton_models_L1_AE, pair_models_L1_AE, singleton_models_L2_AE, pair_models_L2_AE),
                        singleton_models_L1_AE, pair_models_L1_AE
                    )

                    pc_l1_ae_exp = get_point_cloud_from_dsl(l1_expanded_ae, points_per_box=200)
                    pc_l2_ae_exp = get_point_cloud_from_dsl(l2_expanded_ae, points_per_box=200)

                    result_row["chamfer_l1_ae"] = calculate_chamfer_distance(pc_original, pc_l1_ae_exp)
                    result_row["chamfer_l2_ae"] = calculate_chamfer_distance(pc_original, pc_l2_ae_exp)

                except Exception as e_ae:
                    debug_error(f"Failed AE processing for {chair_key}: {e_ae}")
                    result_row["chamfer_l1_ae"] = np.nan
                    result_row["chamfer_l2_ae"] = np.nan

                # 3. Process PCA Models
                try:
                    l1_dsl_pca = all_abstracted_shapes_L1_PCA[chair_key]["dsl"]
                    l2_dsl_pca = all_abstracted_shapes_L2_PCA[chair_key]["dsl"]

                    l1_expanded_pca = expand_l1_to_l0(l1_dsl_pca, singleton_models_L1_PCA, pair_models_L1_PCA)
                    l2_expanded_pca = expand_l1_to_l0(
                        expand_l2_to_l1(l2_dsl_pca, singleton_models_L1_PCA, pair_models_L1_PCA, singleton_models_L2_PCA, pair_models_L2_PCA),
                        singleton_models_L1_PCA, pair_models_L1_PCA
                    )

                    pc_l1_pca_exp = get_point_cloud_from_dsl(l1_expanded_pca, points_per_box=200)
                    pc_l2_pca_exp = get_point_cloud_from_dsl(l2_expanded_pca, points_per_box=200)

                    result_row["chamfer_l1_pca"] = calculate_chamfer_distance(pc_original, pc_l1_pca_exp)
                    result_row["chamfer_l2_pca"] = calculate_chamfer_distance(pc_original, pc_l2_pca_exp)

                except Exception as e_pca:
                    debug_error(f"Failed PCA processing for {chair_key}: {e_pca}")
                    result_row["chamfer_l1_pca"] = np.nan
                    result_row["chamfer_l2_pca"] = np.nan

            except Exception as e_outer:
                debug_error(f"Failed to process {chair_key} entirely: {e_outer}")
                result_row.update({
                    "chamfer_l1_ae": np.nan, "chamfer_l2_ae": np.nan,
                    "chamfer_l1_pca": np.nan, "chamfer_l2_pca": np.nan
                })

            chamfer_results_2d.append(result_row)

        # --- NEW: Save the results to the pickle file ---
        if chamfer_results_2d:
            with open(results_pickle_file, "wb") as f:
                pickle.dump(chamfer_results_2d, f)
            debug_success(f"Saved computed 2D chamfer results to: {results_pickle_file}")

    else:
        debug_error("Cannot run 2D analysis: `all_dsl_shapes` is not defined or is empty.")


# --- 4. Create and display the color-coded matrix (this part runs either way) ---
if chamfer_results_2d:
    debug_success("Generating 2D color matrix (heatmap)...")

    df = pd.DataFrame(chamfer_results_2d)
    df.set_index('chair_key', inplace=True)

    # Apply color-coding by row (axis=1)
    df_styled = df.style.background_gradient(
        cmap='Reds',
        axis=1  # Applies the heatmap per-row
    ).format(
        '{:.8f}'
    ).set_caption(
        "Chamfer Distance (Error) vs. Original L0 Shape (Heatmap applied per-row)"
    )

    print("\n" + "="*80)
    print("2D Geometric Reconstruction Error Matrix")
    print("For each row (chair), lighter is better, darker is worse.")
    print("="*80)
    display(df_styled)

else:
    debug_error("No 2D chamfer results were loaded or computed.")


# In[24]:


## 23. Filtered 2D Geometric Comparison (Only Abstracted Shapes)

from IPython.display import display
import pandas as pd
import numpy as np

# --- Define the results file to load ---
results_pickle_file = saved_directory / "chamfer_results_2d.pkl"
# ---

# Check that the required 'analyze_single_tree' function is available
if 'analyze_single_tree' not in locals():
    debug_error("The helper function 'analyze_single_tree' (from Cell 17) is not defined.")
    debug_error("Please run Cell 17 first.")

# Check that the results file exists
elif not results_pickle_file.exists():
    debug_error(f"Results file not found: {results_pickle_file}")
    debug_error("Please run Cell 22 to generate the results first.")

else:
    debug_success(f"Loading pre-computed 2D chamfer results from: {results_pickle_file}")
    with open(results_pickle_file, "rb") as f:
        chamfer_results_2d = pickle.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(chamfer_results_2d)

    # This list will hold our filter data
    abstraction_check = []

    debug_info("Checking all shapes for any applied abstractions...")

    # Iterate through the DataFrame to check for abstractions
    for chair_key in df['chair_key']:
        total_abstractions = 0
        try:
            # Check L1-AE
            l1_ae_dsl = all_abstracted_shapes_L1_AE[chair_key]["dsl"]
            total_abstractions += analyze_single_tree(l1_ae_dsl)['abstraction_nodes']

            # Check L2-AE
            l2_ae_dsl = all_abstracted_shapes_L2_AE[chair_key]["dsl"]
            total_abstractions += analyze_single_tree(l2_ae_dsl)['abstraction_nodes']

            # Check L1-PCA
            l1_pca_dsl = all_abstracted_shapes_L1_PCA[chair_key]["dsl"]
            total_abstractions += analyze_single_tree(l1_pca_dsl)['abstraction_nodes']

            # Check L2-PCA
            l2_pca_dsl = all_abstracted_shapes_L2_PCA[chair_key]["dsl"]
            total_abstractions += analyze_single_tree(l2_pca_dsl)['abstraction_nodes']

        except KeyError:
            debug_error(f"Could not find chair_key {chair_key} in one of the datasets. Skipping.")
        except Exception as e:
            debug_error(f"An error occurred processing {chair_key}: {e}")

        abstraction_check.append(total_abstractions > 0)

    # Add the check as a new column
    df['has_any_abstraction'] = abstraction_check

    # Filter the DataFrame
    df_filtered = df[df['has_any_abstraction'] == True].copy()

    if df_filtered.empty:
        debug_error("No shapes with abstractions were found in the results.")
    else:
        debug_success(
            f"Filtering complete. Showing {len(df_filtered)} "
            f"of {len(df)} shapes that have at least one abstraction."
        )

        # Drop the helper column and set index for styling
        df_to_style = df_filtered.drop(columns=['has_any_abstraction']).set_index('chair_key')

        # --- Create and display the color-coded matrix ---
        df_styled = df_to_style.style.background_gradient(
            cmap='Reds',
            axis=1  # Applies the heatmap per-row
        ).format(
            '{:.8f}'
        ).set_caption(
            "FILTERED: Chamfer Distance (Error) vs. Original (Heatmap applied per-row)"
        )

        print("\n" + "="*80)
        print("2D Geometric Error (Only Shapes with >= 1 Abstraction)")
        print("For each row (chair), lighter is better, darker is worse.")
        print("="*80)
        display(df_styled)


# In[25]:


plot_dsl_with_k3d(all_abstracted_shapes_L2_PCA["Chair_1133.json"]["dsl"])


# In[ ]:




