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
dataset_directory = src_dir / "abstractionssymh" / "dataset"
saved_directory = src_dir / "abstractionssymh" / "saved"

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
from abstractionssymh.debug_utils import debug_info, debug_error, debug_success
from abstractionssymh.data_loader import parse_json_to_dsl
from abstractionssymh.plot_utils import plot_dsl_with_k3d, plot_dsl_grid
from abstractionssymh.dsl_utils import collect_singleton_and_pair_data
from abstractionssymh.abstraction_utils import (
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


# In[5]:


## 3. Load L0 Chair Dataset

# Set a limit on the number of chairs to load for faster testing.
# Set to None to load all chairs.
CHAIR_LIMIT = 1000

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


# In[6]:


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


# In[7]:


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


## 7.   VAE Pipeline: Train/Load L1 Models

# --- Setup VAE Directories (Local to this cell) ---
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


# In[10]:


singleton_models_L1_VAE.keys()


# In[11]:


pair_models_L1_VAE.keys()


# In[21]:


## 8.   VAE Pipeline: Create L1-VAE Abstracted Dataset

debug_info("--- Creating new L1-VAE Abstracted Dataset ---")

all_abstracted_shapes_L1_VAE = {}
pickle_file_L1_VAE = saved_directory / "all_abstracted_shapes_L1_VAE.pkl"

if pickle_file_L1_VAE.exists():
    with open(pickle_file_L1_VAE, "rb") as f:
        all_abstracted_shapes_L1_VAE = pickle.load(f)
    debug_success(f"Loaded {len(all_abstracted_shapes_L1_VAE)} L1-VAE abstracted shapes.")
else:
    for filename, data in tqdm(all_dsl_shapes.items(), desc="Integrating L1-VAE Abstractions"):

        # We integrate using the VAE models trained in Step 7
        abstracted_dsl = integrate_abstractions(
            data["dsl"],
            singleton_models_L1_VAE,
            pair_models_L1_VAE,
            # IMPORTANT: VAEs are probabilistic and often have higher MSE than standard AEs.
            # We relax the threshold from 0.02 to 0.08 (or 0.10) to allow abstractions to apply.
            error_threshold=2.0, 
            detailed_debug=True
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


# In[25]:


all_abstracted_shapes_L1_VAE


# In[29]:


debug_info("--- Extracting L2-AE Parameters ---")
combined_singletons_detailed_L2_AE = defaultdict(list)
combined_pairs_detailed_L2_AE = defaultdict(list)

for filename, data in tqdm(all_abstracted_shapes_L1_VAE.items(), desc="Aggregating L2-AE Params"):
    # Process Singletons
    for p_name, p_lists in data["singleton_params"].items():
        if "Box" in p_name: continue
        for p_list in p_lists or []:
            # FIX: Key includes the length of the parameters to prevent dimension mismatch
            key = f"{p_name}_dim{len(p_list)}"
            combined_singletons_detailed_L2_AE[key].append({'params': p_list})

    # Process Pairs
    for p_name, p_lists in data["pair_params"].items():
        if "Box" in p_name: continue
        for p_list in p_lists or []:
            # FIX: Key includes the length of the parameters
            key = f"{p_name}_dim{len(p_list)}"
            combined_pairs_detailed_L2_AE[key].append({'params': p_list})

# Prepare L2-AE Training Data
training_singleton_params_L2_AE = {
    k: [r['params'] for r in v] for k, v in combined_singletons_detailed_L2_AE.items() if v
}
training_pair_params_L2_AE = {
    k: [r['params'] for r in v] for k, v in combined_pairs_detailed_L2_AE.items() if v
}

debug_success(f"Found {len(training_singleton_params_L2_AE)} unique dimension-safe singleton and {len(training_pair_params_L2_AE)} pair patterns.")


# In[35]:


training_singleton_params_L2_AE.keys()


# In[36]:


training_pair_params_L2_AE.keys()


# In[31]:


debug_info("--- Starting L2-AE Abstraction Pipeline ---")
models_exist_L2_AE = any(saved_models_L2_AE_dir.glob('*.pth'))
singleton_models_L2_AE = {}
pair_models_L2_AE = {}

if models_exist_L2_AE:
    debug_info(f"--- L2 AE models found. Loading from {saved_models_L2_AE_dir} ---")
    # Load all available .pth files from the directory directly
    for model_path in saved_models_L2_AE_dir.glob('*.pth'):
        # Extract the name from the filename (e.g., "Translate_dim2_pth" -> "Translate_dim2")
        name = model_path.stem.replace("_pth", "")
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.eval()

        # Determine if it belongs in singleton or pair dictionary
        if any(name == k for k in training_singleton_params_L2_AE.keys()):
            singleton_models_L2_AE[name] = model
        elif any(name == k for k in training_pair_params_L2_AE.keys()):
            pair_models_L2_AE[name] = model
else:
    debug_info("--- No L2 AE models found. Starting training... ---")
    # find_abstractions will now receive cleanly separated dimensions
    singleton_models_L2_AE = find_abstractions(
        training_singleton_params_L2_AE, method='ae', structure_type="SINGLETONS_L2_AE", 
        min_examples=50, epochs=20, save_dir=saved_models_L2_AE_dir,
        plot_error_distribution=True
    )
    pair_models_L2_AE = find_abstractions(
        training_pair_params_L2_AE, method='ae', structure_type="PAIRS_L2_AE", 
        min_examples=50, epochs=20, save_dir=saved_models_L2_AE_dir,
        plot_error_distribution=True
    )

    # Save the newly trained models
    for name, model in singleton_models_L2_AE.items():
        torch.save(model, saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth"))
    for name, model in pair_models_L2_AE.items():
        torch.save(model, saved_models_L2_AE_dir / make_safe_filename(name, suffix="pth"))
    debug_success(f"Saved L2 AE models to {saved_models_L2_AE_dir}")

debug_success(f"--- L2 AE Workflow complete. {len(singleton_models_L2_AE)} models ready. ---")


# In[32]:


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


# In[34]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


plot_dsl_with_k3d(all_abstracted_shapes_L2_PCA["Chair_1133.json"]["dsl"])


# In[ ]:




