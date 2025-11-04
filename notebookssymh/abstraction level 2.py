#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 1. Setup Paths

import sys
import os
from pathlib import Path

# Add source directory to path
# Assumes this notebook is in a 'notebooks' folder, and 'src' is one level up
current_path = Path.cwd()
base_project_dir = current_path.parent
src_dir = base_project_dir / "src"

if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Define key directories
dataset_directory = src_dir / "abstractionssymh" / "dataset"
saved_directory = src_dir / "abstractionssymh" / "saved"
saved_models_L1_dir = saved_directory / "models_L1"
saved_models_L2_dir = saved_directory / "models_L2"

# Create directories
saved_directory.mkdir(parents=True, exist_ok=True)
saved_models_L1_dir.mkdir(parents=True, exist_ok=True)
saved_models_L2_dir.mkdir(parents=True, exist_ok=True)

print(f"Base project directory: {base_project_dir}")
print(f"Source directory: {src_dir}")
print(f"Saved directory: {saved_directory}")
print(f"L1 Models directory: {saved_models_L1_dir}")
print(f"L2 Models directory: {saved_models_L2_dir}")


# In[2]:


## 2. Imports

import pickle
import random
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.mixture import GaussianMixture

# Project-specific imports
from abstractionssymh.debug_utils import debug_info, debug_error, debug_success
from abstractionssymh.data_loader import parse_json_to_dsl
from abstractionssymh.plot_utils import plot_dsl_with_k3d, plot_dsl_grid
from abstractionssymh.dsl_utils import collect_singleton_and_pair_data, find_all_subtrees
from abstractionssymh.abstraction_utils import (
    find_abstractions, 
    integrate_abstractions, 
    Abstraction,
    Autoencoder, 
    DEVICE,
    make_safe_filename
)
from abstractionssymh.dsl_nodes import Box # Used for type checking

# Helper function from your notebook (cell [18]) for clustering
def cluster_key_with_gmm(data_dict, pattern_key, max_clusters=8, verbose=True):
    """
    Clusters the points for a given key using a Gaussian Mixture Model (GMM).
    Automatically finds the optimal number of clusters using BIC.
    """
    data_dict = data_dict.copy()
    if pattern_key not in data_dict:
        debug_error(f"'{pattern_key}' not found in dictionary.")
        return data_dict

    records = data_dict.pop(pattern_key)
    df = pd.DataFrame(records)
    param_df = pd.DataFrame(df['params'].to_list())
    param_values = param_df.values

    if len(param_values) < max_clusters:
        debug_info(f"Warning: Not enough data for '{pattern_key}' ({len(param_values)} points) to test {max_clusters} clusters. Skipping.")
        data_dict[pattern_key] = records # Add it back unchanged
        return data_dict

    n_components_range = range(1, max_clusters + 1)
    bic_scores = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(param_values)
        bic_scores.append(gmm.bic(param_values))

    optimal_n_components = np.argmin(bic_scores) + 1
    if verbose:
        debug_info(f"Clustering '{pattern_key}': Optimal clusters = {optimal_n_components}")

    final_gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
    cluster_labels = final_gmm.fit_predict(param_values)

    for cluster_id in range(optimal_n_components):
        cluster_records = df.iloc[cluster_labels == cluster_id].to_dict(orient='records')
        new_key = f"{pattern_key}_cluster{cluster_id}"
        data_dict[new_key] = cluster_records
        if verbose:
            debug_info(f"  -> {new_key}: {len(cluster_records)} points")

    return data_dict

print("All libraries and helpers imported.")


# In[3]:


# In[3]:
## 3. Load Chairs (L1 Dataset)

# --- CONFIGURATION ---
# Set a limit on the number of chairs to load for faster testing.
# Set to None to load all chairs.
CHAIR_LIMIT = None 
# ---

pickle_file = saved_directory / "all_dsl_shapes.pkl"
all_dsl_shapes = {} # This will be our final, limited dictionary
full_dsl_shapes = {} # This will hold the complete dataset

if pickle_file.exists():
    debug_info(f"Loading L1 DSL shapes from pickle: {pickle_file}")
    with open(pickle_file, "rb") as f:
        # Load the full dictionary first
        full_dsl_shapes = pickle.load(f)
    debug_success(f"Loaded {len(full_dsl_shapes)} total shapes from pickle.")
else:
    debug_info(f"Pickle file not found: {pickle_file}")
    debug_info("--- Generating new pickle from JSON files ---")

    # --- Step 1: Define directory ---
    # This path is defined in cell [1]
    chair_directory = dataset_directory / "Chair"
    if not chair_directory.exists():
        debug_error(f"Chair dataset directory not found at: {chair_directory}")

    # --- Step 2: Load DSL objects from JSON files ---
    debug_info("--- Step 2.1: Loading DSL objects from JSON files ---")
    json_files = sorted(list(chair_directory.glob("*.json"))) # Convert generator to list
    if not json_files:
        debug_error(f"No JSON files found in {chair_directory}")

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

    debug_info(f"Loaded {len(full_dsl_shapes)} DSL shapes into memory.")

    # --- Step 3: Collect parameters ---
    debug_info("\n--- Step 2.2: Collecting parameters for each shape ---")
    for name, data in tqdm(full_dsl_shapes.items(), desc="Collecting parameters"):
        dsl_obj = data["dsl"]
        # Use the imported function
        singletons, pairs = collect_singleton_and_pair_data([dsl_obj])
        data["singleton_params"] = singletons
        data["pair_params"] = pairs

    debug_success("Populated singleton and pair parameters for all shapes.")

    # --- Step 4: Save to pickle ---
    try:
        with open(pickle_file, "wb") as f:
            pickle.dump(full_dsl_shapes, f)
        debug_success(f"Saved all {len(full_dsl_shapes)} shapes to {pickle_file}")
    except Exception as e:
        debug_error(f"Failed to save pickle file: {e}")

# --- APPLY CHAIR_LIMIT (runs after loading OR generating) ---
if full_dsl_shapes:
    if CHAIR_LIMIT is not None and len(full_dsl_shapes) > CHAIR_LIMIT:
        debug_info(f"Limiting dataset to {CHAIR_LIMIT} chairs (out of {len(full_dsl_shapes)}).")
        # Convert to list, slice, and convert back to dict
        limited_items = list(full_dsl_shapes.items())[:CHAIR_LIMIT]
        all_dsl_shapes = dict(limited_items)
    else:
        debug_info(f"Using all {len(full_dsl_shapes)} loaded chairs (no limit set or limit not exceeded).")
        all_dsl_shapes = full_dsl_shapes

    debug_success(f"Final dataset size: {len(all_dsl_shapes)} shapes.")
else:
    debug_error("No shapes were loaded or generated.")

# Display a sample
if all_dsl_shapes:
    sample_key = list(all_dsl_shapes.keys())[0]
    print(f"\nSample shape '{sample_key}' keys: {all_dsl_shapes[sample_key].keys()}")
    # print(f"Sample DSL:\n{all_dsl_shapes[sample_key]['dsl']}")


# In[4]:


## 4. Display Chairs (Sample)

if all_dsl_shapes:
    debug_info("Displaying a random 3x3 grid of loaded chairs...")

    # Get 9 random chairs
    num_to_sample = min(9, len(all_dsl_shapes))
    random_keys = random.sample(list(all_dsl_shapes.keys()), num_to_sample)

    dsl_list_for_grid = [all_dsl_shapes[key]['dsl'] for key in random_keys]
    name_list_for_grid = [key.replace('.json', '') for key in random_keys]

    # Plot the grid (will display inline in a notebook)
    plot_dsl_grid(
        dsl_list_for_grid,
        name_list_for_grid,
        save_path=None, # Set to None to display inline
        grid_cols=3,
        grid_title="Random Sample of Loaded Chairs"
    )
else:
    debug_error("No DSL shapes loaded, cannot display.")


# In[5]:


## 5. Extract L1 Structures & Parameters

# This cell corresponds to cell [20] in your notebook
combined_singletons_detailed_pickle = saved_directory / "combined_singletons_detailed_L1.pkl"
combined_pairs_detailed_pickle = saved_directory / "combined_pairs_detailed_L1.pkl"

if combined_singletons_detailed_pickle.exists() and combined_pairs_detailed_pickle.exists():
    debug_info("Loading L1 detailed dictionaries from pickle...")
    with open(combined_singletons_detailed_pickle, "rb") as f:
        combined_singletons_detailed_L1 = pickle.load(f)
    with open(combined_pairs_detailed_pickle, "rb") as f:
        combined_pairs_detailed_L1 = pickle.load(f)
    debug_success("Loaded L1 detailed singleton and pair dictionaries.")
else:
    debug_info("Building L1 detailed dictionaries for singletons and pairs...")
    combined_singletons_detailed_L1 = {}
    combined_pairs_detailed_L1 = {}

    for filename, data in tqdm(all_dsl_shapes.items(), desc="Aggregating L1 Parameters"):
        # SINGLETON parameters
        for pattern_name, param_lists in data["singleton_params"].items():
            for param_list in param_lists or []:
                combined_singletons_detailed_L1.setdefault(pattern_name, []).append({
                    'file': filename,
                    'params': param_list
                })
        # PAIR parameters
        for pattern_name, param_lists in data["pair_params"].items():
            for param_list in param_lists or []:
                combined_pairs_detailed_L1.setdefault(pattern_name, []).append({
                    'file': filename,
                    'params': param_list
                })

    # Apply GMM Clustering
    debug_info("\nApplying GMM Clustering to L1 'Rotate' pattern...")
    combined_singletons_detailed_L1 = cluster_key_with_gmm(
        combined_singletons_detailed_L1, 'Rotate', max_clusters=8, verbose=True
    )
    debug_info("\nApplying GMM Clustering to L1 'Scale' pattern...")
    combined_singletons_detailed_L1 = cluster_key_with_gmm(
        combined_singletons_detailed_L1, 'Scale', max_clusters=8, verbose=True
    )

    # Save to pickle
    with open(combined_singletons_detailed_pickle, "wb") as f:
        pickle.dump(combined_singletons_detailed_L1, f)
    with open(combined_pairs_detailed_pickle, "wb") as f:
        pickle.dump(combined_pairs_detailed_L1, f)
    debug_success("Built, clustered, and saved L1 parameter structures.")

print(f"\nL1 Singleton Keys: {list(combined_singletons_detailed_L1.keys())}")
print(f"L1 Pair Keys: {list(combined_pairs_detailed_L1.keys())}")


# In[6]:


## 6. Prepare L1 Training Data

# This cell corresponds to cell [24] in your notebook
debug_info("--- Preparing L1 data for model training ---")

training_singleton_params_L1 = {}
for pattern_name, records in combined_singletons_detailed_L1.items():
    if records:
        training_singleton_params_L1[pattern_name] = [rec['params'] for rec in records]

training_pair_params_L1 = {}
for pattern_name, records in combined_pairs_detailed_L1.items():
    if records:
        training_pair_params_L1[pattern_name] = [rec['params'] for rec in records]

debug_success(f"L1 Data flattened for AE training.")
print(f"Found {len(training_singleton_params_L1)} L1 singleton patterns to train.")
print(f"Found {len(training_pair_params_L1)} L1 pair patterns to train.")


# In[7]:


# In[7]:
## 7. Train L1 Autoencoders

# This cell corresponds to cell [27] in your notebook
models_exist_L1 = any(saved_models_L1_dir.glob('*.pth'))
singleton_models_L1 = {}
pair_models_L1 = {}

if models_exist_L1:
    debug_info(f"--- L1 models found. Loading from {saved_models_L1_dir} ---")

    # Load L1 Singleton Models
    for name, params in training_singleton_params_L1.items():
        if not params: continue
        num_params = len(params[0])
        if num_params <= 1: continue
        # --- FIX: Add suffix="pth" ---
        save_file = saved_models_L1_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            try:
                model = Autoencoder(num_params, max(1, num_params - 1)).to(DEVICE)
                model.load_state_dict(torch.load(save_file, map_location=DEVICE))
                model.eval()
                singleton_models_L1[name] = model
                debug_success(f"Loaded L1 singleton model for '{name}'")
            except Exception as e:
                debug_error(f"Failed to load L1 model '{name}': {e}")

    # Load L1 Pair Models
    for name, params in training_pair_params_L1.items():
        if not params: continue
        num_params = len(params[0])
        if num_params <= 1: continue
        # --- FIX: Add suffix="pth" ---
        save_file = saved_models_L1_dir / make_safe_filename(name, suffix="pth")
        if save_file.is_file():
            try:
                model = Autoencoder(num_params, max(1, num_params - 1)).to(DEVICE)
                model.load_state_dict(torch.load(save_file, map_location=DEVICE))
                model.eval()
                pair_models_L1[name] = model
                debug_success(f"Loaded L1 pair model for '{name}'")
            except Exception as e:
                debug_error(f"Failed to load L1 model '{name}': {e}")
else:
    debug_info("--- No L1 models found. Starting training process... ---")

    debug_info("Training L1 singleton models...")
    singleton_models_L1 = find_abstractions(
        training_singleton_params_L1, 
        structure_type="SINGLETONS_L1", 
        min_examples=50, 
        epochs=20
    )

    debug_info("Training L1 pair models...")
    pair_models_L1 = find_abstractions(
        training_pair_params_L1, 
        structure_type="PAIRS_L1", 
        min_examples=50, 
        epochs=20
    )

    # Save L1 Models
    for name, model in singleton_models_L1.items():
        # --- FIX: Add suffix="pth" ---
        save_path = saved_models_L1_dir / make_safe_filename(name, suffix="pth")
        torch.save(model.state_dict(), save_path)
    for name, model in pair_models_L1.items():
        # --- FIX: Add suffix="pth" ---
        save_path = saved_models_L1_dir / make_safe_filename(name, suffix="pth")
        torch.save(model.state_dict(), save_path)
    debug_success(f"Saved L1 models to {saved_models_L1_dir}")

debug_success(f"--- L1 Workflow complete. {len(singleton_models_L1)} singleton and {len(pair_models_L1)} pair models are ready. ---")


# In[8]:


## 8. Integrate L1 Abstractions (Example)

# This cell corresponds to cell [32] in your notebook
if all_dsl_shapes and (singleton_models_L1 or pair_models_L1):
    debug_info("--- Testing L1 Abstraction Integration ---")

    # Pick a sample chair
    sample_key = "Chair_274.json" # Use a known complex chair
    if sample_key not in all_dsl_shapes:
        sample_key = list(all_dsl_shapes.keys())[0] # Fallback

    original_dsl = all_dsl_shapes[sample_key]["dsl"]

    # Run the integration
    abstracted_dsl_L1 = integrate_abstractions(
        original_dsl, 
        singleton_models_L1, 
        pair_models_L1, 
        error_threshold=0.01
    )

    print("\n--- ORIGINAL CHAIR DSL (L0) ---")
    print(original_dsl)

    print("\n--- ABSTRACTED CHAIR DSL (L1) ---")
    print(abstracted_dsl_L1)
else:
    debug_error("Cannot run integration: Missing DSL shapes or L1 models.")


# In[9]:


## 9. Create New Dataset with L1 Abstractions

debug_info("--- Creating new L1 Abstracted Dataset ---")
all_abstracted_shapes_L1 = {}
pickle_file_L1 = saved_directory / "all_abstracted_shapes_L1.pkl"

if pickle_file_L1.exists():
    debug_info(f"Loading L1 abstracted shapes from pickle: {pickle_file_L1}")
    with open(pickle_file_L1, "rb") as f:
        all_abstracted_shapes_L1 = pickle.load(f)
    debug_success(f"Loaded {len(all_abstracted_shapes_L1)} L1 abstracted shapes.")
else:
    if all_dsl_shapes and (singleton_models_L1 or pair_models_L1):
        for filename, data in tqdm(all_dsl_shapes.items(), desc="Integrating L1 Abstractions"):
            abstracted_dsl = integrate_abstractions(
                data["dsl"],
                singleton_models_L1,
                pair_models_L1,
                error_threshold=0.01
            )
            # We must also collect the parameters for the new abstracted tree
            # This is a critical step that was in your original cell [3] logic
            l1_singletons, l1_pairs = collect_singleton_and_pair_data([abstracted_dsl])

            all_abstracted_shapes_L1[filename] = {
                "dsl": abstracted_dsl,
                "singleton_params": l1_singletons,
                "pair_params": l1_pairs,
                "original_dsl": data["dsl"] # Keep a reference
            }

        # Save the new dataset
        with open(pickle_file_L1, "wb") as f:
            pickle.dump(all_abstracted_shapes_L1, f)
        debug_success(f"Created and saved {len(all_abstracted_shapes_L1)} L1 abstracted shapes to {pickle_file_L1}")
    else:
        debug_error("Cannot create L1 abstracted dataset: Missing original DSL shapes or L1 models.")

# Display a sample from the new dataset
if all_abstracted_shapes_L1:
    sample_key = list(all_abstracted_shapes_L1.keys())[0]
    print(f"\nSample L1 abstracted shape '{sample_key}' keys: {all_abstracted_shapes_L1[sample_key].keys()}")
    print("\nSample L1 Singleton Params (should include 'Abstraction' nodes):")
    print(all_abstracted_shapes_L1[sample_key]['singleton_params'].keys())


# In[10]:


all_abstracted_shapes_L1


# In[11]:


# In[10]:
## 10. Run Abstraction Pipeline Again (L2)

# Check if we have the L1 abstracted dataset
if not all_abstracted_shapes_L1:
    debug_error("Cannot start L2 pipeline: 'all_abstracted_shapes_L1' is empty.")
else:
    debug_info("--- 🚀 Starting Hierarchical Abstraction Pipeline (L2) ---")

    # --- 10a. Extract L2 Structures & Parameters ---
    debug_info("--- 10a. Extracting L2 Structures & Parameters ---")
    combined_singletons_detailed_L2 = {}
    combined_pairs_detailed_L2 = {}

    for filename, data in tqdm(all_abstracted_shapes_L1.items(), desc="Aggregating L2 Parameters"):
        # Parameters were already collected when creating the L1 dataset
        # We just need to aggregate them in the 'detailed' format
        for pattern_name, param_lists in data["singleton_params"].items():
            for param_list in param_lists or []:
                combined_singletons_detailed_L2.setdefault(pattern_name, []).append({
                    'file': filename,
                    'params': param_list
                })
        for pattern_name, param_lists in data["pair_params"].items():
            for param_list in param_lists or []:
                combined_pairs_detailed_L2.setdefault(pattern_name, []).append({
                    'file': filename,
                    'params': param_list
                })

    debug_success("Aggregated L2 parameters.")
    print(f"\nL2 Singleton Keys: {list(combined_singletons_detailed_L2.keys())}")
    print(f"L2 Pair Keys: {list(combined_pairs_detailed_L2.keys())}")

    # --- 10b. Prepare L2 Training Data ---
    debug_info("--- 10b. Preparing L2 Training Data ---")
    training_singleton_params_L2 = {}
    for pattern_name, records in combined_singletons_detailed_L2.items():
        if records:
            training_singleton_params_L2[pattern_name] = [rec['params'] for rec in records]

    training_pair_params_L2 = {}
    for pattern_name, records in combined_pairs_detailed_L2.items():
        if records:
            training_pair_params_L2[pattern_name] = [rec['params'] for rec in records]

    debug_success(f"L2 Data flattened for AE training.")
    print(f"Found {len(training_singleton_params_L2)} L2 singleton patterns to train.")
    print(f"Found {len(training_pair_params_L2)} L2 pair patterns to train.")

    # --- 10c. Train L2 Autoencoders ---
    debug_info("--- 10c. Loading or Training L2 Autoencoders ---")

    # --- FIX: ADDED LOADING LOGIC ---
    models_exist_L2 = any(saved_models_L2_dir.glob('*.pth'))
    singleton_models_L2 = {}
    pair_models_L2 = {}

    if models_exist_L2:
        debug_info(f"--- L2 models found. Loading from {saved_models_L2_dir} ---")

        # Load L2 Singleton Models
        for name, params in training_singleton_params_L2.items():
            if not params: continue
            num_params = len(params[0])
            if num_params <= 1: continue
            save_file = saved_models_L2_dir / make_safe_filename(name, suffix="pth")
            if save_file.is_file():
                try:
                    model = Autoencoder(num_params, max(1, num_params - 1)).to(DEVICE)
                    model.load_state_dict(torch.load(save_file, map_location=DEVICE))
                    model.eval()
                    singleton_models_L2[name] = model
                    debug_success(f"Loaded L2 singleton model for '{name}'")
                except Exception as e:
                    debug_error(f"Failed to load L2 model '{name}': {e}")

        # Load L2 Pair Models
        for name, params in training_pair_params_L2.items():
            if not params: continue
            num_params = len(params[0])
            if num_params <= 1: continue
            save_file = saved_models_L2_dir / make_safe_filename(name, suffix="pth")
            if save_file.is_file():
                try:
                    model = Autoencoder(num_params, max(1, num_params - 1)).to(DEVICE)
                    model.load_state_dict(torch.load(save_file, map_location=DEVICE))
                    model.eval()
                    pair_models_L2[name] = model
                    debug_success(f"Loaded L2 pair model for '{name}'")
                except Exception as e:
                    debug_error(f"Failed to load L2 model '{name}': {e}")

    else:
        debug_info("--- No L2 models found. Starting training process... ---")

        debug_info("Training L2 singleton models...")
        singleton_models_L2 = find_abstractions(
            training_singleton_params_L2, 
            structure_type="SINGLETONS_L2", 
            min_examples=50, 
            epochs=20
        )

        debug_info("Training L2 pair models...")
        pair_models_L2 = find_abstractions(
            training_pair_params_L2, 
            structure_type="PAIRS_L2", 
            min_examples=50, 
            epochs=20
        )

        # Save L2 Models
        for name, model in singleton_models_L2.items():
            save_path = saved_models_L2_dir / make_safe_filename(name, suffix="pth")
            torch.save(model.state_dict(), save_path)
        for name, model in pair_models_L2.items():
            save_path = saved_models_L2_dir / make_safe_filename(name, suffix="pth")
            torch.save(model.state_dict(), save_path)
        debug_success(f"Saved L2 models to {saved_models_L2_dir}")

    # --- END FIX ---

    debug_success(f"--- L2 Workflow complete. {len(singleton_models_L2)} L2 singleton and {len(pair_models_L2)} L2 pair models are ready. ---")

    # --- 10d. Integrate L2 Abstractions (Example) ---
    debug_info("--- 10d. Testing L2 Abstraction Integration ---")

    sample_key = "Chair_274.json"
    if sample_key not in all_abstracted_shapes_L1:
        sample_key = list(all_abstracted_shapes_L1.keys())[0]

    dsl_L1 = all_abstracted_shapes_L1[sample_key]["dsl"]

    # Run integration using L2 models on the L1-abstracted tree
    abstracted_dsl_L2 = integrate_abstractions(
        dsl_L1, 
        singleton_models_L2, 
        pair_models_L2, 
        error_threshold=0.01
    )

    print(f"\n--- ABSTRACTED CHAIR DSL (L1) ---")
    print(dsl_L1)

    print(f"\n--- HIERARCHICAL ABSTRACTED CHAIR DSL (L2) ---")
    print(abstracted_dsl_L2)

    debug_info("Hierarchical abstraction pipeline finished.")


# In[12]:


## 11. Create New Dataset with L2 Abstractions

debug_info("--- Creating new L2 Abstracted Dataset ---")
all_abstracted_shapes_L2 = {}
pickle_file_L2 = saved_directory / "all_abstracted_shapes_L2.pkl"

if pickle_file_L2.exists():
    debug_info(f"Loading L2 abstracted shapes from pickle: {pickle_file_L2}")
    with open(pickle_file_L2, "rb") as f:
        all_abstracted_shapes_L2 = pickle.load(f)
    debug_success(f"Loaded {len(all_abstracted_shapes_L2)} L2 abstracted shapes.")
else:
    if all_abstracted_shapes_L1 and (singleton_models_L2 or pair_models_L2):
        debug_info("Generating L2 dataset from L1 dataset...")
        for filename, data in tqdm(all_abstracted_shapes_L1.items(), desc="Integrating L2 Abstractions"):
            # Get the L1 abstracted DSL
            dsl_L1 = data["dsl"]

            # Integrate L2 abstractions on top of L1 abstractions
            abstracted_dsl_L2 = integrate_abstractions(
                dsl_L1,
                singleton_models_L2,
                pair_models_L2,
                error_threshold=0.1,
                # detailed_debug=True
            )

            # Collect parameters from the new L2-abstracted tree
            l2_singletons, l2_pairs = collect_singleton_and_pair_data([abstracted_dsl_L2])

            all_abstracted_shapes_L2[filename] = {
                "dsl": abstracted_dsl_L2,
                "singleton_params": l2_singletons,
                "pair_params": l2_pairs,
                "original_dsl": data["original_dsl"] # Pass original reference
            }

        # Save the new L2 dataset
        with open(pickle_file_L2, "wb") as f:
            pickle.dump(all_abstracted_shapes_L2, f)
        debug_success(f"Created and saved {len(all_abstracted_shapes_L2)} L2 abstracted shapes to {pickle_file_L2}")
    else:
        debug_error("Cannot create L2 abstracted dataset: Missing L1 shapes or L2 models.")

# # Display a sample from the new L2 dataset
# if all_abstracted_shapes_L2:
#     sample_key = list(all_abstracted_shapes_L2.keys())[0]
#     print(f"\nSample L2 abstracted shape '{sample_key}' keys: {all_abstracted_shapes_L2[sample_key].keys()}")
#     print("\nSample L2 Singleton Params (should include 'Abs(Abs(...))' patterns):")
#     print(list(all_abstracted_shapes_L2[sample_key]['singleton_params'].keys()))

#     print("\n--- L2 Example ---")
#     print(all_abstracted_shapes_L2[sample_key]['dsl'])


# In[13]:


# In[14]:
## 14. Imports for Visualization

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


# In[14]:


# In[13]:
## 13. Calculate & Display Detailed Comparative Statistics

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
        print(f"\n--- 📊 Statistics for {name} ---")
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
    print(f"--- 📊 Statistics for {name} ---")
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
stats_L0 = calculate_dataset_stats(all_dsl_shapes, "L0 (Original) Dataset")
stats_L1 = calculate_dataset_stats(all_abstracted_shapes_L1, "L1 Abstracted Dataset")
stats_L2 = calculate_dataset_stats(all_abstracted_shapes_L2, "L2 Abstracted Dataset")

# Filter out None in case a dataset was empty
all_stats = [s for s in [stats_L0, stats_L1, stats_L2] if s]


# In[15]:


all_abstracted_shapes_L2["Chair_1.json"]["dsl"]


# In[16]:


plot_dsl_with_k3d(all_abstracted_shapes_L2["Chair_1.json"]["dsl"])


# In[ ]:




