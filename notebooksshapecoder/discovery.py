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


# In[5]:


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


# In[7]:


shapes_l0["172_0_0"]


# In[8]:


from abstractionsshapecoder.dsl_utils import collect_singleton_and_pair_data
from abstractionsshapecoder.debug_utils import debug_info, debug_success

# 1. Isolate the DSL objects from the dictionary
all_l0_dsl_trees = [data['dsl'] for data in shapes_l0.values()]

debug_info(f"Extracting parameters from {len(all_l0_dsl_trees)} L0 shapes...")

# 2. Run the collection utility
# s_data_l0: { 'NodeName': [ [params], [params], ... ] }
# p_data_l0: { 'Parent(Child)': [ [params], [params], ... ] }
s_data_l0, p_data_l0 = collect_singleton_and_pair_data(all_l0_dsl_trees)

debug_success(f"Extraction complete.")
debug_info(f"Unique Singletons found: {len(s_data_l0)}")
debug_info(f"Unique Parent-Child Pairs found: {len(p_data_l0)}")


# In[10]:


import pandas as pd

# Create a summary for Singletons
singleton_summary = []
for node_type, param_list in s_data_l0.items():
    singleton_summary.append({
        "Node Type": node_type,
        "Total Occurrences": len(param_list),
        "Param Dimension": len(param_list[0]) if param_list else 0,
        "Example Params": param_list[0] if param_list else []
    })

# Create a summary for Pairs
pair_summary = []
for pair_name, param_list in p_data_l0.items():
    pair_summary.append({
        "Relationship": pair_name,
        "Total Occurrences": len(param_list),
        "Combined Dimension": len(param_list[0]) if param_list else 0
    })

# Convert to DataFrames for easy viewing
df_singletons = pd.DataFrame(singleton_summary).sort_values("Total Occurrences", ascending=False)
df_pairs = pd.DataFrame(pair_summary).sort_values("Total Occurrences", ascending=False)

print("=== L0 SINGLETON CENSUS ===")
display(df_singletons)

print("\n=== L0 PAIR CENSUS ===")
display(df_pairs)


# In[15]:


s_data_l0["Cuboid"]

