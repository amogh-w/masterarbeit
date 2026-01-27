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


shapes_l0["172_0_0"]


# In[6]:


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


# In[7]:


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


# In[49]:


import torch
import pandas as pd
import itertools

# ==============================================================================
# 1. LANGUAGE DEFINITIONS
# ==============================================================================

class FloatOperation:
    def __init__(self, func, unary: bool = False):
        self.func = func
        self.unary = unary

LANGUAGE = {
    "binary_ops": {
        "Add": FloatOperation(lambda a, b: a + b),
        "Sub": FloatOperation(lambda a, b: a - b),
        "Mul": FloatOperation(lambda a, b: a * b),
        "Div": FloatOperation(lambda a, b: torch.div(a, b + 1e-6)),
    },
    "constants": [0.5, 1.0, 2.0, -1.0, 0.35, -0.31, 0.08, 0.05], 
}

UNARY_OPERATIONS = {
    "Neg": FloatOperation(lambda a: -a, unary=True),
    "Inv": FloatOperation(lambda a: torch.div(1.0, a + 1e-6), unary=True)
}

# ==============================================================================
# 2. DEBUG UTILITIES
# ==============================================================================

def log_best_match(method_name, expr, diff, tol, min_match):
    """Prints a debug line showing how close the best candidate came."""
    mean_err = diff.mean().item()
    matches = (diff < tol).float()
    match_frac = matches.mean().item()

    status = "❌" if match_frac < min_match else "✅"
    print(f"    [{method_name}] {status} Best: {expr} | Err: {mean_err:.4f} | Match: {match_frac*100:.1f}%")

# ==============================================================================
# 3. MATCHING UTILITIES
# ==============================================================================

def compare_with_tolerance(predicted, target, max_error, min_fraction, name="expr"):
    if not isinstance(predicted, torch.Tensor):
        predicted = torch.tensor(predicted, device=target.device, dtype=target.dtype)
    if predicted.dim() == 0 or (predicted.dim() == 1 and predicted.shape[0] == 1):
        predicted = predicted.expand_as(target)

    diff = torch.abs(predicted - target)
    matches = diff < max_error
    fraction = matches.sum().item() / max(1, len(target))

    # Return stats for debugging
    return (fraction >= min_fraction), torch.where(matches)[0].tolist(), diff

def batch_compare(pred_batches, target, max_error, min_fraction, index_pairs, variable_names, op_name):
    results = []
    diffs = torch.abs(pred_batches - target.unsqueeze(0))
    matches = diffs < max_error
    fractions = matches.sum(dim=1).float() / len(target)

    # Find the best one in the batch for debugging
    best_idx = torch.argmax(fractions).item()
    best_expr = f"{op_name}({variable_names[index_pairs[best_idx][0]]}, {variable_names[index_pairs[best_idx][1]]})"
    log_best_match("Binary", best_expr, diffs[best_idx], max_error, min_fraction)

    success_indices = torch.where(fractions >= min_fraction)[0]
    for idx in success_indices:
        match_idx_list = torch.where(matches[idx])[0].tolist()
        expr_vars = tuple(variable_names[j] for j in index_pairs[idx])
        results.append((expr_vars, match_idx_list))
    return results

# ==============================================================================
# 4. SEARCH FUNCTIONS
# ==============================================================================

def search_direct_variable(inp, return_indices=False):
    matches = []
    best_diff, best_name = None, ""
    for name, values in inp["variables"].items():
        success, idx, diff = compare_with_tolerance(values, inp["target"], inp["tol"], inp["min_match"])
        if best_diff is None or diff.mean() < best_diff.mean():
            best_diff, best_name = diff, name
        if success: matches.append((name, idx) if return_indices else name)

    if best_name: log_best_match("Direct", best_name, best_diff, inp["tol"], inp["min_match"])
    return matches

def search_constants(inp, return_indices=False):
    matches = []
    best_diff, best_name = None, ""
    for name, val in inp["simple_consts"].items():
        success, idx, diff = compare_with_tolerance(val, inp["target"], 0.05, inp["min_match"])
        if best_diff is None or diff.mean() < best_diff.mean():
            best_diff, best_name = diff, name
        if success: matches.append((name, idx) if return_indices else name)

    if best_name: log_best_match("Const", best_name, best_diff, 0.05, inp["min_match"])
    return matches

def search_unary_operations(inp, return_indices=False):
    matches = []
    for op_name, op in inp["unary_ops"].items():
        best_diff, best_var = None, ""
        for var_name, var_value in inp["variables"].items():
            prediction = op.func(var_value)
            success, idx, diff = compare_with_tolerance(prediction, inp["target"], inp["tol"], inp["min_match"])
            if best_diff is None or diff.mean() < best_diff.mean():
                best_diff, best_var = diff, var_name
            if success: matches.append((f"{op_name}({var_name})", idx) if return_indices else f"{op_name}({var_name})")
        if best_var: log_best_match("Unary", f"{op_name}({best_var})", best_diff, inp["tol"], inp["min_match"])
    return matches

def search_binary_operations(inp, return_indices=False):
    matches = []
    var_keys = list(inp["variables"].keys())
    const_keys = list(inp["extended_consts"].keys())
    all_keys = var_keys + const_keys

    var_values = list(inp["variables"].values())
    const_values = [c.expand(len(inp["target"])) for c in inp["extended_consts"].values()]
    value_stack = torch.stack(var_values + const_values)

    index_pairs = torch.cartesian_prod(torch.arange(len(var_values)), torch.arange(len(all_keys)))
    left_sides = value_stack[index_pairs[:, 0]]
    right_sides = value_stack[index_pairs[:, 1]]

    for op_name, op in inp["binary_ops"].items():
        results = op.func(left_sides, right_sides)
        comparisons = batch_compare(results, inp["target"], inp["tol"], inp["min_match"], index_pairs, all_keys, op_name)
        for (a, b), idx in comparisons:
            matches.append((f"{op_name}({a}, {b})", idx) if return_indices else f"{op_name}({a}, {b})")
    return matches

# ==============================================================================
# 5. MAIN DRIVER
# ==============================================================================

def discover_symbolic_rules(df: pd.DataFrame, custom_tol=None, custom_min_match=None, custom_constants=None):
    rules = {}
    param_names = list(df.columns)

    # Use user-provided constants or fall back to the defaults
    search_constants_list = custom_constants if custom_constants is not None else [-1., 0., 1., 0.35, -0.31, 0.08]

    for target_col in param_names:
        input_cols = [p for p in param_names if p != target_col]
        target_tensor = torch.tensor(df[target_col].values, dtype=torch.float32)

        # 1. Handle Tolerance
        if custom_tol is not None:
            auto_tol = custom_tol
        else:
            auto_tol = 0.06 if target_tensor.abs().max() < 2.0 else 0.15

        # 2. Handle Match Threshold
        min_match = custom_min_match if custom_min_match is not None else 0.85

        # Create the dictionary of constant tensors
        const_dict = {str(c): torch.tensor(c, dtype=torch.float32) for c in search_constants_list}

        search_space = {
            "target": target_tensor,
            "variables": {inp: torch.tensor(df[inp].values, dtype=torch.float32) for inp in input_cols},
            "simple_consts": const_dict,
            "extended_consts": const_dict, # Now using your custom constants for binary ops too
            "binary_ops": LANGUAGE["binary_ops"],
            "unary_ops": UNARY_OPERATIONS,
            "tol": auto_tol,
            "min_match": min_match
        }

        print(f"\n--- Searching: {target_col} | Tol: {auto_tol} | Match: {min_match*100}% | Consts: {search_constants_list} ---")

        search_methods = [
            search_direct_variable,
            search_constants,
            search_unary_operations,
            search_binary_operations
        ]

        found_rule = None
        for method in search_methods:
            matches = method(search_space, return_indices=True)
            if matches:
                expr, _ = matches[0]
                found_rule = expr
                print(f"✅ SUCCESS: {target_col} ≈ {expr}")
                break

        if not found_rule:
            print(f"❌ No rule found for '{target_col}'")
        else:
            rules[target_col] = found_rule

    return rules


# In[58]:


s_data_l0["Cuboid"]


# In[59]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# --- STEP A: Clustering ---
raw_cuboids = s_data_l0.get("Cuboid", [])
df_l0 = pd.DataFrame(raw_cuboids, columns=['sw', 'sh', 'sd'])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_l0['cluster'] = kmeans.fit_predict(df_l0)

# --- STEP B: Harvest Magic Constants ---
def get_magic_constants(subset):
    # Find the most frequent rounded values (Modes)
    sw_mode = stats.mode(np.round(subset['sw'].values, 2), keepdims=True)[0][0]
    sh_mode = stats.mode(np.round(subset['sh'].values, 2), keepdims=True)[0][0]
    sd_mode = stats.mode(np.round(subset['sd'].values, 2), keepdims=True)[0][0]
    return {'sw': sw_mode, 'sh': sh_mode, 'sd': sd_mode}

# --- STEP C: Compress to L1 ---
s_data_l1 = {"Cuboid": []}
cluster_profiles = {}

for c_id in range(4):
    subset = df_l0[df_l0['cluster'] == c_id]
    cluster_profiles[c_id] = get_magic_constants(subset)

for i, row in df_l0.iterrows():
    c_id = int(row['cluster'])

    # 3D -> 1D/2D Compression Logic based on your Discovered Grammar
    if c_id == 0: # Leg: 1D (Thickness)
        latent = [(row['sw'] + row['sd']) / 2]
    elif c_id == 1: # Seat: 1D (Size)
        latent = [(row['sw'] + row['sd']) / 2]
    elif c_id == 2: # Panel: 2D (Width, Height)
        latent = [row['sw'], row['sh']]
    else: # Joint: 1D (Uniform Scale)
        latent = [(row['sw'] + row['sh'] + row['sd']) / 3]

    s_data_l1["Cuboid"].append({
        "cluster": c_id,
        "latent": [round(x, 4) for x in latent]
    })

print(f"Successfully created s_data_l1 with {len(s_data_l1['Cuboid'])} compressed nodes.")


# In[60]:


s_data_l1


# In[61]:


# --- STEP D: Decompress L1 back to L0 ---
reconstructed_l0 = []

for node in s_data_l1["Cuboid"]:
    c_id = node["cluster"]
    latent = node["latent"]
    profile = cluster_profiles[c_id]

    # Reconstruct 3D vectors based on the Symbolic Grammar
    if c_id == 0: # Leg: sw=latent, sd=latent, sh=constant
        recon = [latent[0], profile['sh'], latent[0]]
    elif c_id == 1: # Seat: sw=latent, sd=latent, sh=constant
        recon = [latent[0], profile['sh'], latent[0]]
    elif c_id == 2: # Panel: sw=latent[0], sh=latent[1], sd=constant
        recon = [latent[0], latent[1], profile['sd']]
    else: # Joint: sw=sh=sd=latent
        recon = [latent[0], latent[0], latent[0]]

    reconstructed_l0.append(recon)

# --- STEP E: Compare Output ---
df_recon = pd.DataFrame(reconstructed_l0, columns=['sw_rec', 'sh_rec', 'sd_rec'])
comparison = pd.concat([df_l0[['sw', 'sh', 'sd']], df_recon], axis=1)

print("\n=== ROUND-TRIP COMPARISON (First 5 Nodes) ===")
print(comparison.head(5).round(3))

# Calculate Accuracy (MSE)
mse = ((df_l0[['sw', 'sh', 'sd']].values - df_recon.values)**2).mean()
print(f"\nReconstruction MSE: {mse:.6f}")
print("Note: A low MSE means the Symbolic Grammar effectively captured the design intent!")


# In[65]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# --- 1. PREP & CLUSTER ---
raw_cuboids = s_data_l0.get("Cuboid", [])
df_l0 = pd.DataFrame(raw_cuboids, columns=['sw', 'sh', 'sd'])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_l0['cluster'] = kmeans.fit_predict(df_l0)

# --- 2. HARVEST CONSTANTS ---
cluster_profiles = {}
for c_id in range(4):
    subset = df_l0[df_l0['cluster'] == c_id]
    if not subset.empty:
        cluster_profiles[c_id] = {
            'sh': float(stats.mode(np.round(subset['sh'].values, 2), keepdims=True)[0][0]),
            'sd': float(stats.mode(np.round(subset['sd'].values, 2), keepdims=True)[0][0])
        }

# --- 3. COMPRESS & PRINT ALL NODES ---
s_data_l1 = {"Cuboid": []}

print(f"{'IDX':<4} | {'TYPE':<8} | {'ORIGINAL (L0)':<20} | {'COMPRESSED (L1)':<15} | {'RECONSTRUCTED'}")
print("-" * 85)

for i, row in df_l0.iterrows():
    c_id = int(row['cluster'])
    sw, sh, sd = row['sw'], row['sh'], row['sd']

    # Apply Grammar Logic to get Latent Code
    if c_id in [0, 1]: # Leg/Seat (3D -> 1D)
        latent = [round((sw + sd) / 2, 4)]
        recon = [latent[0], cluster_profiles[c_id]['sh'], latent[0]]
        type_label = "Leg/Seat"
    elif c_id == 2:    # Panel (3D -> 2D)
        latent = [round(sw, 4), round(sh, 4)]
        recon = [latent[0], latent[1], cluster_profiles[c_id]['sd']]
        type_label = "Panel"
    else:              # Joint (3D -> 1D)
        latent = [round((sw + sh + sd) / 3, 4)]
        recon = [latent[0], latent[0], latent[0]]
        type_label = "Joint"

    # Save to the new L1 structure with consistent keys
    s_data_l1["Cuboid"].append({"c": c_id, "l": latent})

    # Print the row for inspection
    orig_str = f"[{sw:.2f}, {sh:.2f}, {sd:.2f}]"
    lat_str = str(latent)
    recon_str = f"[{recon[0]:.2f}, {recon[1]:.2f}, {recon[2]:.2f}]"
    print(f"{i:<4} | {type_label:<8} | {orig_str:<20} | {lat_str:<15} | {recon_str}")

# Final Metric
print("-" * 85)
print(f"Total Nodes Processed: {len(s_data_l1['Cuboid'])}")


# In[67]:


class SymbolicPartProgram:
    def __init__(self, cluster_id, grammar_rules):
        self.cluster_id = cluster_id
        self.rules = grammar_rules
        self.target_vars = ['sw', 'sh', 'sd']

        # 1. Identify which variables are actually "Free" (Latent)
        # A variable is latent if it has no rule OR if the rules are circular
        self.latent_vars = []
        for v in self.target_vars:
            if v not in self.rules:
                self.latent_vars.append(v)

        # If the grammar is circular (sw=sd, sd=sw), we must pick one to be latent
        if not self.latent_vars:
            # Default to the first variable in alphabetical order if all are ruled
            self.latent_vars = [min(self.rules.keys())]

    def compress(self, l0_row):
        return [l0_row[v] for v in self.latent_vars]

    def decompress(self, latent_code):
        # Start with latent values
        recon = {var: val for var, val in zip(self.latent_vars, latent_code)}

        # We iterate multiple times to resolve dependencies (Simple fixed-point iteration)
        for _ in range(3): 
            for var in self.target_vars:
                if var in recon: continue # Already solved
                if var not in self.rules: continue # No rule to apply

                rule = self.rules[var]

                # Case 1: Constant (e.g., '0.6')
                try:
                    recon[var] = float(rule)
                    continue
                except ValueError:
                    pass

                # Case 2: Direct Alias (e.g., 'sd')
                if rule in recon:
                    recon[var] = recon[rule]
                    continue

                # Case 3: Binary Operation (e.g., 'Add(sd, 0.5)')
                if 'Add' in str(rule):
                    parts = rule.split('(')[1].split(')')[0].split(', ')
                    a_name, b_val = parts[0], float(parts[1])
                    if a_name in recon:
                        recon[var] = recon[a_name] + b_val

        # Final safety check: if something didn't resolve, use a default 0.05
        return [recon.get(v, 0.05) for v in ['sw', 'sh', 'sd']]

# --- RE-RUN THE AUTO-PIPELINE ---
part_library = {c_id: SymbolicPartProgram(c_id, rules) for c_id, rules in final_grammar.items()}
s_data_l1 = {"Cuboid": []}
reconstructed_l0 = []

for i, row in df_l0.iterrows():
    c_id = int(row['cluster'])
    prog = part_library[c_id]

    latent = prog.compress(row)
    s_data_l1["Cuboid"].append({"c": c_id, "l": latent})

    recon = prog.decompress(latent)
    reconstructed_l0.append(recon)

print("✅ Automated Abstraction Successful!")


# In[68]:


reconstructed_l0


# In[69]:


import pandas as pd

# Create a summary dataframe for clear comparison
summary_data = []

for i in range(len(df_l0)):
    # Original L0
    orig = df_l0.iloc[i]

    # Compressed L1 (from our automated s_data_l1)
    l1_node = s_data_l1["Cuboid"][i]

    # Reconstructed L0
    recon = reconstructed_l0[i]

    summary_data.append({
        "ID": i,
        "Cluster": l1_node["c"],
        "Original [sw, sh, sd]": [round(orig['sw'], 3), round(orig['sh'], 3), round(orig['sd'], 3)],
        "Latent Code": l1_node["l"],
        "Reconstructed": [round(recon[0], 3), round(recon[1], 3), round(recon[2], 3)],
        "Dimensions Saved": 3 - len(l1_node["l"])
    })

df_summary = pd.DataFrame(summary_data)

# Display a slice of the table covering different clusters
print(f"{'IDX':<5} | {'TYPE':<10} | {'ORIGINAL':<22} | {'LATENT':<12} | {'RECONSTRUCTED'}")
print("-" * 85)
# Show a representative sample (first few nodes)
for idx, row in df_summary.head(15).iterrows():
    c_id = row['Cluster']
    type_map = {0: "Leg", 1: "Seat", 2: "Back", 3: "Joint"}
    label = type_map.get(c_id, f"C_{c_id}")

    print(f"{row['ID']:<5} | {label:<10} | {str(row['Original [sw, sh, sd]']):<22} | {str(row['Latent Code']):<12} | {str(row['Reconstructed'])}")

# Final Stats
total_params_l0 = len(df_l0) * 3
total_params_l1 = sum(len(node["l"]) for node in s_data_l1["Cuboid"])
compression_ratio = (1 - (total_params_l1 / total_params_l0)) * 100

print("-" * 85)
print(f"Total Parameters L0: {total_params_l0}")
print(f"Total Parameters L1: {total_params_l1}")
print(f"Overall Parameter Reduction: {compression_ratio:.2f}%")


# In[ ]:




