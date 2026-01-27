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


shapes_l0["172_0_0"]


# In[5]:


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


# In[6]:


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


# In[7]:


import torch

class FloatOperation:
    def __init__(self, func, unary: bool = False):
        self.func = func
        self.unary = unary

LANGUAGE = {
    "binary_ops": {
        "Add": FloatOperation(lambda a, b: a + b),
        "Sub": FloatOperation(lambda a, b: a - b),
        "Mul": FloatOperation(lambda a, b: a * b),
    },
    "unary_ops": {"Neg": FloatOperation(lambda a: -a, unary=True)}
}

def log_best_match(method_name, target_name, expr, diff, tol, min_match):
    """Utility to print how close the best candidate came to success."""
    mean_err = diff.mean().item()
    match_frac = (diff < tol).float().mean().item()
    status = "[SUCCESS]" if match_frac >= min_match else "[FAILED ]"

    print(f"    {status} {method_name:10} | Target: {target_name:4} | Best: {expr:20} | "
          f"AvgErr: {mean_err:.4f} | Match: {match_frac*100:5.1f}% (Req: {min_match*100}%)")

def compare_with_tolerance(predicted, target, max_error, min_fraction):
    if not isinstance(predicted, torch.Tensor):
        predicted = torch.tensor(predicted, device=target.device, dtype=target.dtype)
    if predicted.dim() == 0: 
        predicted = predicted.expand_as(target)

    diff = torch.abs(predicted - target)
    success = (diff < max_error).float().mean().item() >= min_fraction
    return success, diff

def search_direct_variable(inp, target_name):
    best_diff, best_name = None, None
    for name, values in inp["variables"].items():
        success, diff = compare_with_tolerance(values, inp["target"], inp["tol"], inp["min_match"])
        if best_diff is None or diff.mean() < best_diff.mean():
            best_diff, best_name = diff, name
        if success:
            log_best_match("Direct", target_name, name, diff, inp["tol"], inp["min_match"])
            return name

    if best_name:
        log_best_match("Direct", target_name, best_name, best_diff, inp["tol"], inp["min_match"])
    return None

def search_constants(inp, target_name):
    best_diff, best_name = None, None
    for name, val in inp["simple_consts"].items():
        # Constants usually use 0.05 tol as per your original code
        success, diff = compare_with_tolerance(val, inp["target"], 0.05, inp["min_match"])
        if best_diff is None or diff.mean() < best_diff.mean():
            best_diff, best_name = diff, name
        if success:
            log_best_match("Const", target_name, name, diff, 0.05, inp["min_match"])
            return name

    if best_name:
        log_best_match("Const", target_name, best_name, best_diff, 0.05, inp["min_match"])
    return None

def search_binary_operations(inp, target_name):
    var_keys = list(inp["variables"].keys())
    const_keys = list(inp["extended_consts"].keys())
    all_keys = var_keys + const_keys

    value_stack = torch.stack(list(inp["variables"].values()) + 
                             [c.expand(len(inp["target"])) for c in inp["extended_consts"].values()])

    index_pairs = torch.cartesian_prod(torch.arange(len(inp["variables"])), torch.arange(len(all_keys)))

    global_best_fraction = -1.0
    global_best_expr = ""
    global_best_diff = None

    for op_name, op in LANGUAGE["binary_ops"].items():
        results = op.func(value_stack[index_pairs[:, 0]], value_stack[index_pairs[:, 1]])
        diffs = torch.abs(results - inp["target"].unsqueeze(0))
        fractions = (diffs < inp["tol"]).float().mean(dim=1)

        # Find best in this specific operation
        best_val, best_idx = torch.max(fractions, dim=0)
        idx = best_idx.item()
        expr = f"{op_name}({all_keys[index_pairs[idx][0]]}, {all_keys[index_pairs[idx][1]]})"

        if best_val.item() > global_best_fraction:
            global_best_fraction = best_val.item()
            global_best_expr = expr
            global_best_diff = diffs[idx]

        if best_val.item() >= inp["min_match"]:
            log_best_match("Binary", target_name, expr, diffs[idx], inp["tol"], inp["min_match"])
            return expr

    if global_best_expr:
        log_best_match("Binary", target_name, global_best_expr, global_best_diff, inp["tol"], inp["min_match"])
    return None

def discover_symbolic_rules(df, custom_tol=0.08, custom_min_match=0.85, custom_constants=None):
    rules = {}
    const_dict = {str(round(c, 2)): torch.tensor(c, dtype=torch.float32) for c in (custom_constants or [0.5, 1.0])}

    for target_col in df.columns:
        print(f"--- Searching for target: {target_col} ---")
        input_cols = [p for p in df.columns if p != target_col]
        target_tensor = torch.tensor(df[target_col].values, dtype=torch.float32)

        search_space = {
            "target": target_tensor,
            "variables": {inp: torch.tensor(df[inp].values, dtype=torch.float32) for inp in input_cols},
            "simple_consts": const_dict, "extended_consts": const_dict,
            "tol": custom_tol, "min_match": custom_min_match
        }

        found_for_col = False
        for search_fn, name in [(search_direct_variable, "Direct"), 
                                (search_constants, "Const"), 
                                (search_binary_operations, "Binary")]:
            found = search_fn(search_space, target_col)
            if found:
                rules[target_col] = found
                found_for_col = True
                break

        if not found_for_col:
            print(f"    [FAILED ] No rule discovered for {target_col}")

    return rules


# In[8]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# ==========================================
# 1. CLUSTERING & RULE DISCOVERY
# ==========================================
raw_cuboids = s_data_l0.get("Cuboid", [])
df_l0 = pd.DataFrame(raw_cuboids, columns=['sw', 'sh', 'sd'])

# Group cuboids into 4 functional types (e.g., Legs, Seats, Backs, Joints)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_l0['cluster'] = kmeans.fit_predict(df_l0)

cluster_grammars = {}
s_data_l1 = {"Cuboid": []}

print(f"{'CLUSTER':<8} | {'RULES FOUND'}")
print("-" * 65)

for c_id in range(4):
    subset = df_l0[df_l0['cluster'] == c_id].drop(columns=['cluster'])

    # Auto-harvest potential constants (modes) for this specific cluster
    modes = [float(stats.mode(np.round(subset[col].values, 2), keepdims=True)[0][0]) for col in subset.columns]

    # Discover Symbolic Rules (Symmetry or Constants)
    # Using your optimized thresholds: 5% tolerance and 40% minimum match
    rules = discover_symbolic_rules(
        subset, 
        custom_constants=list(set(modes + [0.0, 1.0])), 
        custom_tol=0.05, 
        custom_min_match=0.40
    )
    cluster_grammars[c_id] = rules
    print(f"{c_id:<8} | {rules}")

# ==========================================
# 2. ABSTRACTION (COMPRESSION TO L1)
# ==========================================
for _, row in df_l0.iterrows():
    c_id = int(row['cluster'])
    rules = cluster_grammars[c_id]

    # Compress: Keep only dimensions that ARE NOT explained by a rule
    latent = [row[v] for v in ['sw', 'sh', 'sd'] if v not in rules]

    # Fallback: preserve at least one dimension to keep the instance unique
    if not latent: 
        latent = [row['sw']] 

    s_data_l1["Cuboid"].append({"c": c_id, "l": [round(x, 4) for x in latent]})

# ==========================================
# 3. RECONSTRUCTION (DECOMPRESSION TO L0)
# ==========================================
reconstructed_rows = []

for node in s_data_l1["Cuboid"]:
    c_id = node["c"]
    latent = node["l"]
    rules = cluster_grammars[c_id]

    # Identify which variables are "Free" (Latent)
    latent_vars = [v for v in ['sw', 'sh', 'sd'] if v not in rules]
    if not latent_vars: 
        latent_vars = ['sw']

    # Map latent values back to their variable names
    recon = {var: val for var, val in zip(latent_vars, latent)}

    # Apply rules to fill in the missing/fixed dimensions
    for target, expr in rules.items():
        if expr in recon: 
            # Alias/Symmetry rule (e.g., sd = sw)
            recon[target] = recon[expr]
        else: 
            # Constant rule (e.g., sh = 0.05)
            try:
                recon[target] = float(expr)
            except ValueError:
                recon[target] = 0.05 # Default safety fallback

    reconstructed_rows.append([recon.get('sw', 0), recon.get('sh', 0), recon.get('sd', 0)])

# ==========================================
# 4. FINAL PERFORMANCE METRICS
# ==========================================
df_recon = pd.DataFrame(reconstructed_rows, columns=['sw_r', 'sh_r', 'sd_r'])

# Parameter Savings
total_l0_params = len(df_l0) * 3
total_l1_params = sum(len(node["l"]) for node in s_data_l1["Cuboid"])
reduction = (1 - total_l1_params / total_l0_params) * 100

# Error Metric
mse = ((df_l0[['sw', 'sh', 'sd']].values - df_recon.values)**2).mean()

print("-" * 65)
print(f"L0 Parameters: {total_l0_params}")
print(f"L1 Parameters: {total_l1_params}")
print(f"Compression:   {reduction:.2f}%")
print(f"Reconstruction MSE: {mse:.6f}")


# In[9]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# ==========================================
# 1. CLUSTERING & RULE DISCOVERY (TRANSLATE)
# ==========================================
raw_translates = s_data_l0.get("Translate", [])
# Translation nodes use tx, ty, tz (X, Y, Z coordinates)
df_l0_t = pd.DataFrame(raw_translates, columns=['tx', 'ty', 'tz'])

# Cluster positions (helps group parts like "Left Legs" vs "Right Legs")
kmeans_t = KMeans(n_clusters=4, random_state=42, n_init=10)
df_l0_t['cluster'] = kmeans_t.fit_predict(df_l0_t)

cluster_grammars_t = {}
s_data_l1_t = {"Translate": []}

print(f"{'CLUSTER':<8} | {'RULES FOUND'}")
print("-" * 65)

for c_id in range(4):
    subset = df_l0_t[df_l0_t['cluster'] == c_id].drop(columns=['cluster'])

    # --- ENHANCED CONSTANT HARVESTING ---
    # We take top 5 frequent positions (tx, ty, or tz) to catch standard offsets
    dynamic_consts = [0.0] # 0 is a critical constant for center-alignment
    for col in subset.columns:
        top_vals = subset[col].round(2).value_counts().head(5).index.tolist()
        dynamic_consts.extend(top_vals)
    dynamic_consts = list(set([x for x in dynamic_consts if not np.isnan(x)]))

    # Discover Rules with slightly relaxed tolerance (0.06) to catch near-misses
    rules = discover_symbolic_rules(
        subset, 
        custom_constants=dynamic_consts, 
        custom_tol=0.06, 
        custom_min_match=0.40
    )
    cluster_grammars_t[c_id] = rules
    print(f"{c_id:<8} | {rules}")

# ==========================================
# 2. ABSTRACTION (L0 -> L1)
# ==========================================
for _, row in df_l0_t.iterrows():
    c_id = int(row['cluster'])
    rules = cluster_grammars_t[c_id]

    # Compress: tx, ty, tz are only latent if they aren't ruled
    latent = [row[v] for v in ['tx', 'ty', 'tz'] if v not in rules]

    if not latent: 
        latent = [row['ty']] # Default to vertical position if everything else is fixed

    s_data_l1_t["Translate"].append({"c": c_id, "l": [round(x, 4) for x in latent]})

# ==========================================
# 3. RECONSTRUCTION (L1 -> L0)
# ==========================================
reconstructed_t = []

for node in s_data_l1_t["Translate"]:
    c_id = node["c"]
    latent = node["l"]
    rules = cluster_grammars_t[c_id]

    latent_vars = [v for v in ['tx', 'ty', 'tz'] if v not in rules]
    if not latent_vars: latent_vars = ['ty']

    recon = {var: val for var, val in zip(latent_vars, latent)}

    for target, expr in rules.items():
        if expr in recon: # Alias/Symmetry (e.g. tz = tx)
            recon[target] = recon[expr]
        else: # Constant (e.g. ty = 0.45)
            try:
                recon[target] = float(expr)
            except:
                recon[target] = 0.0

    reconstructed_t.append([recon.get('tx', 0), recon.get('ty', 0), recon.get('tz', 0)])

# ==========================================
# 4. METRICS
# ==========================================
df_recon_t = pd.DataFrame(reconstructed_t, columns=['tx_r', 'ty_r', 'tz_r'])

total_l0_t = len(df_l0_t) * 3
total_l1_t = sum(len(node["l"]) for node in s_data_l1_t["Translate"])
reduction_t = (1 - total_l1_t / total_l0_t) * 100
mse_t = ((df_l0_t[['tx', 'ty', 'tz']].values - df_recon_t.values)**2).mean()

print("-" * 65)
print(f"L0 Parameters: {total_l0_t}")
print(f"L1 Parameters: {total_l1_t}")
print(f"Compression:   {reduction_t:.2f}%")
print(f"Reconstruction MSE: {mse_t:.6f}")


# In[10]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# ==========================================
# 1. CLUSTERING & RULE DISCOVERY (ROTATE)
# ==========================================
raw_rotates = s_data_l0.get("Rotate", [])
# Assuming Rotate has 4 parameters: qx, qy, qz, qw
df_l0_r = pd.DataFrame(raw_rotates, columns=['qx', 'qy', 'qz', 'qw'])

# Cluster rotations (groups parts with similar orientations)
kmeans_r = KMeans(n_clusters=4, random_state=42, n_init=10)
df_l0_r['cluster'] = kmeans_r.fit_predict(df_l0_r)

cluster_grammars_r = {}
s_data_l1_r = {"Rotate": []}

print(f"{'CLUSTER':<8} | {'RULES FOUND'}")
print("-" * 65)

for c_id in range(4):
    subset = df_l0_r[df_l0_r['cluster'] == c_id].drop(columns=['cluster'])

    # --- ROTATION SPECIFIC CONSTANTS ---
    # Common rotation values: 0, 1, 0.707 (45/90 deg in quaternions), -1
    dynamic_consts = [0.0, 1.0, -1.0, 0.707, -0.707] 
    for col in subset.columns:
        top_vals = subset[col].round(3).value_counts().head(5).index.tolist()
        dynamic_consts.extend(top_vals)
    dynamic_consts = list(set([x for x in dynamic_consts if not np.isnan(x)]))

    # Discover Rules
    # Using 0.06 tol to account for floating point precision in rotations
    rules = discover_symbolic_rules(
        subset, 
        custom_constants=dynamic_consts, 
        custom_tol=0.06, 
        custom_min_match=0.40
    )
    cluster_grammars_r[c_id] = rules
    print(f"{c_id:<8} | {rules}")

# ==========================================
# 2. ABSTRACTION (L0 -> L1)
# ==========================================
for _, row in df_l0_r.iterrows():
    c_id = int(row['cluster'])
    rules = cluster_grammars_r[c_id]

    # Compress: keep only the components that aren't fixed or aliased
    latent = [row[v] for v in ['qx', 'qy', 'qz', 'qw'] if v not in rules]

    # Fallback: if it's a perfect constant rotation, keep qw to represent the instance
    if not latent: 
        latent = [row['qw']] 

    s_data_l1_r["Rotate"].append({"c": c_id, "l": [round(x, 4) for x in latent]})

# ==========================================
# 3. RECONSTRUCTION (L1 -> L0)
# ==========================================
reconstructed_r = []

for node in s_data_l1_r["Rotate"]:
    c_id = node["c"]
    latent = node["l"]
    rules = cluster_grammars_r[c_id]

    latent_vars = [v for v in ['qx', 'qy', 'qz', 'qw'] if v not in rules]
    if not latent_vars: latent_vars = ['qw']

    recon = {var: val for var, val in zip(latent_vars, latent)}

    for target, expr in rules.items():
        if expr in recon:
            recon[target] = recon[expr]
        else:
            try:
                recon[target] = float(expr)
            except:
                # Default for Quaternions is usually (0,0,0,1)
                recon[target] = 1.0 if target == 'qw' else 0.0

    reconstructed_r.append([recon.get('qx', 0), recon.get('qy', 0), recon.get('qz', 0), recon.get('qw', 1)])

# ==========================================
# 4. METRICS
# ==========================================
df_recon_r = pd.DataFrame(reconstructed_r, columns=['qx_r', 'qy_r', 'qz_r', 'qw_r'])

total_l0_r = len(df_l0_r) * 4
total_l1_r = sum(len(node["l"]) for node in s_data_l1_r["Rotate"])
reduction_r = (1 - total_l1_r / total_l0_r) * 100
mse_r = ((df_l0_r[['qx', 'qy', 'qz', 'qw']].values - df_recon_r.values)**2).mean()

print("-" * 65)
print(f"L0 Parameters: {total_l0_r}")
print(f"L1 Parameters: {total_l1_r}")
print(f"Compression:   {reduction_r:.2f}%")
print(f"Reconstruction MSE: {mse_r:.6f}")


# In[11]:


import pandas as pd
import numpy as np

# ==========================================
# 1. AGGREGATE ACTUAL DATA
# ==========================================
def get_report_data(type_name, df_orig, s_data_l1_list, recon_list):
    # Calculate row-wise MSE
    orig_vals = df_orig.drop(columns=['cluster'], errors='ignore').values
    recon_vals = np.array(recon_list)
    row_mse = ((orig_vals - recon_vals)**2).mean(axis=1)

    # Prepare strings for display
    return pd.DataFrame({
        'Type': type_name,
        'Original L0': [list(np.round(x, 3)) for x in orig_vals],
        'Latent L1': [node['l'] for node in s_data_l1_list],
        'Reconstructed': [list(np.round(x, 3)) for x in recon_vals],
        'MSE Error': row_mse
    })

# Combine all results
report_cuboid = get_report_data("Cuboid", df_l0, s_data_l1["Cuboid"], reconstructed_rows)
report_trans = get_report_data("Translate", df_l0_t, s_data_l1_t["Translate"], reconstructed_t)
report_rotate = get_report_data("Rotate", df_l0_r, s_data_l1_r["Rotate"], reconstructed_r)

full_report_df = pd.concat([report_cuboid, report_trans, report_rotate]).reset_index(drop=True)

# ==========================================
# 2. FORMATTED REPORT & HEATMAP
# ==========================================
print("=== SYMBOLIC ABSTRACTION PERFORMANCE REPORT ===")

# Summary Statistics Table
summary_stats = pd.DataFrame({
    'Metric': ['Total L0 Params', 'Total L1 Params', 'Compression %', 'Global MSE'],
    'Value': [
        total_l0_params + total_l0_t + total_l0_r,
        total_l1_params + total_l1_t + total_l1_r,
        f"{((1 - (total_l1_params + total_l1_t + total_l1_r) / (total_l0_params + total_l0_t + total_l0_r)) * 100):.2f}%",
        f"{(mse + mse_t + mse_r) / 3:.6f}"
    ]
})
display(summary_stats)

# Styling Function for Error Highlighting
def error_heatmap(val):
    # Normalizing color: 0 error is white, high error is red
    # Adjust 'max_val' based on your typical error range (e.g., 0.1)
    color_intensity = min(int(255 * (val / 0.1)), 255) 
    inv_intensity = 255 - color_intensity
    return f'background-color: rgb(255, {inv_intensity}, {inv_intensity})'

# Display top 20 rows with the most interesting errors
print("\n--- TOP ERROR SAMPLES (MANUAL INSPECTION) ---")
styled_report = full_report_df.sort_values("MSE Error", ascending=False).head(25).style.applymap(
    error_heatmap, subset=['MSE Error']
)

display(styled_report)


# In[12]:


import pandas as pd
import numpy as np

# ==========================================
# 1. CLEAN AGGREGATION FUNCTION
# ==========================================
def get_report_data_clean(type_name, df_orig, s_data_l1_list, recon_list):
    # Calculate row-wise MSE
    orig_vals = df_orig.drop(columns=['cluster'], errors='ignore').values
    recon_vals = np.array(recon_list)
    row_mse = ((orig_vals - recon_vals)**2).mean(axis=1)

    # helper to force python floats and 2 decimal rounding
    def clean_list(arr):
        return [round(float(x), 2) for x in arr]

    # Prepare strings for display
    return pd.DataFrame({
        'Type': type_name,
        'Original L0': [clean_list(x) for x in orig_vals],
        'Latent L1': [clean_list(node['l']) for node in s_data_l1_list],
        'Reconstructed': [clean_list(x) for x in recon_vals],
        'MSE Error': row_mse
    })

# Combine results with rounding
report_cuboid = get_report_data_clean("Cuboid", df_l0, s_data_l1["Cuboid"], reconstructed_rows)
report_trans = get_report_data_clean("Translate", df_l0_t, s_data_l1_t["Translate"], reconstructed_t)
report_rotate = get_report_data_clean("Rotate", df_l0_r, s_data_l1_r["Rotate"], reconstructed_r)

full_report_df = pd.concat([report_cuboid, report_trans, report_rotate]).reset_index(drop=True)

# Display with Heatmap
print("=== SYMBOLIC ABSTRACTION PERFORMANCE REPORT (ROUNDED) ===")
styled_report = full_report_df.sort_values("MSE Error", ascending=False).head(25).style.applymap(
    error_heatmap, subset=['MSE Error']
)
display(styled_report)


# In[13]:


# Display top 20 rows with the most interesting errors
print("\n--- TOP ERROR SAMPLES (MANUAL INSPECTION) ---")
styled_report = full_report_df.sort_values("MSE Error", ascending=True).head(25).style.applymap(
    error_heatmap, subset=['MSE Error']
)

styled_report


# In[14]:


full_report_df


# In[15]:


p_data_l0.keys()


# In[16]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# The relationships found in your p_data_l0
pair_keys = ['Rotate(Rotate)', 'Rotate(Translate)', 'Translate(Cuboid)', 'Union(Rotate)', 'Union(Translate)']

pair_grammars = {}
s_data_l1_pairs = {}

print(f"{'RELATIONSHIP':<20} | {'CLUSTER':<8} | {'RULES FOUND'}")
print("-" * 85)

for p_key in pair_keys:
    raw_data = p_data_l0.get(p_key, [])
    if not raw_data:
        continue

    # Create column names based on the pair (e.g., p_qx, p_qy... c_tx, c_ty...)
    # We dynamically detect dimension based on the first entry
    parent_dim = 4 if "Rotate" in p_key.split('(')[0] else 3
    child_dim = len(raw_data[0]) - parent_dim

    p_cols = [f'p{i}' for i in range(parent_dim)]
    c_cols = [f'c{i}' for i in range(child_dim)]
    all_cols = p_cols + c_cols

    df_pair = pd.DataFrame(raw_data, columns=all_cols)

    # 1. Cluster the relationships
    n_clusters = min(3, len(df_pair)) 
    kmeans_p = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_pair['cluster'] = kmeans_p.fit_predict(df_pair)

    cluster_rules = {}
    s_data_l1_pairs[p_key] = []

    for c_id in range(n_clusters):
        subset = df_pair[df_pair['cluster'] == c_id].drop(columns=['cluster'])

        # 2. Harvest constants from both parent and child values
        dynamic_consts = [0.0, 0.5, 1.0, -1.0]
        for col in subset.columns:
            top_vals = subset[col].round(2).value_counts().head(3).index.tolist()
            dynamic_consts.extend(top_vals)
        dynamic_consts = list(set(dynamic_consts))

        # 3. Discover Rules (Cross-parameter relationships)
        rules = discover_symbolic_rules(
            subset, 
            custom_constants=dynamic_consts, 
            custom_tol=0.07, 
            custom_min_match=0.40
        )
        cluster_rules[c_id] = rules
        print(f"{p_key[:20]:<20} | {c_id:<8} | {rules}")

    pair_grammars[p_key] = cluster_rules


# In[17]:


pair_reports = []

for p_key, clusters in pair_grammars.items():
    raw_data = p_data_l0[p_key]
    parent_dim = 4 if "Rotate" in p_key.split('(')[0] else 3
    p_cols = [f'p{i}' for i in range(parent_dim)]
    c_cols = [f'c{i}' for i in range(len(raw_data[0]) - parent_dim)]
    all_cols = p_cols + c_cols

    df_pair = pd.DataFrame(raw_data, columns=all_cols)
    df_pair['cluster'] = KMeans(n_clusters=len(clusters), random_state=42, n_init=10).fit_predict(df_pair)

    recon_rows = []

    for i, row in df_pair.iterrows():
        c_id = int(row['cluster'])
        rules = clusters[c_id]

        # Compression: Only keep columns not ruled
        latent = [row[v] for v in all_cols if v not in rules]
        if not latent: latent = [row[all_cols[0]]]

        # Reconstruction logic
        recon = {v: row[v] if v not in rules else None for v in all_cols}
        # (Simplified recon for report mapping)
        for target, expr in rules.items():
            if expr in all_cols: recon[target] = row[expr]
            else: 
                try: recon[target] = float(expr)
                except: recon[target] = 0.0

        recon_rows.append([recon[v] for v in all_cols])

    # Calculate MSE for this relationship type
    orig_np = df_pair[all_cols].values
    recon_np = np.array(recon_rows)
    mse_p = ((orig_np - recon_np)**2).mean()

    pair_reports.append({
        'Relationship': p_key,
        'L0 Params': len(df_pair) * len(all_cols),
        'L1 Params': sum([len([v for v in all_cols if v not in clusters[int(df_pair.iloc[j]['cluster'])]]) for j in range(len(df_pair))]),
        'MSE': mse_p
    })

df_pair_report = pd.DataFrame(pair_reports)


# In[18]:


df_pair_report


# In[19]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# 1. Initialization
pair_keys = ['Rotate(Rotate)', 'Rotate(Translate)', 'Translate(Cuboid)', 'Union(Rotate)', 'Union(Translate)']
pair_grammars = {}
s_data_l1_pairs = {}
pair_reconstructions = {} # This fixes your NameError

print(f"{'RELATIONSHIP':<20} | {'CLUSTER':<8} | {'RULES FOUND'}")
print("-" * 85)

# 2. Discovery & Reconstruction Loop
for p_key in pair_keys:
    raw_data = p_data_l0.get(p_key, [])
    if not raw_data: continue

    # Setup dimensions
    parent_dim = 4 if "Rotate" in p_key.split('(')[0] else 3
    child_dim = len(raw_data[0]) - parent_dim
    all_cols = [f'p{i}' for i in range(parent_dim)] + [f'c{i}' for i in range(child_dim)]

    df_pair = pd.DataFrame(raw_data, columns=all_cols)
    n_clusters = min(3, len(df_pair)) 
    df_pair['cluster'] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(df_pair)

    cluster_rules = {}
    s_data_l1_pairs[p_key] = []
    recons_for_this_key = []

    for c_id in range(n_clusters):
        subset = df_pair[df_pair['cluster'] == c_id].drop(columns=['cluster'])

        # Harvest top constants
        dynamic_consts = [0.0, 0.5, 1.0, -1.0]
        for col in subset.columns:
            dynamic_consts.extend(subset[col].round(2).value_counts().head(3).index.tolist())

        # Discover Rules
        rules = discover_symbolic_rules(subset, custom_constants=list(set(dynamic_consts)), custom_tol=0.07, custom_min_match=0.40)
        cluster_rules[c_id] = rules
        print(f"{p_key[:20]:<20} | {c_id:<8} | {rules}")

    # 3. Apply Compression and Generate Reconstructions
    for _, row in df_pair.iterrows():
        c_id = int(row['cluster'])
        rules = cluster_rules[c_id]

        # L1 Latent
        latent = [row[v] for v in all_cols if v not in rules]
        if not latent: latent = [row[all_cols[0]]]
        s_data_l1_pairs[p_key].append({"c": c_id, "l": [round(x, 4) for x in latent]})

        # Reconstruct (Logic for the Report)
        latent_vars = [v for v in all_cols if v not in rules]
        if not latent_vars: latent_vars = [all_cols[0]]
        recon_dict = {var: val for var, val in zip(latent_vars, latent)}

        for target, expr in rules.items():
            if expr in recon_dict: recon_dict[target] = recon_dict[expr]
            else:
                try: recon_dict[target] = float(expr)
                except: recon_dict[target] = 0.0

        recons_for_this_key.append([recon_dict.get(v, 0.0) for v in all_cols])

    pair_reconstructions[p_key] = recons_for_this_key
    pair_grammars[p_key] = cluster_rules

# ==========================================
# 4. FINAL DETAILED REPORT
# ==========================================
pair_report_list = []
for p_key in pair_keys:
    if p_key not in pair_reconstructions: continue
    orig_np = np.array(p_data_l0[p_key])
    recon_np = np.array(pair_reconstructions[p_key])
    row_mse = ((orig_np - recon_np)**2).mean(axis=1)

    pair_report_list.append(pd.DataFrame({
        'Relationship': p_key,
        'Original L0': [list(np.round(x, 3)) for x in orig_np],
        'Latent L1': [node['l'] for node in s_data_l1_pairs[p_key]],
        'Reconstructed': [list(np.round(x, 3)) for x in recon_np],
        'MSE Error': row_mse
    }))

full_pair_report_df = pd.concat(pair_report_list).reset_index(drop=True)

def pair_error_heatmap(val):
    intensity = min(int(255 * (val / 0.1)), 255)
    return f'background-color: rgb(255, {255-intensity}, {255-intensity})'

display(full_pair_report_df.sort_values("MSE Error", ascending=False).head(25).style.applymap(
    pair_error_heatmap, subset=['MSE Error']
))


# In[20]:


# ==========================================
# 2. CLEAN AGGREGATION FOR PAIRS
# ==========================================
pair_report_list_clean = []

def clean_list(arr):
    return [round(float(x), 2) for x in arr]

for p_key in pair_keys:
    if p_key not in pair_reconstructions: continue

    orig_np = np.array(p_data_l0[p_key])
    recon_np = np.array(pair_reconstructions[p_key])
    row_mse = ((orig_np - recon_np)**2).mean(axis=1)

    pair_report_list_clean.append(pd.DataFrame({
        'Relationship': p_key,
        'Original L0': [clean_list(x) for x in orig_np],
        'Latent L1': [clean_list(node['l']) for node in s_data_l1_pairs[p_key]],
        'Reconstructed': [clean_list(x) for x in recon_np],
        'MSE Error': row_mse
    }))

full_pair_report_clean = pd.concat(pair_report_list_clean).reset_index(drop=True)

print("\n--- TOP PAIR ERROR SAMPLES (ROUNDED) ---")
styled_pair_report = full_pair_report_clean.sort_values("MSE Error", ascending=False).head(25).style.applymap(
    pair_error_heatmap, subset=['MSE Error']
)
display(styled_pair_report)


# In[21]:


full_report_df


# In[23]:


full_pair_report_clean


# In[ ]:




