#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from pathlib import Path
import random

# Add source directory to path
sys.path.append(os.path.abspath("../src"))

# Import project utilities
from abstractionssymh.debug_utils import debug_info, debug_error, debug_success
from abstractionssymh.data_loader import parse_json_to_dsl
from abstractionssymh.plot_utils import plot_dsl_with_k3d
from abstractionssymh.dsl_utils import collect_singleton_and_pair_data
from abstractionssymh.abstraction_utils import find_abstractions, Abstraction, integrate_abstractions


# In[2]:


def get_dataset_directory():
    current_path = Path.cwd()
    base_project_dir = current_path.parent
    dataset_directory = base_project_dir / "src" / "abstractionssymh" / "dataset"

    debug_info("Current notebook location:", current_path)
    debug_info("Base project directory:", base_project_dir)
    debug_info("Target dataset directory:", dataset_directory)

    return base_project_dir, dataset_directory

base_project_dir, dataset_directory = get_dataset_directory()


# In[3]:


def load_chair_dsl(chair_directory, use_random=False):
    json_files = sorted(chair_directory.glob("*.json"))
    if not json_files:
        debug_error("No JSON files found in:", chair_directory)
        return None

    file_path = random.choice(json_files) if use_random else json_files[0]
    debug_success("Loading chair file:", file_path)

    json_content = file_path.read_text(encoding="utf-8")
    dsl_object = parse_json_to_dsl(json_content)

    debug_success("Successfully parsed DSL object")
    debug_info(dsl_object)
    return dsl_object

def plot_chair(dsl_obj):
    if dsl_obj is None:
        debug_error("No DSL object to plot.")
        return
    try:
        debug_info("Rendering DSL object with k3d...")
        plot_dsl_with_k3d(dsl_obj)
        debug_success("Plotting complete.")
    except Exception as e:
        debug_error("Failed to plot DSL object:", e)


# In[4]:


chair_directory = dataset_directory / "Chair"
dsl_object = load_chair_dsl(chair_directory, use_random=False)
plot_chair(dsl_object)


# In[5]:


import pickle
from pathlib import Path

saved_directory = base_project_dir / "src" / "abstractionssymh" / "saved"
saved_directory.mkdir(parents=True, exist_ok=True)

# File paths for pickled data
singleton_file = saved_directory / Path("singleton_params.pkl")
pair_file = saved_directory / Path("pair_params.pkl")


# In[6]:


# if singleton_file.is_file() and pair_file.is_file():
#     with open(singleton_file, "rb") as f:
#         singleton_params = pickle.load(f)
#     with open(pair_file, "rb") as f:
#         pair_params = pickle.load(f)
#     debug_success("Loaded singleton and pair parameters from files.")

#     num_singleton = sum(len(v) for v in singleton_params.values())
#     num_pair = sum(len(v) for v in pair_params.values())
#     debug_success(f"Loaded {num_singleton} singleton and {num_pair} pair parameter sets.")
# else:
#     debug_info("Pickle files not found. Uncomment the collection block to generate new data.")


# In[7]:


chair_directory = dataset_directory / "Chair"
json_files = sorted(list(chair_directory.glob("*.json")))

try:
    debug_info("Starting to load DSL shapes from JSON files...")
    all_dsl_shapes = [parse_json_to_dsl(Path(f).read_text()) for f in json_files[:500]]
    # all_dsl_shapes = [parse_json_to_dsl(Path(f).read_text()) for f in json_files]
    debug_success(f"Loaded {len(all_dsl_shapes)} shapes from dataset.")
except Exception as e:
    debug_error("Failed to load DSL shapes:", e)
    all_dsl_shapes = []


# In[8]:


# Collect data fresh
singleton_params, pair_params = collect_singleton_and_pair_data(all_dsl_shapes)

# Save to files
with open(singleton_file, "wb") as f:
    pickle.dump(singleton_params, f)
with open(pair_file, "wb") as f:
    pickle.dump(pair_params, f)
debug_success("Collected and saved new singleton and pair parameter sets.")

num_singleton = sum(len(v) for v in singleton_params.values())
num_pair = sum(len(v) for v in pair_params.values())
debug_success(f"Collected {num_singleton} singleton and {num_pair} pair parameter sets.")


# In[9]:


import numpy as np

for key in singleton_params.keys():
    print(f"{key}: {len(singleton_params[key])} samples, with size {np.array(singleton_params[key][0]).size}")

for key in pair_params.keys():
    print(f"{key}: {len(pair_params[key])} samples, with size {np.array(pair_params[key][0]).size}")


# In[ ]:


import numpy as np
import k3d
import hdbscan
import umap
from k3d.colormaps import matplotlib_color_maps

def run_hdbscan(data, min_cluster_size):
    print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}...")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Found {n_clusters} clusters and {n_noise} noise points.")
    return labels


def plot_clusters(data, labels, param_name):
    dim = data.shape[1]
    plot_title = f'HDBSCAN clusters for {param_name}'

    if dim == 3:
        positions_3d = data
    elif dim > 3:
        print(f"Data is {dim}D. Reducing to 3D with UMAP for visualization.")
        reducer = umap.UMAP(n_components=3)
        positions_3d = reducer.fit_transform(data)
        plot_title += ' (UMAP Projection)'
    else:
        print(f"Data is {dim}D. Cannot create a 3D visualization.")
        return

    # Create and display the k3d plot
    plot = k3d.plot(name=plot_title)
    points_plot = k3d.points(
        positions=positions_3d.astype(np.float32),
        point_size=0.05,
        attribute=labels.astype(np.float32),
        color_map=matplotlib_color_maps.Paired,
        color_range=[np.min(labels), np.max(labels)]
    )
    plot += points_plot
    plot.display()


def save_clusters_to_dict(original_data, labels, param_name, target_dict):
    unique_labels = set(labels)
    print(f"Saving clusters for {param_name}...")
    for cluster_id in unique_labels:
        if cluster_id == -1: # Ignore noise points
            continue

        mask = (labels == cluster_id)
        key = f"{param_name}_cluster{cluster_id}"
        target_dict[key] = original_data[mask]
        print(f"  - Saved {len(original_data[mask])} samples to '{key}'")


# In[23]:


singleton_params_clustered = {}


# --- Workflow for "Scale" (3D data) ---
print("--- Processing Scale ---")
scale_array = np.array(singleton_params["Scale"])
# Step 1: Cluster the data
scale_labels = run_hdbscan(scale_array, min_cluster_size=50)
# Step 2: Visualize the clusters
plot_clusters(scale_array, scale_labels, "Scale")
# Step 3: Save the results
save_clusters_to_dict(scale_array, scale_labels, "Scale", singleton_params_clustered)
print("-" * 30 + "\n")


# --- Workflow for "Rotate" (4D data) ---
print("--- Processing Rotate ---")
rotate_array = np.array(singleton_params["Rotate"])
# Step 1: Cluster the data
rotate_labels = run_hdbscan(rotate_array, min_cluster_size=30)
# Step 2: Visualize the clusters
plot_clusters(rotate_array, rotate_labels, "Rotate")
# Step 3: Save the results
save_clusters_to_dict(rotate_array, rotate_labels, "Rotate", singleton_params_clustered)
print("-" * 30 + "\n")


# --- Verify the final output ---
print("✅ Processing complete! Final clustered dictionary keys:")
print(list(singleton_params_clustered.keys()))


# In[24]:


singleton_params_clustered


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





# In[28]:


debug_info("Starting singleton model training...")
singleton_models = find_abstractions(singleton_params, structure_type="SINGLETONS", min_examples=50, epochs=25)

debug_info("Starting pair model training...")
pair_models = find_abstractions(pair_params, structure_type="PAIRS", min_examples=50, epochs=25)

debug_success(f"Training complete. {len(singleton_models)} singleton models and {len(pair_models)} pair models trained.")


# In[30]:


if all_dsl_shapes and singleton_models and pair_models:
    random_chair = random.choice(all_dsl_shapes)
    debug_info("--- ORIGINAL CHAIR ---")
    debug_info(f"Serialized children count: {len(random_chair.serialize()[1][1])}")
    debug_info(f"Preview: {random_chair}")

    abstracted_chair = integrate_abstractions(
        random_chair, singleton_models, pair_models, error_threshold=0.01
    )

    debug_info("\n--- ABSTRACTED CHAIR ---")
    if isinstance(abstracted_chair, Abstraction):
        debug_info(f"Abstraction pattern: {abstracted_chair.pattern_name}, compressed dim: {len(abstracted_chair.compressed_params)}")
    debug_info(f"Preview: {abstracted_chair}")

    # Visualization
    plot_chair(random_chair)
    plot_chair(abstracted_chair)
else:
    debug_error("Cannot run test: DSL shapes or trained models are missing.")


# In[31]:


print("Original Chair Object:")
print(random_chair)

print("\nAbstracted Chair Object:")
print(abstracted_chair)


# In[ ]:




