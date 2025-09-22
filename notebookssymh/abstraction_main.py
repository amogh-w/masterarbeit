#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.abspath("../src"))


# In[2]:


from pathlib import Path
from abstractionssymh.debug_utils import debug_info, debug_error, debug_success

def get_dataset_directory():
    current_path = Path.cwd()
    base_project_dir = current_path.parent
    dataset_directory = base_project_dir / "src" / "abstractionssymh" / "dataset"

    debug_info("Current notebook location:", current_path)
    debug_info("Base project directory:", base_project_dir)
    debug_info("Target dataset directory:", dataset_directory)

    return dataset_directory


# In[3]:


dataset_directory = get_dataset_directory()


# In[4]:


import random
from abstractionssymh.data_loader import parse_json_to_dsl

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


# In[5]:


chair_directory = dataset_directory / "Chair"
dsl_object = load_chair_dsl(chair_directory, use_random=False)


# In[6]:


from abstractionssymh.plot_utils import plot_dsl_with_k3d

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


# In[7]:


plot_chair(dsl_object)


# In[8]:


from abstractionssymh.dsl_utils import collect_singleton_and_pair_data


# In[9]:


# Load DSL shapes

from abstractionssymh.abstraction_utils import find_abstractions


chair_directory = dataset_directory / "Chair"
json_files = sorted(list(chair_directory.glob("*.json")))

try:
    debug_info("Starting to load DSL shapes from JSON files...")
    all_dsl_shapes = [parse_json_to_dsl(Path(f).read_text()) for f in json_files[:100]]
    debug_success(f"Loaded {len(all_dsl_shapes)} shapes from dataset.")
except Exception as e:
    debug_error("Failed to load DSL shapes:", e)
    all_dsl_shapes = []

# Collect parameters
if all_dsl_shapes:
    debug_info("Collecting singleton and pair parameters from loaded shapes...")
    singleton_params, pair_params = collect_singleton_and_pair_data(all_dsl_shapes)

    num_singleton = sum(len(v) for v in singleton_params.values())
    num_pair = sum(len(v) for v in pair_params.values())
    debug_success(f"Collected {num_singleton} singleton and {num_pair} pair parameter sets.")

    # Report top structures by count
    # top_singletons = sorted(singleton_params.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    # top_pairs = sorted(pair_params.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    # debug_info("Top singleton structures by example count:", top_singletons)
    # debug_info("Top pair structures by example count:", top_pairs)

    # Train singleton models
    debug_info("Starting singleton model training...")
    singleton_models = find_abstractions(singleton_params, structure_type="SINGLETONS", min_examples=50, epochs=25)

    # Train pair models
    debug_info("Starting pair model training...")
    pair_models = find_abstractions(pair_params, structure_type="PAIRS", min_examples=50, epochs=25)

    debug_success(f"Training complete. {len(singleton_models)} singleton models and {len(pair_models)} pair models trained.")

else:
    debug_error("No DSL shapes available. Abstraction pipeline skipped.")


# In[10]:


from abstractionssymh.abstraction_utils import Abstraction, integrate_abstractions

# Test on a random DSL shape
if all_dsl_shapes and singleton_models and pair_models:
    random_chair = random.choice(all_dsl_shapes)
    debug_info("--- ORIGINAL CHAIR ---")
    debug_info(f"Type: {type(random_chair).__name__}")
    debug_info(f"Serialized children count: {len(random_chair.serialize()[1][1])}")
    debug_info(f"Preview: {random_chair}")

    # Integrate abstractions
    debug_info("Integrating abstractions...")
    abstracted_chair = integrate_abstractions(
        random_chair, singleton_models, pair_models, error_threshold=0.01
    )

    debug_info("\n--- ABSTRACTED CHAIR ---")
    debug_info(f"Type: {type(abstracted_chair).__name__}")
    if isinstance(abstracted_chair, Abstraction):
        debug_info(f"Abstraction pattern: {abstracted_chair.pattern_name}, compressed dim: {len(abstracted_chair.compressed_params)}")
    debug_info(f"Preview: {abstracted_chair}")

    # Visualization
    try:
        debug_info("Plotting original and abstracted DSL shapes...")
        plot_dsl_with_k3d(random_chair)
        plot_dsl_with_k3d(abstracted_chair)
        debug_success("Visualization complete.")
    except Exception as e:
        debug_error("Plotting failed:", e)
else:
    debug_error("Cannot run test: DSL shapes or trained models are missing.")


# In[11]:


print(abstracted_chair)


# In[12]:


print(random_chair)

