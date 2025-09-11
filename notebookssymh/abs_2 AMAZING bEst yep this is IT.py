#!/usr/bin/env python
# coding: utf-8

# # 1. Setup and Project Path Configuration
# 
# This initial block ensures that the Python script can find and import modules from the project's source directory (`src`). This is a crucial step for setting up the environment, especially when running the notebook from a different directory than the main project folder.
# 

# In[200]:


import sys
import os

# Adds the parent directory's 'src' folder to the Python system path.
# This allows us to import our custom modules like 'abstractionssymh'
# as if they were standard libraries.
sys.path.append(os.path.abspath("../src"))


# In[5]:


from termcolor import cprint

cprint("Hello, World!", "green")


# # 2. Directory and File Path Definition
# 
# Here, we define the paths to our data directories. This is a good practice as it makes the code more portable and easier to manage if the project structure changes. It uses the `pathlib` module, which provides a modern, object-oriented way to handle file system paths.
# 

# In[201]:


from pathlib import Path

# Get the current working directory where this notebook is located.
current_path = Path(os.getcwd())
# The base project directory is one level up from the notebook's location.
base_project_dir = current_path.parent
# Construct the full path to the specific dataset directory we'll be using.
dataset_directory = base_project_dir / "src" / "abstractionssymh" / "dataset"

print(f"Current Notebook location: {current_path}")
print(f"Base project directory: {base_project_dir}")
print(f"Target dataset directory: {dataset_directory}")


# # 3. A Quick Test: Loading and Parsing a Single Shape
# 
# This cell serves as a quick sanity check to ensure our data loader is working correctly. It reads a single JSON file and converts its content into a structured DSL (Domain-Specific Language) object.

# In[202]:


import random
from abstractionssymh.data_loader import parse_json_to_dsl

# Define the specific directory for 'Chair' models.
chair_directory = dataset_directory / "Chair"
# Get a list of all JSON files in the chair directory, sorted alphabetically.
json_files = sorted(list(chair_directory.glob("*.json")))

# Print the path of the first file to confirm we've found our data.
print(f"Loading and parsing the first file: {json_files[0]}")

# Open and read the JSON content of the first file.
# with open(json_files[0], 'r', encoding='utf-8') as f:
#     json_content = f.read()

# Open and read the JSON content of a random file.
random_chair_file = random.choice(json_files)
print(f"Randomly selected chair: {random_chair_file}")
with open(random_chair_file, 'r', encoding='utf-8') as f:
    json_content = f.read()

# Convert the raw JSON string into a structured DSL object.
dsl_object = parse_json_to_dsl(json_content)

# Print the parsed DSL object to visually confirm the conversion was successful.
print("\nSuccessfully parsed DSL object:")
print(dsl_object)


# In[203]:


from abstractionssymh.plot_utils import plot_dsl_with_k3d

plot_dsl_with_k3d(dsl_object)


# # 4. Core Analysis Logic: Functions for Tree Traversal and Parameter Collection
# 
# This is the heart of the script. The two functions below are responsible for navigating the complex tree-like structure of the DSL objects and extracting the numerical parameters associated with different patterns (individual nodes and parent-child pairs).

# In[204]:


from collections import defaultdict

# --- Data Collection Logic ---
def find_all_subtrees(node):
    _, (_, children) = node.serialize()
    for item in children:
        if hasattr(item, 'serialize'): yield from find_all_subtrees(item)
    yield node
def collect_singleton_and_pair_data(dsl_shapes):
    s_data, p_data = defaultdict(list), defaultdict(list)
    print("\n🔎 Analyzing all shapes and collecting parameters...")
    for i, shape in enumerate(dsl_shapes):
        if (i + 1) % 200 == 0 or i == len(dsl_shapes) - 1: print(f"  Processing shape {i+1}/{len(dsl_shapes)}...")
        for node in find_all_subtrees(shape):
            name, (p_params, children) = type(node).__name__, node.serialize()[1]
            if p_params: s_data[name].append(p_params)
            child_nodes = [item for item in children if hasattr(item, 'serialize')]
            for child in child_nodes:
                c_name, (c_params, _) = type(child).__name__, child.serialize()[1]
                pair_sig, combo_params = f"{name}({c_name})", p_params + c_params
                if combo_params: p_data[pair_sig].append(combo_params)
    print("✅ Analysis complete.")
    return dict(s_data), dict(p_data)


# In[205]:


# ==============================================================================
# SECTION 2: AUTOENCODER AND ABSTRACTION NODE
# ==============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from abstractionssymh.dsl_nodes import Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))
    def forward(self, x):
        return self.encoder(x), self.decoder(self.encoder(x))

def instantiate_pattern(pattern_name, params, children):
    # This function is a registry of how to build every known pattern.
    param_split = {'Translate(Rotate)': 3, 'Rotate(Scale)': 4, 'Scale(Box)': 3, 'Union(Translate)': 0}
    if pattern_name in ['Scale', 'Rotate', 'Translate']:
        NodeClass = globals()[pattern_name]
        return NodeClass(children[0], params)
    elif pattern_name in param_split:
        p_len = param_split[pattern_name]
        p_params, c_params = params[:p_len], params[p_len:]
        parent_name, child_name = pattern_name.split('(')[0], pattern_name.split('(')[1][:-1]
        ParentClass, ChildClass = globals()[parent_name], globals()[child_name]
        if child_name == 'Box': child_node = Box(-1)
        else: child_node = ChildClass(children[0], c_params)
        if ParentClass == Union: return Union(child_node, children[1])
        else: return ParentClass(child_node, p_params)
    return Box(-1)

class Abstraction:
    def __init__(self, pattern_name, compressed_params, model, children=None):
        self.pattern_name, self.compressed_params, self.model = pattern_name, compressed_params, model
        self.children = children if children is not None else []
    def __str__(self): return f"Abs({self.pattern_name}, dim={len(self.compressed_params)}) | pattern_name={self.pattern_name}, compressed_params={self.compressed_params}, model={self.model}, children={self.children}"
    def expand(self):
        self.model.eval()
        with torch.no_grad():
            params_tensor = torch.tensor(self.compressed_params, dtype=torch.float32).unsqueeze(0)
            reconstructed_params = self.model.decoder(params_tensor).squeeze().tolist()
        rebuilt_node = instantiate_pattern(self.pattern_name, reconstructed_params, self.children)
        return rebuilt_node.expand()

def prepare_autoencoder_train_data(parameters, mask, batch_size=64):
    tensor = torch.tensor(parameters, dtype=torch.float32)
    if mask.numel() != tensor.shape[0]: raise ValueError("Mask and data size mismatch")
    return DataLoader(TensorDataset(tensor[mask]), batch_size=batch_size, shuffle=True)

def is_well_explained(model, parameters_tensor, error_threshold):
    model.eval()
    with torch.no_grad():
        _, reconstructions = model(parameters_tensor)
        error, _ = torch.max(torch.abs(reconstructions - parameters_tensor), dim=-1)
        return error < error_threshold


# In[206]:


# ==============================================================================
# SECTION 3: THE `find_abstractions` TRAINING PIPELINE
# ==============================================================================

def find_abstractions(structures, structure_type="PATTERNS", min_examples=100, retrain_iterations=3, error_threshold=0.05, epochs=50):
    trained_models = {}
    print("\n" + "="*50 + f"\n===== STARTING ABSTRACTION PIPELINE FOR: {structure_type} =====\n" + "="*50 + "\n")
    sorted_structures = sorted(structures.items(), key=lambda item: len(item[1]), reverse=True)
    for name, parameters in sorted_structures:
        if len(parameters) < min_examples: continue
        num_params = len(parameters[0])
        if num_params <= 1: continue
        print(f"\n--- Training for '{name}' ({len(parameters)} examples, dim={num_params}) ---")
        params_tensor = torch.tensor(parameters, dtype=torch.float32)
        mask = torch.ones(len(parameters), dtype=torch.bool)
        for i in range(retrain_iterations):
            if torch.sum(mask).item() == 0: break
            train_dl = prepare_autoencoder_train_data(parameters, mask=mask)
            model = Autoencoder(num_params, max(1, num_params - 1))
            optimizer = AdamW(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            model.train()
            for epoch in range(epochs):
                for batch in train_dl:
                    x = batch[0]
                    optimizer.zero_grad()
                    _, x_rec = model(x)
                    loss = loss_fn(x_rec, x)
                    loss.backward()
                    optimizer.step()
            mask = is_well_explained(model, params_tensor, error_threshold=error_threshold)
        trained_models[name] = model
        print(f"  Final well-explained count: {torch.sum(mask).item()}/{len(parameters)}")
    print("\n" + "="*50 + f"\n===== PIPELINE COMPLETE FOR: {structure_type} =====\n" + "="*50 + "\n")
    return trained_models


# In[207]:


# ==============================================================================
# SECTION 4: THE `integrate_abstractions` FINAL LOGIC
# ==============================================================================

def integrate_abstractions(node, singleton_models, pair_models, error_threshold, depth=0):
    indent = "  " * depth
    if isinstance(node, Abstraction): return node

    _, (_, children) = node.serialize()
    rebuilt_children = [integrate_abstractions(c, singleton_models, pair_models, error_threshold, depth+1) if hasattr(c, 'serialize') else c for c in children]

    try:
        if isinstance(node, Union): current_node = Union(rebuilt_children[0], rebuilt_children[1])
        elif hasattr(node, 'child'): current_node = type(node)(rebuilt_children[0], **{k:v for k,v in node.__dict__.items() if k != 'child'})
        else: current_node = type(node)(*rebuilt_children)
    except: current_node = node

    child_nodes = [c for c in rebuilt_children if hasattr(c, 'serialize')]

    # Check for PAIR abstractions first (more specific)
    if len(child_nodes) == 1:
        child_node = child_nodes[0]
        pair_sig = f"{type(current_node).__name__}({type(child_node).__name__})"
        if pair_sig in pair_models and not isinstance(child_node, Abstraction):
            model, (p_params, _), (c_params, c_children) = pair_models[pair_sig], current_node.serialize()[1], child_node.serialize()[1]
            if p_params + c_params:
                print(f"{indent}Checking pair: '{pair_sig}'")
                combined_params = torch.tensor(p_params + c_params, dtype=torch.float32).unsqueeze(0)
                _, reconstruction = model(combined_params)
                error = torch.max(torch.abs(reconstruction - combined_params)).item()
                print(f"{indent}  - Error: {error:.4f} | Threshold: {error_threshold}")
                if error < error_threshold:
                    encoding, _ = model(combined_params)
                    grandchildren = [c for c in c_children if hasattr(c, 'serialize')]
                    print(f"{indent}  - PASSED: Abstracting '{pair_sig}'")
                    return Abstraction(pair_sig, encoding.squeeze().tolist(), model, children=grandchildren)
                else:
                    print(f"{indent}  - FAILED: Error too high.")

    # If no pair was made, check for SINGLETON abstraction
    name = type(current_node).__name__
    if name in singleton_models:
        model, (p_params, _) = singleton_models[name], current_node.serialize()[1]
        if p_params:
            print(f"{indent}Checking singleton: '{name}'")
            params_tensor = torch.tensor(p_params, dtype=torch.float32).unsqueeze(0)
            _, reconstruction = model(params_tensor)
            error = torch.max(torch.abs(reconstruction - params_tensor)).item()
            print(f"{indent}  - Error: {error:.4f} | Threshold: {error_threshold}")
            if error < error_threshold:
                encoding, _ = model(params_tensor)
                print(f"{indent}  - PASSED: Abstracting '{name}'")
                return Abstraction(name, encoding.squeeze().tolist(), model, children=child_nodes)
            else:
                print(f"{indent}  - FAILED: Error too high.")

    return current_node


# In[208]:


# --- 1. Load your dataset ---

# Get the current working directory where this notebook is located.
current_path = Path(os.getcwd())
# The base project directory is one level up from the notebook's location.
base_project_dir = current_path.parent
# Construct the full path to the specific dataset directory we'll be using.
dataset_directory = base_project_dir / "src" / "abstractionssymh" / "dataset"
chair_directory = dataset_directory / "Chair"
json_files = sorted(list(chair_directory.glob("*.json")))
all_dsl_shapes = [parse_json_to_dsl(Path(f).read_text()) for f in json_files[:1000]] # Using a small subset for speed
print(f"📂 Loaded {len(all_dsl_shapes)} shapes.")

# --- 2. Collect parameters ---
singleton_params, pair_params = collect_singleton_and_pair_data(all_dsl_shapes)

# --- 3. Run the training pipelines ---
singleton_models = find_abstractions(singleton_params, structure_type="SINGLETONS", min_examples=50, epochs=25)
pair_models = find_abstractions(pair_params, structure_type="PAIRS", min_examples=50, epochs=25)

# --- 4. Pick a random shape and integrate abstractions ---
# if all_dsl_shapes:
#     print("\n" + "="*50 + "\n===== STARTING ABSTRACTION INTEGRATION =====\n" + "="*50 + "\n")
#     random_chair = random.choice(all_dsl_shapes)

#     print("--- ORIGINAL CHAIR ---")
#     print(random_chair)

#     # This is the final step
#     abstracted_chair = integrate_abstractions(random_chair, singleton_models, pair_models, error_threshold=0.05)

#     print("\n--- ABSTRACTED CHAIR ---")
#     print(abstracted_chair)


# In[217]:


# --- 4. Pick a random shape and integrate abstractions ---
if all_dsl_shapes:
    print("\n" + "="*50 + "\n===== STARTING ABSTRACTION INTEGRATION =====\n" + "="*50 + "\n")
    random_chair = random.choice(all_dsl_shapes)

    print("--- ORIGINAL CHAIR ---")
    print(random_chair)

    # This is the final step
    abstracted_chair = integrate_abstractions(random_chair, singleton_models, pair_models, error_threshold=0.01)

    print("\n--- ABSTRACTED CHAIR ---")
    print(abstracted_chair)


# In[218]:


singleton_params.keys()


# In[219]:


pair_params.keys()


# In[226]:


print(random_chair)


# In[221]:


print(abstracted_chair)


# In[222]:


abstracted_chair.expand()


# In[223]:


plot_dsl_with_k3d(abstracted_chair)


# In[227]:


plot_dsl_with_k3d(random_chair)


# In[ ]:




