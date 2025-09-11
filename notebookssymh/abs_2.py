#!/usr/bin/env python
# coding: utf-8

# # 1. Setup and Project Path Configuration
# 
# This initial block ensures that the Python script can find and import modules from the project's source directory (`src`). This is a crucial step for setting up the environment, especially when running the notebook from a different directory than the main project folder.
# 

# In[1]:


import sys
import os

# Adds the parent directory's 'src' folder to the Python system path.
# This allows us to import our custom modules like 'abstractionssymh'
# as if they were standard libraries.
sys.path.append(os.path.abspath("../src"))


# # 2. Directory and File Path Definition
# 
# Here, we define the paths to our data directories. This is a good practice as it makes the code more portable and easier to manage if the project structure changes. It uses the `pathlib` module, which provides a modern, object-oriented way to handle file system paths.
# 

# In[2]:


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

# In[9]:


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


# In[10]:


from abstractionssymh.plot_utils import plot_dsl_with_k3d

plot_dsl_with_k3d(dsl_object)


# # 4. Core Analysis Logic: Functions for Tree Traversal and Parameter Collection
# 
# This is the heart of the script. The two functions below are responsible for navigating the complex tree-like structure of the DSL objects and extracting the numerical parameters associated with different patterns (individual nodes and parent-child pairs).

# In[11]:


from collections import defaultdict

def find_all_subtrees(node):
    """
    Recursively traverses a DSL tree and yields every sub-tree node (in a post-order traversal).

    Post-order traversal means a node's children are visited before the node itself. This
    is a robust way to ensure we analyze every component of the DSL shape.

    Args:
        node: The root node of a DSL object or a sub-tree.

    Yields:
        Every node in the tree, from the bottom-most leaves up to the root.
    """
    # Use the .serialize() method to get the children and other parameters of the node.
    # The method returns a tuple: (node_type_name, (parent_params, other_params_and_children)).
    _, (_, other_params_and_children) = node.serialize()

    # Iterate through all children to find sub-trees.
    for item in other_params_and_children:
        # Check if the item is a DSL node (by checking for the 'serialize' method).
        if hasattr(item, 'serialize'):
            # If it's a child node, recursively call the function to go deeper.
            yield from find_all_subtrees(item)

    # After all children have been yielded, yield the current node itself.
    yield node


def collect_pattern_parameters(dsl_shapes):
    """
    Analyzes a list of DSL objects to find singletons and parent-child pairs,
    collecting their associated numerical parameters based on the 'shallow pair' logic.

    A 'singleton' is a single node with parameters.
    A 'shallow pair' is a parent node and its immediate child, with combined parameters.

    Args:
        dsl_shapes: A list of DSL objects (the parsed shapes from the dataset).

    Returns:
        A tuple containing two dictionaries:
        1. A dictionary mapping singleton type names (e.g., 'Cuboid') to a list of their parameters.
        2. A dictionary mapping pair signatures (e.g., 'Symmetry(Cuboid)') to a list of their combined parameters.
    """
    # Use defaultdict to automatically create an empty list for a new key.
    singletons_data = defaultdict(list)
    pairs_data = defaultdict(list)

    print("\n🔎 Analyzing all shapes and collecting 'shallow pair' parameters...")
    # Iterate through each shape in the dataset.
    for i, shape in enumerate(dsl_shapes):
        # Provide a progress update for every 100 shapes.
        if (i + 1) % 100 == 0 or i == len(dsl_shapes) - 1:
            print(f"  Processing shape {i+1}/{len(dsl_shapes)}...")

        # Traverse the current shape's tree, from leaves up to the root.
        for node in find_all_subtrees(shape):
            node_type_name = type(node).__name__
            # Get the parameters and children of the current node.
            _, (parent_float_params, other_params_and_children) = node.serialize()

            # --- 1. Log the singleton's parameters (if the node has any).
            if parent_float_params:
                singletons_data[node_type_name].append(parent_float_params)

            # --- 2. Find and log the shallow pair's combined parameters.
            # Get a list of the *immediate* child nodes of the current node.
            child_nodes = [item for item in other_params_and_children if hasattr(item, 'serialize')]

            # Iterate through each child to form a pair.
            for child in child_nodes:
                child_type_name = type(child).__name__
                # Create a unique signature for the parent-child pair.
                pair_signature = f"{node_type_name}({child_type_name})"

                # Get *only* the float parameters from the immediate child,
                # ignoring any of its own children.
                _, (child_float_params, _) = child.serialize()

                # Combine the parent's parameters with the child's parameters.
                combined_params = parent_float_params + child_float_params

                # Only add the combined parameters if there are any.
                if combined_params:
                    pairs_data[pair_signature].append(combined_params)

    print("✅ Analysis complete.")
    return dict(singletons_data), dict(pairs_data)


# # 5. Main Execution: Loading, Analysis, and Reporting
# 
# This final section orchestrates the entire process. It loads the full dataset, calls our analysis function, and then presents the results in a clear, sorted format. This part of the code is self-contained and demonstrates the full workflow.
# 

# In[12]:


# --- 1. Load the ENTIRE dataset ---
chair_directory = dataset_directory / "Chair"
json_files = sorted(list(chair_directory.glob("*.json")))

all_dsl_shapes = []
print(f"📂 Loading {len(json_files)} shapes from {chair_directory}...")
for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        json_content = f.read()
    # Parse each JSON file and add the resulting DSL object to our list.
    dsl_object = parse_json_to_dsl(json_content)
    all_dsl_shapes.append(dsl_object)
print("✅ All shapes loaded into memory.")

# --- 2. Collect parameters for all patterns ---
singleton_params, pair_params = collect_pattern_parameters(all_dsl_shapes)

# --- 3. Sort the results by the number of times each pattern was found ---
# We use a lambda function to sort by the length of the list of parameters (the count).
sorted_singletons = sorted(singleton_params.items(), key=lambda item: len(item[1]), reverse=True)
sorted_pairs = sorted(pair_params.items(), key=lambda item: len(item[1]), reverse=True)

# --- 4. Print the final, summarized results ---

print("\n--- 📊 Total Singletons Parameter Summary ---")
for name, params_list in sorted_singletons:
    count = len(params_list)
    # Get the dimension of the parameters (e.g., a 3D vector has dimension 3).
    param_dim = len(params_list[0]) if count > 0 else 0
    print(f"  - Singleton: '{name}'")
    print(f"    - Found {count} times")
    print(f"    - Parameter Dimension: {param_dim}")

print("\n--- 📊 Total Pairs Parameter Summary (Shallow Combined) ---")
for name, params_list in sorted_pairs:
    count = len(params_list)
    param_dim = len(params_list[0]) if count > 0 else 0
    print(f"  - Pair: '{name}'")
    print(f"    - Found {count} times")
    print(f"    - Parameter Dimension: {param_dim}")
    # Show a sample of the parameters for a clearer understanding.
    print(f"    - Sample Params: {params_list[0]}")


# In[6]:


dict(sorted_singletons).keys()


# In[ ]:


dict(sorted_pairs).keys()


# In[13]:


dict(sorted_singletons)["Scale"]


# In[15]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

def prepare_autoencoder_train_data(parameters, mask, batch_size=64):
    tensor = torch.tensor(parameters, dtype=torch.float32)
    # Ensure mask is a boolean tensor and has the correct number of elements
    if mask.numel() != tensor.shape[0]:
        raise ValueError(f"Mask size ({mask.numel()}) does not match data size ({tensor.shape[0]})")
    dataset = TensorDataset(tensor[mask])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def is_well_explained(model, parameters_tensor, error_threshold):
    model.eval()
    with torch.no_grad():
        reconstructions = model(parameters_tensor)
        error, _ = torch.max(torch.abs(reconstructions - parameters_tensor), dim=-1)
        return error < error_threshold


# In[19]:


def find_abstractions(structures, structure_type="PATTERNS", min_examples=100, retrain_iterations=3, error_threshold=0.05, epochs=50):
    trained_models = {}
    print("\n" + "="*50)
    print(f"===== STARTING ABSTRACTION PIPELINE FOR: {structure_type} =====")
    print("="*50 + "\n")

    # Sort the structures by the number of examples, descending
    sorted_structures = sorted(structures.items(), key=lambda item: len(item[1]), reverse=True)

    for name, parameters in sorted_structures:
        if len(parameters) < min_examples:
            continue

        num_float_parameters = len(parameters[0])
        if num_float_parameters <= 1:
            continue

        print(f"\n--- Training for '{name}' ---")
        print(f"Found {len(parameters)} examples | Param Dim: {num_float_parameters}")

        params_tensor = torch.tensor(parameters, dtype=torch.float32)
        well_explained_mask = torch.ones(len(parameters), dtype=torch.bool)

        for i in range(retrain_iterations):
            num_training_examples = torch.sum(well_explained_mask).item()
            if num_training_examples == 0:
                print("  No well-explained examples left to train on. Stopping.")
                break

            train_dl = prepare_autoencoder_train_data(parameters, mask=well_explained_mask)

            hidden_dim = max(1, num_float_parameters - 1)
            model = Autoencoder(num_float_parameters, hidden_dim)
            optimizer = AdamW(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            model.train()
            for epoch in range(epochs):
                for batch in train_dl:
                    x = batch[0]
                    optimizer.zero_grad()
                    x_rec = model(x)
                    loss = loss_fn(x_rec, x)
                    loss.backward()
                    optimizer.step()

            well_explained_mask = is_well_explained(model, params_tensor, error_threshold=error_threshold)

        trained_models[name] = model
        print(f"  Final well-explained count: {torch.sum(well_explained_mask).item()}/{len(parameters)}")

    print("\n" + "="*50)
    print(f"===== PIPELINE COMPLETE FOR: {structure_type} =====")
    print(f"Successfully trained models for {len(trained_models)} patterns.")
    print("="*50 + "\n")
    return trained_models


# In[20]:


def find_all_subtrees(node):
    _, (_, children) = node.serialize()
    for item in children:
        if hasattr(item, 'serialize'): yield from find_all_subtrees(item)
    yield node

def collect_singleton_and_pair_data(dsl_shapes):
    singleton_data = defaultdict(list)
    pair_data = defaultdict(list)
    print("\n🔎 Analyzing all shapes and collecting parameters...")
    for i, shape in enumerate(dsl_shapes):
        if (i + 1) % 200 == 0 or i == len(dsl_shapes) - 1: print(f"  Processing shape {i+1}/{len(dsl_shapes)}...")
        for node in find_all_subtrees(shape):
            name, (p_params, children) = type(node).__name__, node.serialize()[1]
            if p_params:
                singleton_data[name].append(p_params)

            child_nodes = [item for item in children if hasattr(item, 'serialize')]
            for child in child_nodes:
                c_name, (c_params, _) = type(child).__name__, child.serialize()[1]
                pair_sig = f"{name}({c_name})"
                combo_params = p_params + c_params
                if combo_params:
                    pair_data[pair_sig].append(combo_params)
    print("✅ Analysis complete.")
    return dict(singleton_data), dict(pair_data)


# In[22]:


# --- 2. Collect parameters for singletons and pairs separately ---
singleton_params, pair_params = collect_singleton_and_pair_data(all_dsl_shapes[:100])

# --- 3. Run the training pipeline for SINGLETONS ---
singleton_models = find_abstractions(singleton_params, structure_type="SINGLETONS")

# --- 4. Run the training pipeline for PAIRS ---
pair_models = find_abstractions(pair_params, structure_type="PAIRS")

# --- 5. Print a final summary of all trained models ---
print("\n--- 📖 Final Summary of All Trained Models ---")
print("\n--- Singleton Models ---")
if singleton_models:
    for name, model in singleton_models.items():
        input_dim = model.encoder[0].in_features
        hidden_dim = model.encoder[-1].out_features
        print(f"  - '{name}': Compresses from {input_dim} -> {hidden_dim} dimensions.")
else:
    print("  No singleton models were trained.")

print("\n--- Pair Models ---")
if pair_models:
    for name, model in pair_models.items():
        input_dim = model.encoder[0].in_features
        hidden_dim = model.encoder[-1].out_features
        print(f"  - '{name}': Compresses from {input_dim} -> {hidden_dim} dimensions.")
else:
    print("  No pair models were trained.")


# In[23]:


# --- 6. Final Evaluation of All Trained Models ---
print("\n--- 📈 Final Model Performance Summary ---")
error_threshold = 0.05 # Use the same threshold as in training for a fair comparison

print("\n--- Singleton Models Performance ---")
if singleton_models:
    for name, model in singleton_models.items():
        # Get the original, full list of parameters for this pattern
        params_list = singleton_params.get(name, [])
        if not params_list: continue

        params_tensor = torch.tensor(params_list, dtype=torch.float32)
        well_explained_mask = is_well_explained(model, params_tensor, error_threshold=error_threshold)

        num_well_explained = torch.sum(well_explained_mask).item()
        total_examples = len(params_list)
        percentage = (num_well_explained / total_examples) * 100

        print(f"  - Model: '{name}'")
        print(f"    - Well-Explained: {num_well_explained} / {total_examples} ({percentage:.2f}%)")
else:
    print("  No singleton models to evaluate.")

print("\n--- Pair Models Performance ---")
if pair_models:
    for name, model in pair_models.items():
        # Get the original, full list of parameters for this pattern
        params_list = pair_params.get(name, [])
        if not params_list: continue

        params_tensor = torch.tensor(params_list, dtype=torch.float32)
        well_explained_mask = is_well_explained(model, params_tensor, error_threshold=error_threshold)

        num_well_explained = torch.sum(well_explained_mask).item()
        total_examples = len(params_list)
        percentage = (num_well_explained / total_examples) * 100

        print(f"  - Model: '{name}'")
        print(f"    - Well-Explained: {num_well_explained} / {total_examples} ({percentage:.2f}%)")
else:
    print("  No pair models to evaluate.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import numpy as np
import k3d

# 1. Extract the 'Scale' parameters from your variable
print("Extracting 'Scale' parameters...")
params_dict = dict(sorted_singletons)
scale_params_list = params_dict.get('Scale')

if not scale_params_list:
    print("Could not find 'Scale' data in the variable.")
else:
    print(f"Found {len(scale_params_list)} data points for 'Scale'.")
    # 2. Convert to a NumPy array, ensuring the data type is float32 for K3D
    scale_params_array = np.array(scale_params_list, dtype=np.float32)

    # 3. Create the interactive K3D plot
    print("Generating interactive K3D plot...")
    plot = k3d.plot(name='Scale Parameter Clusters')

    # Add the points to the plot
    # You might need to adjust point_size based on the scale of your data
    points_object = k3d.points(
        positions=scale_params_array, 
        point_size=0.05,
        shader='3d'
    )
    plot += points_object

    print("✅ Plot generated. Use your mouse to rotate, pan, and zoom.")
    # In a Jupyter environment, this will display the interactive widget
    plot.display()


# In[ ]:




