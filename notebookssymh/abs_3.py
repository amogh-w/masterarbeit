#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sys
import os
sys.path.append(os.path.abspath("../src"))


# In[19]:


DEBUG = True

from termcolor import cprint

def debug_info(*args):
    """Prints a yellow 'INFO' message if DEBUG is True."""
    if DEBUG:
        cprint(f"[INFO] {' '.join(map(str, args))}", "yellow")

def debug_error(*args):
    """Prints a red 'ERROR' message if DEBUG is True."""
    if DEBUG:
        cprint(f"[ERROR] {' '.join(map(str, args))}", "red")

def debug_success(*args):
    """Prints a green 'SUCCESS' message if DEBUG is True."""
    if DEBUG:
        cprint(f"[SUCCESS] {' '.join(map(str, args))}", "green")


# In[20]:


from pathlib import Path

def get_dataset_directory():
    current_path = Path.cwd()
    base_project_dir = current_path.parent
    dataset_directory = base_project_dir / "src" / "abstractionssymh" / "dataset"

    debug_info("Current notebook location:", current_path)
    debug_info("Base project directory:", base_project_dir)
    debug_info("Target dataset directory:", dataset_directory)

    return dataset_directory

dataset_directory = get_dataset_directory()


# In[21]:


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

chair_directory = dataset_directory / "Chair"
dsl_object = load_chair_dsl(chair_directory, use_random=False)


# In[22]:


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

plot_chair(dsl_object)


# In[23]:


from collections import defaultdict

def find_all_subtrees(node):
    """Recursively yields all subtrees (nodes) in a DSL object."""
    name, (params, children) = type(node).__name__, node.serialize()[1]
    for child in children:
        if hasattr(child, "serialize"):
            yield from find_all_subtrees(child)
    yield node


# In[24]:


def collect_singleton_and_pair_data(dsl_shapes):
    """Collects singleton parameters and parent-child pair parameters."""
    if not dsl_shapes:
        debug_error("No DSL shapes provided for analysis.")
        return {}, {}

    s_data, p_data = defaultdict(list), defaultdict(list)

    for shape in dsl_shapes:
        for node in find_all_subtrees(shape):
            name, (p_params, children) = type(node).__name__, node.serialize()[1]

            if p_params:
                s_data[name].append(p_params)

            for child in (c for c in children if hasattr(c, "serialize")):
                c_name, (c_params, _) = type(child).__name__, child.serialize()[1]
                combo_params = p_params + c_params
                if combo_params:
                    pair_sig = f"{name}({c_name})"
                    p_data[pair_sig].append(combo_params)

    debug_success("Collected keys:",
                  f"{s_data.keys()} singletons, {p_data.keys()} pairs")
    debug_success("Collected parameters:",
                  f"{len(s_data)} singletons, {len(p_data)} pairs")
    return dict(sorted(s_data.items())), dict(sorted(p_data.items()))


# In[25]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from abstractionssymh.dsl_nodes import Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans


class Autoencoder(nn.Module):
    """Simple fully-connected autoencoder with configurable hidden dim."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        debug_info(f"Initializing Autoencoder: input_dim={input_dim}, hidden_dim={hidden_dim}")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

        debug_success("Autoencoder initialized successfully.")

    def forward(self, x):
        # debug_info(f"Forward pass: input shape={tuple(x.shape)}")
        latent = self.encoder(x)
        # debug_info(f"Encoded latent shape={tuple(latent.shape)}")
        decoded = self.decoder(latent)
        # debug_info(f"Decoded output shape={tuple(decoded.shape)}")
        return latent, decoded


# In[26]:


def instantiate_pattern(pattern_name, params, children):
    """Rebuilds a DSL node from its pattern name and parameters."""
    debug_info(f"Instantiating pattern: {pattern_name} with params={params} and {len(children)} children")

    param_split = {
        "Translate(Rotate)": 3,
        "Rotate(Scale)": 4,
        "Scale(Box)": 3,
        "Union(Translate)": 0,
    }

    if pattern_name in ["Scale", "Rotate", "Translate"]:
        NodeClass = globals()[pattern_name]
        debug_success(f"Direct instantiation of {pattern_name}")
        return NodeClass(children[0], params)

    elif pattern_name in param_split:
        p_len = param_split[pattern_name]
        p_params, c_params = params[:p_len], params[p_len:]
        parent_name, child_name = pattern_name.split("(")[0], pattern_name.split("(")[1][:-1]
        ParentClass, ChildClass = globals()[parent_name], globals()[child_name]

        debug_info(f"Split into parent={parent_name} with {p_params}, child={child_name} with {c_params}")

        if child_name == "Box":
            child_node = Box(-1)
        else:
            child_node = ChildClass(children[0], c_params)

        if ParentClass == Union:
            debug_success(f"Constructed Union node: {parent_name}({child_name})")
            return Union(child_node, children[1])
        else:
            debug_success(f"Constructed composite node: {parent_name}({child_name})")
            return ParentClass(child_node, p_params)

    debug_error(f"Unknown pattern {pattern_name}, defaulting to Box(-1)")
    return Box(-1)


# In[27]:


class Abstraction:
    """Encapsulates an abstracted pattern with compressed parameters."""
    def __init__(self, pattern_name, compressed_params, model, children=None):
        self.pattern_name = pattern_name
        self.compressed_params = compressed_params
        self.model = model
        self.children = children if children is not None else []

        debug_success(f"Created Abstraction: {self}")

    def __str__(self):
        return f"Abs({self.pattern_name}, dim={len(self.compressed_params)})"

    __repr__ = __str__  # makes debugging in lists/dicts nicer

    def expand(self):
        """Reconstructs the full DSL node from compressed parameters."""
        debug_info(f"Expanding abstraction: {self}")

        if not self.compressed_params:
            debug_error("No compressed params found. Returning Box(-1).")
            return Box(-1)  # fallback if params are missing

        self.model.eval()
        with torch.no_grad():
            params_tensor = torch.tensor(
                self.compressed_params, dtype=torch.float32
            ).unsqueeze(0)
            debug_info(f"Compressed params tensor shape: {tuple(params_tensor.shape)}")
            reconstructed_params = self.model.decoder(params_tensor).squeeze().tolist()
            debug_success(f"Reconstructed params: {reconstructed_params}")

        rebuilt_node = instantiate_pattern(self.pattern_name, reconstructed_params, self.children)
        debug_success(f"Successfully rebuilt node: {rebuilt_node}")
        return rebuilt_node.expand()


# In[28]:


def prepare_autoencoder_train_data(parameters, mask, batch_size=64):
    """Creates a DataLoader from parameters and a boolean mask."""
    debug_info(f"Preparing DataLoader: batch_size={batch_size}, mask sum={mask.sum().item()}")
    tensor = torch.tensor(parameters, dtype=torch.float32)
    if mask.shape[0] != tensor.shape[0]:
        debug_error(f"Mask size {mask.shape[0]} does not match data size {tensor.shape[0]}")
        raise ValueError(f"Mask size {mask.shape[0]} != data size {tensor.shape[0]}")
    dataloader = DataLoader(TensorDataset(tensor[mask]), batch_size=batch_size, shuffle=True)
    debug_success("DataLoader prepared successfully.")
    return dataloader


# In[29]:


def is_well_explained(model, parameters_tensor, error_threshold):
    """Checks if parameters are well reconstructed by the autoencoder."""
    model.eval()
    with torch.no_grad():
        _, reconstructions = model(parameters_tensor)
        error, _ = torch.max(torch.abs(reconstructions - parameters_tensor), dim=-1)
    debug_info(f"Max reconstruction error per sample: {error}")
    well_explained = error < error_threshold
    debug_info(f"Number of well-explained examples: {well_explained.sum().item()}/{len(well_explained)}")
    return well_explained


# In[30]:


def train_autoencoder(model, dataloader, epochs=50, lr=1e-3):
    """Trains an autoencoder on the given DataLoader."""
    debug_info(f"Training autoencoder for {epochs} epochs, learning rate={lr}")
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            _, x_rec = model(x)
            loss = loss_fn(x_rec, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        debug_info(f"Epoch {epoch+1}/{epochs} - Average loss: {epoch_loss / len(dataloader.dataset):.6f}")

    debug_success("Autoencoder training complete.")
    return model


# In[31]:


def find_abstractions(
    structures,
    structure_type="PATTERNS",
    min_examples=100,
    retrain_iterations=3,
    error_threshold=0.05,
    epochs=50,
):
    """Trains autoencoders for each structure type and returns models."""
    trained_models = {}
    sorted_structures = sorted(structures.items(), key=lambda item: len(item[1]), reverse=True)

    debug_info(f"Finding abstractions for {len(sorted_structures)} structures of type {structure_type}")

    for name, parameters in sorted_structures:
        if len(parameters) < min_examples:
            debug_info(f"Skipping {name} (only {len(parameters)} examples)")
            continue

        num_params = len(parameters[0])
        if num_params <= 1:
            debug_info(f"Skipping {name} (too few params: {num_params})")
            continue

        debug_success(f"Training model for {name} with {len(parameters)} samples, {num_params} params")
        params_tensor = torch.tensor(parameters, dtype=torch.float32)
        mask = torch.ones(len(parameters), dtype=torch.bool)

        for iteration in range(retrain_iterations):
            if not mask.any():
                debug_info(f"All {name} examples filtered out at iteration {iteration}")
                break

            debug_info(f"Iteration {iteration+1}/{retrain_iterations} for {name}")
            dataloader = prepare_autoencoder_train_data(parameters, mask=mask)
            model = Autoencoder(num_params, max(1, num_params - 1))
            model = train_autoencoder(model, dataloader, epochs=epochs)

            mask = is_well_explained(model, params_tensor, error_threshold)

        trained_models[name] = model
        debug_success(f"Model training finished for {name}. Total well-explained examples: {mask.sum().item()}/{len(parameters)}")

    debug_success(f"Finished training {len(trained_models)} {structure_type}")
    return trained_models


# In[32]:


def integrate_abstractions(node, singleton_models, pair_models, error_threshold, depth=0):
    """Recursively integrates trained abstractions into a DSL tree."""
    indent = "  " * depth
    debug_info(f"{indent}Integrating node: {type(node).__name__}")

    if isinstance(node, Abstraction):
        debug_info(f"{indent}Node is already an Abstraction, returning as-is")
        return node

    _, (_, children) = node.serialize()
    rebuilt_children = [
        integrate_abstractions(c, singleton_models, pair_models, error_threshold, depth + 1)
        if hasattr(c, "serialize") else c
        for c in children
    ]

    try:
        if isinstance(node, Union):
            current_node = Union(rebuilt_children[0], rebuilt_children[1])
        elif hasattr(node, "child"):
            kwargs = {k: v for k, v in node.__dict__.items() if k != "child"}
            current_node = type(node)(rebuilt_children[0], **kwargs)
        else:
            current_node = type(node)(*rebuilt_children)
    except Exception as e:
        debug_error(f"{indent}Node reconstruction failed: {e}")
        return node

    child_nodes = [c for c in rebuilt_children if hasattr(c, "serialize")]

    # Try pair abstraction
    if len(child_nodes) == 1:
        child_node = child_nodes[0]
        pair_sig = f"{type(current_node).__name__}({type(child_node).__name__})"

        if pair_sig in pair_models and not isinstance(child_node, Abstraction):
            model = pair_models[pair_sig]
            (p_params, _), (c_params, c_children) = current_node.serialize()[1], child_node.serialize()[1]

            if p_params + c_params:
                combined = torch.tensor(p_params + c_params, dtype=torch.float32).unsqueeze(0)
                _, reconstruction = model(combined)
                error = torch.max(torch.abs(reconstruction - combined)).item()
                debug_info(f"{indent}Pair abstraction {pair_sig}: error={error:.6f}")

                if error < error_threshold:
                    encoding, _ = model(combined)
                    grandchildren = [c for c in c_children if hasattr(c, "serialize")]
                    debug_success(f"{indent}Pair abstraction applied: {pair_sig}")
                    return Abstraction(pair_sig, encoding.squeeze().tolist(), model, children=grandchildren)

    # Try singleton abstraction
    name = type(current_node).__name__
    if name in singleton_models:
        model = singleton_models[name]
        p_params, _ = current_node.serialize()[1]

        if p_params:
            params_tensor = torch.tensor(p_params, dtype=torch.float32).unsqueeze(0)
            _, reconstruction = model(params_tensor)
            error = torch.max(torch.abs(reconstruction - params_tensor)).item()
            debug_info(f"{indent}Singleton abstraction {name}: error={error:.6f}")

            if error < error_threshold:
                encoding, _ = model(params_tensor)
                debug_success(f"{indent}Singleton abstraction applied: {name}")
                return Abstraction(name, encoding.squeeze().tolist(), model, children=child_nodes)

    debug_info(f"{indent}No abstraction applied, returning original node")
    return current_node


# In[33]:


# Load DSL shapes

chair_directory = dataset_directory / "Chair"
json_files = sorted(list(chair_directory.glob("*.json")))

try:
    debug_info("Starting to load DSL shapes from JSON files...")
    all_dsl_shapes = [parse_json_to_dsl(Path(f).read_text()) for f in json_files[:10]]
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


# In[34]:


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


# In[ ]:





# In[ ]:




