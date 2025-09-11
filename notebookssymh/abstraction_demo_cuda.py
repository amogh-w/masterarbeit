#!/usr/bin/env python
# coding: utf-8

# In[12]:


import sys
import os
sys.path.append(os.path.abspath("../src"))


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


from collections import defaultdict

def find_all_subtrees(node):
    """Recursively yields all subtrees (nodes) in a DSL object."""
    name, (params, children) = type(node).__name__, node.serialize()[1]
    for child in children:
        if hasattr(child, "serialize"):
            yield from find_all_subtrees(child)
    yield node


# In[18]:


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


# In[19]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from abstractionssymh.dsl_nodes import Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[20]:


device


# In[21]:


def t(tensor):
    return tensor.to(device)


# In[22]:


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


# In[23]:


def instantiate_pattern(pattern_name, params, children):
    """
    Rebuilds a DSL sub-tree from its pattern name, parameters, and preserved children.
    Handles both SINGLETON and PAIR patterns.
    """
    debug_info(f"[INSTANTIATE] Pattern: '{pattern_name}' | params={params[:10]}{'...' if len(params)>10 else ''} | #children={len(children)}")

    # Registry defining parent parameter lengths for known PAIRS
    param_split = {
        'Scale(Box)': 3, 'Rotate(Scale)': 4, 'Translate(Rotate)': 3,
        'Union(Translate)': 0, 'Union(SymRef)': 0, 'Union(SymRot)': 0, 'Union(SymTrans)': 0,
        'SymRef(Union)': 6, 'SymRef(Translate)': 6, 'SymRot(Union)': 6,
        'SymRot(Translate)': 6, 'SymTrans(Translate)': 3
    }

    # --- SINGLETON RECONSTRUCTION ---
    if pattern_name in ['Scale', 'Rotate', 'Translate', 'SymRef', 'SymRot', 'SymTrans']:
        NodeClass = globals()[pattern_name]
        child_node = children[0] if children else Box(-1)

        if NodeClass == SymRef:
            debug_info(f"  [SINGLETON] SymRef with params length {len(params)}")
            return SymRef(child_node, plane_normal=params[:3], point_on_plane=params[3:])
        if NodeClass == SymRot:
            debug_info(f"  [SINGLETON] SymRot with params length {len(params)}")
            return SymRot(child_node, axis=params[:3], center=params[3:], n_fold=-1)
        if NodeClass == SymTrans:
            debug_info(f"  [SINGLETON] SymTrans with params length {len(params)}")
            return SymTrans(child_node, end_point=params, n_fold=-1)

        return NodeClass(child_node, params)

    # --- PAIR RECONSTRUCTION ---
    elif pattern_name in param_split:
        p_len = param_split[pattern_name]
        p_params, c_params = params[:p_len], params[p_len:]
        parent_name, child_name = pattern_name.split('(')[0], pattern_name.split('(')[1][:-1]
        ParentClass, ChildClass = globals()[parent_name], globals()[child_name]

        # --- Build child node ---
        grandchild_node = children[0] if children else Box(-1)
        if ChildClass == Box:
            child_node = Box(-1)
        elif ChildClass == Union:
            if len(children) >= 2:
                child_node = Union(children[0], children[1])
            else:
                debug_error(f"  [PAIR] Unexpected children for Union in {pattern_name}")
                child_node = Box(-1)
        elif ChildClass == SymRef:
            child_node = SymRef(grandchild_node, plane_normal=c_params[:3], point_on_plane=c_params[3:])
        elif ChildClass == SymRot:
            child_node = SymRot(grandchild_node, axis=c_params[:3], center=c_params[3:], n_fold=-1)
        elif ChildClass == SymTrans:
            child_node = SymTrans(grandchild_node, end_point=c_params, n_fold=-1)
        else:
            child_node = ChildClass(grandchild_node, c_params)

        # --- Build parent node ---
        if ParentClass == Union:
            if len(children) >= 2:
                return Union(child_node, children[1])
            else:
                debug_error(f"  [PAIR] Union parent with missing second child in {pattern_name}")
                return Box(-1)
        elif ParentClass == SymRef:
            return SymRef(child_node, plane_normal=p_params[:3], point_on_plane=p_params[3:])
        elif ParentClass == SymRot:
            return SymRot(child_node, axis=p_params[:3], center=p_params[3:], n_fold=-1)
        elif ParentClass == SymTrans:
            return SymTrans(child_node, end_point=p_params, n_fold=-1)
        else:
            return ParentClass(child_node, p_params)

    debug_error(f"[UNKNOWN PATTERN] '{pattern_name}', defaulting to Box(-1)")
    return Box(-1)


# In[24]:


import textwrap


class Abstraction:
    def __init__(self, pattern_name, compressed_params, model, children=None):
        self.pattern_name, self.compressed_params, self.model = pattern_name, compressed_params, model
        self.children = children if children is not None else []

    def __str__(self):
        # Create the header for the Abstraction node
        header = f"Abs({self.pattern_name}, dim={len(self.compressed_params)})"

        # If the abstraction has preserved children, indent and print them
        if not self.children:
            return header
        else:
            # Indent each child's own string representation
            child_strs = [textwrap.indent(str(c), '    ') for c in self.children]
            # Join the indented child strings
            return f"{header}(\n" + ",\n".join(child_strs) + "\n)"

    __repr__ = __str__  # makes debugging in lists/dicts nicer

    def expand(self):
        """Reconstructs the full DSL node from compressed parameters."""
        debug_info(f"Expanding abstraction: {self}")

        if not self.compressed_params:
            debug_error("No compressed params found. Returning Box(-1).")
            return Box(-1)  # fallback if params are missing

        self.model.eval()
        with torch.no_grad():
            params_tensor = t(torch.tensor(
                self.compressed_params, dtype=torch.float32
            )).unsqueeze(0)
            debug_info(f"Compressed params tensor shape: {tuple(params_tensor.shape)}")
            reconstructed_params = self.model.decoder(params_tensor).squeeze().tolist()
            debug_success(f"Reconstructed params: {reconstructed_params}")

        rebuilt_node = instantiate_pattern(self.pattern_name, reconstructed_params, self.children)
        debug_success(f"Successfully rebuilt node: {rebuilt_node}")
        return rebuilt_node.expand()


# In[25]:


def prepare_autoencoder_train_data(parameters, mask, batch_size=64):
    """Creates a DataLoader from parameters and a boolean mask."""
    debug_info(f"Preparing DataLoader: batch_size={batch_size}, mask sum={mask.sum().item()}")
    tensor = t(torch.tensor(parameters, dtype=torch.float32))
    if mask.shape[0] != tensor.shape[0]:
        debug_error(f"Mask size {mask.shape[0]} does not match data size {tensor.shape[0]}")
        raise ValueError(f"Mask size {mask.shape[0]} != data size {tensor.shape[0]}")
    dataloader = DataLoader(TensorDataset(tensor[mask]), batch_size=batch_size, shuffle=True)
    debug_success("DataLoader prepared successfully.")
    return dataloader


# In[26]:


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


# In[27]:


def train_autoencoder(model, dataloader, epochs=50, lr=1e-3):
    """Trains an autoencoder on the given DataLoader."""
    debug_info(f"Training autoencoder for {epochs} epochs, learning rate={lr}")
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            _, x_rec = model(x)
            loss = loss_fn(x_rec, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        debug_info(f"Epoch {epoch+1}/{epochs} - Average loss: {epoch_loss / len(dataloader.dataset):.6f}")

    debug_success("Autoencoder training complete.")
    return model


# In[28]:


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
        params_tensor = t(torch.tensor(parameters, dtype=torch.float32))
        mask = torch.ones(len(parameters), dtype=torch.bool)

        for iteration in range(retrain_iterations):
            if not mask.any():
                debug_info(f"All {name} examples filtered out at iteration {iteration}")
                break

            debug_info(f"Iteration {iteration+1}/{retrain_iterations} for {name}")
            dataloader = prepare_autoencoder_train_data(parameters, mask=mask)
            model = Autoencoder(num_params, max(1, num_params - 1)).to(device)
            model = train_autoencoder(model, dataloader, epochs=epochs)

            mask = is_well_explained(model, params_tensor, error_threshold)

        trained_models[name] = model
        debug_success(f"Model training finished for {name}. Total well-explained examples: {mask.sum().item()}/{len(parameters)}")

    debug_success(f"Finished training {len(trained_models)} {structure_type}")
    return trained_models


# In[29]:


def integrate_abstractions(node, singleton_models, pair_models, error_threshold, depth=0):
    indent = "  " * depth

    if isinstance(node, Abstraction):
        debug_info(f"{indent}[Abstraction] Node already abstracted: {node.pattern_name}")
        return node

    # 1. Recurse to children first
    _, (_, children) = node.serialize()
    rebuilt_children = [
        integrate_abstractions(c, singleton_models, pair_models, error_threshold, depth+1)
        if hasattr(c, 'serialize') else c
        for c in children
    ]

    # 2. Rebuild current node
    try:
        if isinstance(node, Union):
            current_node = Union(rebuilt_children[0], rebuilt_children[1])
        elif isinstance(node, SymRef):
            current_node = SymRef(rebuilt_children[0], plane_normal=node.plane, point_on_plane=node.point_on_plane)
        elif isinstance(node, SymRot):
            current_node = SymRot(rebuilt_children[0], axis=node.axis, center=node.center, n_fold=node.n)
        elif isinstance(node, SymTrans):
            current_node = SymTrans(rebuilt_children[0], end_point=node.end_point, n_fold=node.n)
        elif hasattr(node, 'child'):
            kwargs = {k:v for k,v in node.__dict__.items() if k != 'child'}
            current_node = type(node)(rebuilt_children[0], **kwargs)
        else:
            current_node = type(node)(*rebuilt_children)
        debug_info(f"{indent}[Rebuilt] {type(current_node).__name__} with {len(rebuilt_children)} children")
    except Exception as e:
        debug_error(f"{indent}[Rebuild FAILED] {type(node).__name__} at depth {depth}: {e}")
        return node

    # 3. Try pair abstraction
    child_nodes = [c for c in rebuilt_children if hasattr(c, 'serialize')]
    if len(child_nodes) == 1:
        child_node = child_nodes[0]
        if not isinstance(child_node, Abstraction):
            pair_sig = f"{type(current_node).__name__}({type(child_node).__name__})"
            if pair_sig in pair_models:
                model, (p_params, _), (c_params, c_children) = pair_models[pair_sig], current_node.serialize()[1], child_node.serialize()[1]
                if p_params + c_params:
                    combined = t(torch.tensor(p_params + c_params, dtype=torch.float32)).unsqueeze(0)
                    _, reconstruction = model(combined)
                    error = torch.max(torch.abs(reconstruction - combined)).item()
                    debug_info(f"{indent}[PAIR CHECK] {pair_sig}, error={error:.4f}")
                    if error < error_threshold:
                        encoding, _ = model(combined)
                        grandchildren = [c for c in c_children if hasattr(c, 'serialize')]
                        debug_success(f"{indent}[PAIR ABSTRACTION CREATED] {pair_sig}")
                        return Abstraction(pair_sig, encoding.squeeze().tolist(), model, children=grandchildren)

    # 4. Try singleton abstraction
    name = type(current_node).__name__
    if name in singleton_models:
        model, (p_params, _) = singleton_models[name], current_node.serialize()[1]
        if p_params:
            params_tensor = t(torch.tensor(p_params, dtype=torch.float32)).unsqueeze(0)
            _, reconstruction = model(params_tensor)
            error = torch.max(torch.abs(reconstruction - params_tensor)).item()
            debug_info(f"{indent}[SINGLETON CHECK] {name}, error={error:.4f}")
            if error < error_threshold:
                encoding, _ = model(params_tensor)
                debug_success(f"{indent}[SINGLETON ABSTRACTION CREATED] {name}")
                return Abstraction(name, encoding.squeeze().tolist(), model, children=child_nodes)

    # 5. No abstraction, return rebuilt node
    debug_info(f"{indent}[RETURN NODE] {type(current_node).__name__}")
    return current_node


# In[30]:


# Load DSL shapes

chair_directory = dataset_directory / "Chair"
json_files = sorted(list(chair_directory.glob("*.json")))

try:
    debug_info("Starting to load DSL shapes from JSON files...")
    all_dsl_shapes = [parse_json_to_dsl(Path(f).read_text()) for f in json_files[:1000]]
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


# In[36]:


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


# In[37]:


print(abstracted_chair)


# In[38]:


print(random_chair)


# In[ ]:




