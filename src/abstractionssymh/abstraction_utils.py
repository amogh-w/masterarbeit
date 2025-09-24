import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import textwrap

import matplotlib.pyplot as plt
from IPython.display import display
from pathlib import Path

from abstractionssymh.dsl_nodes import (
    Box,
    Scale,
    Rotate,
    Translate,
    Union,
    SymRef,
    SymRot,
    SymTrans,
)
from abstractionssymh.debug_utils import debug_info, debug_error, debug_success

# ==============================================================================
# --- CONFIGURABLE PARAMETERS ---
# ==============================================================================

# --- Hardware Configuration ---
# Set the device to use for tensor computations ('cuda' for GPU, 'cpu' for CPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Autoencoder Architecture ---
# Defines the sizes of the intermediate layers in the autoencoder.
# The network structure will be: input -> LAYER_1_SIZE -> LAYER_2_SIZE -> latent_space
AUTOENCODER_LAYER_1_SIZE = 32
AUTOENCODER_LAYER_2_SIZE = 16

# --- Training Hyperparameters ---
# Number of epochs to train each autoencoder model.
EPOCHS = 50
# Batch size for training the autoencoder.
BATCH_SIZE = 64
# Learning rate for the AdamW optimizer.
LEARNING_RATE = 1e-3

# --- Abstraction Finding Parameters ---
# Minimum number of examples required to attempt creating an abstraction for a pattern.
MIN_EXAMPLES_FOR_ABSTRACTION = 100
# Number of times to retrain the autoencoder, filtering out poorly-explained examples each time.
RETRAIN_ITERATIONS = 1
# Maximum reconstruction error for an example to be considered "well-explained" by the autoencoder.
# This threshold is used both for filtering during training and for deciding when to abstract a node.
ERROR_THRESHOLD = 0.05

# ==============================================================================
# --- CORE LOGIC ---
# ==============================================================================


def t(tensor):
    """Convenience function to move a tensor to the configured device."""
    return tensor.to(DEVICE)


class Autoencoder(nn.Module):
    """Simple fully-connected autoencoder with configurable hidden dim."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        debug_info(
            f"Initializing Autoencoder: input_dim={input_dim}, hidden_dim={hidden_dim}"
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, AUTOENCODER_LAYER_1_SIZE),
            nn.ReLU(),
            nn.Linear(AUTOENCODER_LAYER_1_SIZE, AUTOENCODER_LAYER_2_SIZE),
            nn.ReLU(),
            nn.Linear(AUTOENCODER_LAYER_2_SIZE, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, AUTOENCODER_LAYER_2_SIZE),
            nn.ReLU(),
            nn.Linear(AUTOENCODER_LAYER_2_SIZE, AUTOENCODER_LAYER_1_SIZE),
            nn.ReLU(),
            nn.Linear(AUTOENCODER_LAYER_1_SIZE, input_dim),
        )

        debug_success("Autoencoder initialized successfully.")

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded


def instantiate_pattern(pattern_name, params, children):
    """
    Rebuilds a DSL sub-tree from its pattern name, parameters, and preserved children.
    Handles both SINGLETON and PAIR patterns.
    """
    debug_info(
        f"[INSTANTIATE] Pattern: '{pattern_name}' | params={params[:10]}{'...' if len(params)>10 else ''} | #children={len(children)}"
    )

    # Registry defining parent parameter lengths for known PAIRS
    param_split = {
        "Scale(Box)": 3,
        "Rotate(Scale)": 4,
        "Translate(Rotate)": 3,
        "Union(Translate)": 0,
        "Union(SymRef)": 0,
        "Union(SymRot)": 0,
        "Union(SymTrans)": 0,
        "SymRef(Union)": 6,
        "SymRef(Translate)": 6,
        "SymRot(Union)": 6,
        "SymRot(Translate)": 6,
        "SymTrans(Translate)": 3,
    }

    # --- SINGLETON RECONSTRUCTION ---
    if pattern_name in ["Scale", "Rotate", "Translate", "SymRef", "SymRot", "SymTrans"]:
        NodeClass = globals()[pattern_name]
        child_node = children[0] if children else Box(-1)

        if NodeClass == SymRef:
            debug_info(f"  [SINGLETON] SymRef with params length {len(params)}")
            return SymRef(
                child_node, plane_normal=params[:3], point_on_plane=params[3:]
            )
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
        parent_name, child_name = (
            pattern_name.split("(")[0],
            pattern_name.split("(")[1][:-1],
        )
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
            child_node = SymRef(
                grandchild_node, plane_normal=c_params[:3], point_on_plane=c_params[3:]
            )
        elif ChildClass == SymRot:
            child_node = SymRot(
                grandchild_node, axis=c_params[:3], center=c_params[3:], n_fold=-1
            )
        elif ChildClass == SymTrans:
            child_node = SymTrans(grandchild_node, end_point=c_params, n_fold=-1)
        else:
            child_node = ChildClass(grandchild_node, c_params)

        # --- Build parent node ---
        if ParentClass == Union:
            if len(children) >= 2:
                return Union(child_node, children[1])
            else:
                debug_error(
                    f"  [PAIR] Union parent with missing second child in {pattern_name}"
                )
                return Box(-1)
        elif ParentClass == SymRef:
            return SymRef(
                child_node, plane_normal=p_params[:3], point_on_plane=p_params[3:]
            )
        elif ParentClass == SymRot:
            return SymRot(child_node, axis=p_params[:3], center=p_params[3:], n_fold=-1)
        elif ParentClass == SymTrans:
            return SymTrans(child_node, end_point=p_params, n_fold=-1)
        else:
            return ParentClass(child_node, p_params)

    debug_error(f"[UNKNOWN PATTERN] '{pattern_name}', defaulting to Box(-1)")
    return Box(-1)


class Abstraction:
    def __init__(self, pattern_name, compressed_params, model, children=None):
        self.pattern_name, self.compressed_params, self.model = (
            pattern_name,
            compressed_params,
            model,
        )
        self.children = children if children is not None else []

    def __str__(self):
        header = f"Abs({self.pattern_name}, dim={len(self.compressed_params)})"
        if not self.children:
            return header
        else:
            child_strs = [textwrap.indent(str(c), "    ") for c in self.children]
            return f"{header}(\n" + ",\n".join(child_strs) + "\n)"

    __repr__ = __str__

    def expand(self):
        """Reconstructs the full DSL node from compressed parameters."""
        debug_info(f"Expanding abstraction: {self}")
        if not self.compressed_params:
            debug_error("No compressed params found. Returning Box(-1).")
            return Box(-1)

        self.model.eval()
        with torch.no_grad():
            params_tensor = t(
                torch.tensor(self.compressed_params, dtype=torch.float32)
            ).unsqueeze(0)
            debug_info(f"Compressed params tensor shape: {tuple(params_tensor.shape)}")
            reconstructed_params = self.model.decoder(params_tensor).squeeze().tolist()
            debug_success(f"Reconstructed params: {reconstructed_params}")

        rebuilt_node = instantiate_pattern(
            self.pattern_name, reconstructed_params, self.children
        )
        debug_success(f"Successfully rebuilt node: {rebuilt_node}")
        return rebuilt_node.expand()


def prepare_autoencoder_train_data(parameters, mask, batch_size=BATCH_SIZE):
    """Creates a DataLoader from parameters and a boolean mask."""
    debug_info(
        f"Preparing DataLoader: batch_size={batch_size}, mask sum={mask.sum().item()}"
    )
    tensor = t(torch.tensor(parameters, dtype=torch.float32))
    if mask.shape[0] != tensor.shape[0]:
        debug_error(
            f"Mask size {mask.shape[0]} does not match data size {tensor.shape[0]}"
        )
        raise ValueError(f"Mask size {mask.shape[0]} != data size {tensor.shape[0]}")
    dataloader = DataLoader(
        TensorDataset(tensor[mask]), batch_size=batch_size, shuffle=True
    )
    debug_success("DataLoader prepared successfully.")
    return dataloader


def is_well_explained(model, parameters_tensor, error_threshold=ERROR_THRESHOLD):
    """Checks if parameters are well reconstructed by the autoencoder."""
    model.eval()
    with torch.no_grad():
        _, reconstructions = model(parameters_tensor)
        error, _ = torch.max(torch.abs(reconstructions - parameters_tensor), dim=-1)
    debug_info(f"Max reconstruction error per sample: {error}")
    well_explained = error < error_threshold
    debug_info(
        f"Number of well-explained examples: {well_explained.sum().item()}/{len(well_explained)}"
    )
    return well_explained


def train_autoencoder(model, dataloader, model_name, epochs=EPOCHS, lr=LEARNING_RATE):
    """Trains an autoencoder and plots and saves its loss curve."""
    debug_info(f"Training autoencoder for {epochs} epochs, learning rate={lr}")
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    
    epoch_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            _, x_rec = model(x)
            loss = loss_fn(x_rec, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        
        avg_epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_losses.append(avg_epoch_loss)
        debug_info(
            f"Epoch {epoch+1}/{epochs} - Average loss: {avg_epoch_loss:.6f}"
        )

    debug_success("Autoencoder training complete.")
    
    # --- PLOTTING LOGIC ---
    debug_info(f"Plotting training loss for model: {model_name}")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', label='Training Loss')
    ax.set_title(f"Training Loss for Model: {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average MSE Loss")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    # Ensure integer ticks for epochs, especially for smaller epoch counts
    if epochs <= 25:
        ax.set_xticks(range(1, epochs + 1))
    fig.tight_layout()

    # --- SAVING THE PLOT ---
    try:
        # Create a 'saved' directory relative to this file's location
        save_dir = Path(__file__).parent / "saved"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize the model name to create a valid filename
        safe_filename = model_name.replace("(", "_").replace(")", "").replace("/", "_") + "_loss_chart.png"
        save_path = save_dir / safe_filename
        
        fig.savefig(save_path)
        debug_success(f"Loss chart saved to: {save_path}")
    except Exception as e:
        debug_error(f"Failed to save loss chart: {e}")

    plt.show() # Display the plot in interactive environments
    plt.close(fig) # Close the figure to free up memory

    return model


def find_abstractions(
    structures,
    structure_type="PATTERNS",
    min_examples=MIN_EXAMPLES_FOR_ABSTRACTION,
    retrain_iterations=RETRAIN_ITERATIONS,
    error_threshold=ERROR_THRESHOLD,
    epochs=EPOCHS,
):
    """Trains autoencoders for each structure type and returns models."""
    trained_models = {}
    sorted_structures = sorted(
        structures.items(), key=lambda item: len(item[1]), reverse=True
    )

    debug_info(
        f"Finding abstractions for {len(sorted_structures)} structures of type {structure_type}"
    )

    for name, parameters in sorted_structures:
        if len(parameters) < min_examples:
            debug_info(f"Skipping {name} (only {len(parameters)} examples)")
            continue

        num_params = len(parameters[0])
        if num_params <= 1:
            debug_info(f"Skipping {name} (too few params: {num_params})")
            continue

        debug_success(
            f"Training model for {name} with {len(parameters)} samples, {num_params} params"
        )
        params_tensor = t(torch.tensor(parameters, dtype=torch.float32))
        mask = torch.ones(len(parameters), dtype=torch.bool)

        for iteration in range(retrain_iterations):
            if not mask.any():
                debug_info(f"All {name} examples filtered out at iteration {iteration}")
                break

            debug_info(f"Iteration {iteration+1}/{retrain_iterations} for {name}")
            dataloader = prepare_autoencoder_train_data(parameters, mask=mask)
            model = Autoencoder(num_params, max(1, num_params - 1)).to(DEVICE)
            
            # --- FIX: Pass the 'name' of the pattern as the model_name for plotting ---
            model = train_autoencoder(model, dataloader, model_name=name, epochs=epochs)

            mask = is_well_explained(model, params_tensor, error_threshold)

        trained_models[name] = model
        debug_success(
            f"Model training finished for {name}. Total well-explained examples: {mask.sum().item()}/{len(parameters)}"
        )

    debug_success(f"Finished training {len(trained_models)} {structure_type}")
    return trained_models


def integrate_abstractions(
    node, singleton_models, pair_models, error_threshold=ERROR_THRESHOLD, depth=0
):
    """Recursively traverses a DSL tree and replaces well-explained patterns with Abstraction nodes."""
    indent = "  " * depth

    if isinstance(node, Abstraction):
        debug_info(
            f"{indent}[Abstraction] Node already abstracted: {node.pattern_name}"
        )
        return node

    # 1. Recurse to children first
    _, (_, children) = node.serialize()
    rebuilt_children = [
        (
            integrate_abstractions(
                c, singleton_models, pair_models, error_threshold, depth + 1
            )
            if hasattr(c, "serialize")
            else c
        )
        for c in children
    ]

    # 2. Rebuild current node with potentially abstracted children
    try:
        if isinstance(node, Union):
            current_node = Union(rebuilt_children[0], rebuilt_children[1])
        elif isinstance(node, SymRef):
            current_node = SymRef(
                rebuilt_children[0],
                plane_normal=node.plane,
                point_on_plane=node.point_on_plane,
            )
        elif isinstance(node, SymRot):
            current_node = SymRot(
                rebuilt_children[0], axis=node.axis, center=node.center, n_fold=node.n
            )
        elif isinstance(node, SymTrans):
            current_node = SymTrans(
                rebuilt_children[0], end_point=node.end_point, n_fold=node.n
            )
        elif hasattr(node, "child"):
            kwargs = {k: v for k, v in node.__dict__.items() if k != "child"}
            current_node = type(node)(rebuilt_children[0], **kwargs)
        else:
            current_node = type(node)(*rebuilt_children)
        debug_info(
            f"{indent}[Rebuilt] {type(current_node).__name__} with {len(rebuilt_children)} children"
        )
    except Exception as e:
        debug_error(
            f"{indent}[Rebuild FAILED] {type(node).__name__} at depth {depth}: {e}"
        )
        return node

    # 3. Try to abstract the current node as part of a PAIR
    child_nodes = [c for c in rebuilt_children if hasattr(c, "serialize")]
    if len(child_nodes) == 1:
        child_node = child_nodes[0]
        if not isinstance(child_node, Abstraction):
            pair_sig = f"{type(current_node).__name__}({type(child_node).__name__})"
            if pair_sig in pair_models:
                model, (p_params, _), (c_params, c_children) = (
                    pair_models[pair_sig],
                    current_node.serialize()[1],
                    child_node.serialize()[1],
                )
                if p_params + c_params:
                    combined = t(
                        torch.tensor(p_params + c_params, dtype=torch.float32)
                    ).unsqueeze(0)
                    _, reconstruction = model(combined)
                    error = torch.max(torch.abs(reconstruction - combined)).item()
                    debug_info(f"{indent}[PAIR CHECK] {pair_sig}, error={error:.4f}")
                    if error < error_threshold:
                        encoding, _ = model(combined)
                        grandchildren = [
                            c for c in c_children if hasattr(c, "serialize")
                        ]
                        debug_success(f"{indent}[PAIR ABSTRACTION CREATED] {pair_sig}")
                        return Abstraction(
                            pair_sig,
                            encoding.squeeze().tolist(),
                            model,
                            children=grandchildren,
                        )

    # 4. Try to abstract the current node as a SINGLETON
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
                return Abstraction(
                    name, encoding.squeeze().tolist(), model, children=child_nodes
                )

    # 5. No abstraction was applicable, return the rebuilt node
    debug_info(f"{indent}[RETURN NODE] {type(current_node).__name__}")
    return current_node