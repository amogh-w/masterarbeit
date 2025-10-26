import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import textwrap
import matplotlib.pyplot as plt
from pathlib import Path
# from tqdm import tqdm
from tqdm.auto import tqdm

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTOENCODER_LAYER_1_SIZE = 32
AUTOENCODER_LAYER_2_SIZE = 16
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MIN_EXAMPLES_FOR_ABSTRACTION = 100
RETRAIN_ITERATIONS = 1
ERROR_THRESHOLD = 0.05

# ==============================================================================
# --- CORE LOGIC ---
# ==============================================================================


def t(tensor):
    """Moves a tensor to the configured device.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor on DEVICE.
    """
    return tensor.to(DEVICE)


class Autoencoder(nn.Module):
    """Simple fully-connected autoencoder with configurable hidden dimension."""

    def __init__(self, input_dim, hidden_dim):
        """Initializes encoder and decoder layers.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of latent space.
        """
        super().__init__()
        debug_info(f"Initializing Autoencoder: input_dim={input_dim}, hidden_dim={hidden_dim}")

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
        """Performs a forward pass through the encoder and decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Latent representation and reconstruction.
        """
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded


def instantiate_pattern(pattern_name, params, children):
    """Rebuilds a DSL node from a pattern name, parameters, and children.

    Handles both singleton and parent-child pair patterns.

    Args:
        pattern_name (str): Name of the pattern (e.g., 'Scale', 'Translate(Box)').
        params (list): Parameters for the node.
        children (list): Child nodes.

    Returns:
        DSL node: Reconstructed DSL node.

    Raises:
        ValueError: If a pattern name is unrecognized or children are missing.
    """
    debug_info(
        f"[INSTANTIATE] Pattern: '{pattern_name}' | params={params[:10]}{'...' if len(params) > 10 else ''} | #children={len(children)}"
    )

    # Predefined mapping for pair patterns: number of parent parameters
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
            return SymRef(child_node, plane_normal=params[:3], point_on_plane=params[3:])
        if NodeClass == SymRot:
            return SymRot(child_node, axis=params[:3], center=params[3:], n_fold=-1)
        if NodeClass == SymTrans:
            return SymTrans(child_node, end_point=params, n_fold=-1)

        return NodeClass(child_node, params)

    # --- PAIR RECONSTRUCTION ---
    elif pattern_name in param_split:
        p_len = param_split[pattern_name]
        p_params, c_params = params[:p_len], params[p_len:]
        parent_name, child_name = pattern_name.split("(")[0], pattern_name.split("(")[1][:-1]
        ParentClass, ChildClass = globals()[parent_name], globals()[child_name]

        # --- Build child node ---
        grandchild_node = children[0] if children else Box(-1)
        if ChildClass == Box:
            child_node = Box(-1)
        elif ChildClass == Union:
            child_node = Union(children[0], children[1]) if len(children) >= 2 else Box(-1)
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
            return Union(child_node, children[1]) if len(children) >= 2 else Box(-1)
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


class Abstraction:
    """Represents a compressed DSL pattern using an autoencoder."""

    def __init__(self, pattern_name, compressed_params, model, children=None):
        """Initializes an abstraction node.

        Args:
            pattern_name (str): Name of the pattern.
            compressed_params (list): Compressed latent representation.
            model (Autoencoder): Trained autoencoder model for reconstruction.
            children (list, optional): Child nodes. Defaults to None.
        """
        self.pattern_name = pattern_name
        self.compressed_params = compressed_params
        self.model = model
        self.children = children if children is not None else []

    def __str__(self):
        """Returns a string representation of the abstraction."""
        header = f"Abs({self.pattern_name}, dim={len(self.compressed_params)})"
        param_str = f"compressed_params={self.compressed_params}"
        
        if not self.children:
            return f"{header}, {param_str}"
        else:
            child_strs = [textwrap.indent(str(c), "    ") for c in self.children]
            return f"{header}, {param_str}(\n" + ",\n".join(child_strs) + "\n)"

    __repr__ = __str__

    def expand(self):
        """Reconstructs the full DSL node from compressed parameters.

        Returns:
            DSL node: Expanded DSL node.
        """
        debug_info(f"Expanding abstraction: {self}")
        if not self.compressed_params:
            debug_error("No compressed params found. Returning Box(-1).")
            return Box(-1)

        self.model.eval()
        with torch.no_grad():
            params_tensor = t(torch.tensor(self.compressed_params, dtype=torch.float32)).unsqueeze(0)
            reconstructed_params = self.model.decoder(params_tensor).squeeze().tolist()
            debug_success(f"Reconstructed params: {reconstructed_params}")

        rebuilt_node = instantiate_pattern(self.pattern_name, reconstructed_params, self.children)
        debug_success(f"Successfully rebuilt node: {rebuilt_node}")
        return rebuilt_node.expand()


def prepare_autoencoder_train_data(parameters, mask, batch_size=BATCH_SIZE):
    """Creates a DataLoader from parameters and a boolean mask.

    Args:
        parameters (list): Parameter vectors.
        mask (torch.Tensor): Boolean mask selecting examples.
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.

    Returns:
        DataLoader: PyTorch DataLoader for training.
    """
    tensor = t(torch.tensor(parameters, dtype=torch.float32))
    if mask.shape[0] != tensor.shape[0]:
        raise ValueError(f"Mask size {mask.shape[0]} != data size {tensor.shape[0]}")
    dataloader = DataLoader(TensorDataset(tensor[mask]), batch_size=batch_size, shuffle=True)
    return dataloader


def is_well_explained(model, parameters_tensor, error_threshold=ERROR_THRESHOLD):
    """Determines if a parameter set is well reconstructed by an autoencoder.

    Args:
        model (Autoencoder): Trained autoencoder.
        parameters_tensor (torch.Tensor): Input parameter vectors.
        error_threshold (float, optional): Maximum reconstruction error allowed.

    Returns:
        torch.BoolTensor: Boolean mask of well-explained examples.
    """
    model.eval()
    with torch.no_grad():
        _, reconstructions = model(parameters_tensor)
        error, _ = torch.max(torch.abs(reconstructions - parameters_tensor), dim=-1)
    well_explained = error < error_threshold
    return well_explained


# --- Helper to create safe filenames (same as models) ---
def make_safe_filename(name: str, suffix: str = "") -> str:
    """
    Creates a safe filename.
    If suffix is provided, it's appended as an extension.
    e.g., ("Translate(Rotate)", "pth") -> "translate_rotate.pth"
    e.g., ("My Model", "loss_chart.png") -> "my_model.loss_chart.png"
    """
    safe = re.sub(r'[^\w\-]+', '_', name)
    safe = re.sub(r'_+', '_', safe).strip('_').lower()
    if suffix:
        # Add a dot before the suffix to make it a file extension
        return f"{safe}.{suffix}"
    # Return just the safe name if no suffix
    return safe

# from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.optim import AdamW
import torch.nn as nn

def train_autoencoder(model, dataloader, model_name, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Trains an autoencoder with a single tqdm progress bar and plots training loss.

    Args:
        model (Autoencoder): Model to train.
        dataloader (DataLoader): Training data.
        model_name (str): Name for plotting/saving the loss chart.
        epochs (int, optional): Number of epochs.
        lr (float, optional): Learning rate.

    Returns:
        Autoencoder: Trained autoencoder.
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    epoch_losses = []

    total_batches = epochs * len(dataloader)
    global_batch = 0

    # Single outer tqdm
    with tqdm(total=total_batches, desc=f"Training {model_name}", unit="batch") as pbar:
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
                global_batch += 1

                avg_batch_loss = epoch_loss / ((global_batch - 1) % len(dataloader) + 1)
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs}",
                    "batch_loss": f"{loss.item():.6f}",
                    "avg_epoch_loss": f"{avg_batch_loss:.6f}"
                })
                pbar.update(1)

            avg_epoch_loss = epoch_loss / len(dataloader.dataset)
            epoch_losses.append(avg_epoch_loss)
            # tqdm.write(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.6f}")

    # Plot epoch losses
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', label='Training Loss')
    ax.set_title(f"Training Loss for Model: {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average MSE Loss")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    if epochs <= 25:
        ax.set_xticks(range(1, epochs + 1))
    fig.tight_layout()

    # Save loss chart using safe filename logic
    try:
        save_dir = Path(__file__).parent / "saved" / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = make_safe_filename(model_name, suffix="loss_chart") + ".png"
        fig.savefig(save_dir / safe_filename)
    except Exception as e:
        debug_error(f"Failed to save loss chart: {e}")

    plt.show()
    plt.close(fig)

    return model


def find_abstractions(structures, structure_type="PATTERNS", min_examples=MIN_EXAMPLES_FOR_ABSTRACTION,
                      retrain_iterations=RETRAIN_ITERATIONS, error_threshold=ERROR_THRESHOLD, epochs=EPOCHS):
    """Trains autoencoders for each structure type and returns models.

    Args:
        structures (dict): Mapping pattern names to parameter lists.
        structure_type (str, optional): Type description for logging.
        min_examples (int, optional): Minimum examples to train a model.
        retrain_iterations (int, optional): Number of retraining passes.
        error_threshold (float, optional): Maximum reconstruction error allowed.
        epochs (int, optional): Training epochs.

    Returns:
        dict: Trained models keyed by pattern name.
    """
    trained_models = {}
    sorted_structures = sorted(structures.items(), key=lambda item: len(item[1]), reverse=True)
    for name, parameters in sorted_structures:
        if len(parameters) < min_examples:
            continue
        num_params = len(parameters[0])
        if num_params <= 1:
            continue
        params_tensor = t(torch.tensor(parameters, dtype=torch.float32))
        mask = torch.ones(len(parameters), dtype=torch.bool)
        for iteration in range(retrain_iterations):
            if not mask.any():
                break
            dataloader = prepare_autoencoder_train_data(parameters, mask=mask)
            model = Autoencoder(num_params, max(1, num_params - 1)).to(DEVICE)
            model = train_autoencoder(model, dataloader, model_name=name, epochs=epochs)
            mask = is_well_explained(model, params_tensor, error_threshold)
        trained_models[name] = model
    return trained_models


def integrate_abstractions(node, singleton_models, pair_models, error_threshold=ERROR_THRESHOLD, depth=0, detailed_debug=False):
    """
    Recursively abstracts a DSL tree with trained models.
    *** UPDATED: All detailed logging is now behind the detailed_debug flag. ***
    """
    indent = "  " * depth
    
    # --- Always log entry ---
    node_name_repr = f"Abs({node.pattern_name})" if isinstance(node, Abstraction) else type(node).__name__
    if detailed_debug: debug_info(f"{indent}[{depth}] Processing: {node_name_repr}")

    if isinstance(node, Abstraction):
        if detailed_debug: debug_info(f"{indent}[{depth}] Node is already an Abstraction. Recursing on its {len(node.children)} children.")
        rebuilt_children = [
            integrate_abstractions(c, singleton_models, pair_models, error_threshold, depth + 1, detailed_debug)
            for c in node.children
        ]
        node.children = rebuilt_children
        if detailed_debug: debug_info(f"{indent}[{depth}] Returning existing Abstraction node: {node_name_repr}")
        return node
    
    if not hasattr(node, "serialize"):
        # --- Always log critical errors ---
        debug_error(f"{indent}[{depth}] Node {type(node).__name__} has no 'serialize' method. Returning as-is.")
        return node
        
    # --- RECURSIVE STEP ---
    _, (_, children) = node.serialize()
    valid_children = [c for c in children if hasattr(c, "serialize") or isinstance(c, Abstraction)]
    if detailed_debug: debug_info(f"{indent}[{depth}] Recursing on {len(valid_children)} children for {node_name_repr}...")
    rebuilt_children = [
        integrate_abstractions(c, singleton_models, pair_models, error_threshold, depth + 1, detailed_debug)
        if hasattr(c, "serialize") or isinstance(c, Abstraction) else c
        for c in children
    ]
    if detailed_debug: debug_info(f"{indent}[{depth}] All children for {node_name_repr} processed.")

    # --- REBUILD ---
    current_node = None
    try:
        # Rebuild logic... (same as before)
        if isinstance(node, Union):
            current_node = Union(rebuilt_children[0], rebuilt_children[1])
        elif isinstance(node, SymRef):
            current_node = SymRef(rebuilt_children[0], plane_normal=node.plane, point_on_plane=node.point_on_plane)
        elif isinstance(node, SymRot):
            current_node = SymRot(rebuilt_children[0], axis=node.axis, center=node.center, n_fold=node.n)
        elif isinstance(node, SymTrans):
            current_node = SymTrans(rebuilt_children[0], end_point=node.end_point, n_fold=node.n)
        elif hasattr(node, "child"):
            kwargs = {k: v for k, v in node.__dict__.items() if k != "child"}
            current_node = type(node)(rebuilt_children[0], **kwargs)
        else:
            current_node = type(node)(*rebuilt_children) # Box
        if detailed_debug: debug_info(f"{indent}[{depth}] Rebuilt {type(current_node).__name__} with new children.")
    except Exception as e:
        # --- Always log critical errors ---
        debug_error(f"{indent}[{depth}] FAILED to rebuild node {node_name_repr}: {e}. Returning original node.")
        return node # Return the original node if rebuild fails

    # --- PAIR ABSTRACTION ATTEMPT ---
    child_nodes = [c for c in rebuilt_children if hasattr(c, "serialize") or isinstance(c, Abstraction)]
    
    if len(child_nodes) == 1:
        child_node = child_nodes[0]
        child_name = ""
        c_params = None
        c_children = []

        # Logic to get child info... (same as before)
        if isinstance(child_node, Abstraction):
            child_name = f"Abs({child_node.pattern_name})"
            c_params = child_node.compressed_params
            c_children = child_node.children
        elif hasattr(child_node, "serialize"):
            child_name = type(child_node).__name__
            c_params, c_children = child_node.serialize()[1]
        
        if child_name:
            pair_sig = f"{type(current_node).__name__}({child_name})"
            if detailed_debug: debug_info(f"{indent}[{depth}] -> Checking for PAIR abstraction: {pair_sig}")
            
            if pair_sig in pair_models:
                if detailed_debug: debug_info(f"{indent}[{depth}]    Found model for {pair_sig}.")
                model = pair_models[pair_sig]
                p_params, _ = current_node.serialize()[1]
                p_params_list = list(p_params or [])
                c_params_list = list(c_params or [])
                combined = t(torch.tensor(p_params_list + c_params_list, dtype=torch.float32)).unsqueeze(0)
                
                if combined.shape[1] == 0:
                    if detailed_debug: debug_info(f"{indent}[{depth}]    Skipping (no params to abstract).")
                elif combined.shape[1] != model.encoder[0].in_features:
                    # --- Always log critical errors ---
                    debug_error(f"{indent}[{depth}]    Shape mismatch for {pair_sig}: Model expects {model.encoder[0].in_features}, got {combined.shape[1]}")
                else:
                    _, reconstruction = model(combined)
                    error = torch.max(torch.abs(reconstruction - combined)).item()
                    if detailed_debug: debug_info(f"{indent}[{depth}]    Reconstruction error: {error:.4f} (Threshold: {error_threshold})")
                    
                    if error < error_threshold:
                        # --- Log success only if detailed ---
                        if detailed_debug: debug_success(f"{indent}[{depth}]    SUCCESS: Applying PAIR abstraction for {pair_sig}!")
                        encoding, _ = model(combined)
                        grandchildren = [c for c in c_children if hasattr(c, "serialize") or isinstance(c, Abstraction)]
                        if detailed_debug: debug_info(f"{indent}[{depth}] Returning NEW Abstraction node: {pair_sig}")
                        return Abstraction(pair_sig, encoding.squeeze().tolist(), model, children=grandchildren)
            else:
                 if detailed_debug: debug_info(f"{indent}[{depth}]    No model found for {pair_sig}.")

    # --- SINGLETON ABSTRACTION ATTEMPT ---
    name = type(current_node).__name__
    if detailed_debug: debug_info(f"{indent}[{depth}] -> Checking for SINGLETON abstraction: {name}")
    
    if name in singleton_models:
        if detailed_debug: debug_info(f"{indent}[{depth}]    Found model for {name}.")
        model, (p_params, _) = singleton_models[name], current_node.serialize()[1]
        
        if not p_params:
            if detailed_debug: debug_info(f"{indent}[{depth}]    Skipping (no params to abstract, e.g., Union).")
            if detailed_debug: debug_info(f"{indent}[{depth}] Returning rebuilt node: {name}")
            return current_node
            
        params_tensor = t(torch.tensor(p_params, dtype=torch.float32)).unsqueeze(0)
        
        if params_tensor.shape[1] != model.encoder[0].in_features:
            # --- Always log critical errors ---
            debug_error(f"{indent}[{depth}]    Shape mismatch for {name}: Model expects {model.encoder[0].in_features}, got {params_tensor.shape[1]}")
        else:
            _, reconstruction = model(params_tensor)
            error = torch.max(torch.abs(reconstruction - params_tensor)).item()
            if detailed_debug: debug_info(f"{indent}[{depth}]    Reconstruction error: {error:.4f} (Threshold: {error_threshold})")
            
            if error < error_threshold:
                # --- Log success only if detailed ---
                if detailed_debug: debug_success(f"{indent}[{depth}]    SUCCESS: Applying SINGLETON abstraction for {name}!")
                encoding, _ = model(params_tensor)
                if detailed_debug: debug_info(f"{indent}[{depth}] Returning NEW Abstraction node: {name}")
                return Abstraction(name, encoding.squeeze().tolist(), model, children=child_nodes)
    else:
        if detailed_debug: debug_info(f"{indent}[{depth}]    No model found for {name}.")

    # --- EXIT (No Abstraction Applied) ---
    if detailed_debug: debug_info(f"{indent}[{depth}] No abstraction applied. Returning rebuilt {name}.")
    return current_node