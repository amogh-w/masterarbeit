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
        *** MODIFIED to store input_dim. ***

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of latent space.
        """
        super().__init__()
        self.input_dim = input_dim  # <--- ADD THIS LINE
        self.hidden_dim = hidden_dim # <--- ADD THIS LINE
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
    
# === In: abstraction_utils.py ===
# (Place this after the Autoencoder class)

class PCAModel(nn.Module):
    """
    A model that uses PCA for encoding/decoding, designed as a 
    drop-in replacement for the Autoencoder.
    
    It expects *normalized* data for fitting and forward pass, and
    stores the normalization stats just like the Autoencoder.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # We register 'components' as a buffer. It's part of the model's
        # state and moves with .to(DEVICE), but is not a trainable parameter.
        self.register_buffer('components', torch.randn(input_dim, hidden_dim))
        
        # These will be attached by the find_abstractions_pca function
        self.data_mean_ = None
        self.data_std_ = None

    def fit(self, normalized_data_tensor):
        """
        'Trains' the PCA model by calculating the principal components.
        
        Args:
            normalized_data_tensor (torch.Tensor): The *normalized* training data.
        """
        if normalized_data_tensor.shape[0] < self.input_dim:
            debug_error(f"PCA fit error: Not enough samples ({normalized_data_tensor.shape[0]}) to compute covariance for {self.input_dim} features.")
            # Keep random components as a fallback, but this model will be bad
            return
            
        # 1. Calculate covariance matrix
        # Data should be (n_samples, n_features)
        cov_matrix = torch.cov(normalized_data_tensor.T)
        
        # 2. Get eigenvalues and eigenvectors
        # eigh is for symmetric matrices (like covariance)
        # We only need the eigenvectors
        try:
            _, eigenvectors = torch.linalg.eigh(cov_matrix)
        except Exception as e:
            debug_error(f"PCA linalg.eigh failed: {e}")
            return # Keep random components

        # 3. Store the top 'hidden_dim' eigenvectors
        # Eigenvectors are sorted by eigenvalue (ascending), so we take the last ones.
        self.components = eigenvectors[:, -self.hidden_dim:]
        debug_success(f"PCA fit complete. Components shape: {self.components.shape}")
        
    def encoder(self, normalized_x):
        """Projects normalized data onto the principal components."""
        # (N, D) @ (D, K) -> (N, K)
        return normalized_x @ self.components

    def decoder(self, z):
        """Reconstructs normalized data from the latent representation."""
        # (N, K) @ (K, D) -> (N, D)  (where K, D is components.T)
        return z @ self.components.T

    def forward(self, normalized_x):
        """Performs a forward pass: encode -> decode."""
        latent = self.encoder(normalized_x)
        decoded = self.decoder(latent)
        return latent, decoded


def instantiate_pattern(pattern_name, params, children):
    """Rebuilds a DSL node from a pattern name, parameters, and children.
    FIXED: Now properly handles nested abstraction patterns.
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
        # Add the problematic nested patterns
        "Translate(Abs(Rotate(Scale)))": 3,
        "Rotate(Abs(Scale))": 4,
        "Scale(Abs(Rotate))": 3,
    }

    # --- SINGLETON RECONSTRUCTION ---
    # FIX: Check for basic singleton patterns first
    singleton_patterns = ["Scale", "Rotate", "Translate", "SymRef", "SymRot", "SymTrans"]
    if pattern_name in singleton_patterns:
        debug_info(f"  Handling singleton pattern: {pattern_name}")
        NodeClass = globals()[pattern_name]
        child_node = children[0] if children else Box(-1)

        if NodeClass == SymRef:
            if len(params) >= 6:
                return SymRef(child_node, plane_normal=params[:3], point_on_plane=params[3:6])
            else:
                debug_error(f"SymRef needs 6 params, got {len(params)}")
                return Box(-1)
        elif NodeClass == SymRot:
            if len(params) >= 6:
                return SymRot(child_node, axis=params[:3], center=params[3:6], n_fold=-1)
            else:
                debug_error(f"SymRot needs 6 params, got {len(params)}")
                return Box(-1)
        elif NodeClass == SymTrans:
            if len(params) >= 3:
                return SymTrans(child_node, end_point=params[:3], n_fold=-1)
            else:
                debug_error(f"SymTrans needs 3 params, got {len(params)}")
                return Box(-1)
        elif NodeClass == Rotate:
            if len(params) >= 4:
                return Rotate(child_node, params[:4])  # quaternion
            else:
                debug_error(f"Rotate needs 4 params, got {len(params)}")
                return Box(-1)
        elif NodeClass == Translate:
            if len(params) >= 3:
                return Translate(child_node, params[:3])  # center
            else:
                debug_error(f"Translate needs 3 params, got {len(params)}")
                return Box(-1)
        elif NodeClass == Scale:
            if len(params) >= 3:
                return Scale(child_node, params[:3])  # lengths
            else:
                debug_error(f"Scale needs 3 params, got {len(params)}")
                return Box(-1)
        else:
            return NodeClass(child_node, params)

    # --- PAIR RECONSTRUCTION ---
    elif pattern_name in param_split:
        p_len = param_split[pattern_name]
        p_params, c_params = params[:p_len], params[p_len:]
        
        # Extract parent and child names - FIXED: Handle nested abstractions properly
        if '(' in pattern_name and ')' in pattern_name:
            # Find the first opening parenthesis and the last closing parenthesis
            first_paren = pattern_name.find('(')
            last_paren = pattern_name.rfind(')')
            
            parent_name = pattern_name[:first_paren]
            child_name = pattern_name[first_paren+1:last_paren]  # Get everything between ( and )
        else:
            debug_error(f"Invalid pair pattern: {pattern_name}")
            return Box(-1)
        
        debug_info(f"  Parent: {parent_name} with params {p_params}")
        debug_info(f"  Child: {child_name} with params {c_params}")

        try:
            ParentClass = globals()[parent_name]
            
            # Handle abstraction children - FIXED: Properly handle nested abstraction patterns
            if child_name.startswith("Abs("):
                # The entire child_name is an abstraction pattern like "Abs(Rotate(Scale))"
                child_node = Abstraction(child_name, c_params, model=None, children=children)
            else:
                # Regular DSL class child
                ChildClass = globals()[child_name]
                grandchild_node = children[0] if children else Box(-1)
                
                if ChildClass == Box:
                    child_node = Box(-1)
                elif ChildClass == Union:
                    child_node = Union(children[0], children[1]) if len(children) >= 2 else Box(-1)
                elif ChildClass == SymRef:
                    child_node = SymRef(grandchild_node, plane_normal=c_params[:3], point_on_plane=c_params[3:6]) if len(c_params) >= 6 else Box(-1)
                elif ChildClass == SymRot:
                    child_node = SymRot(grandchild_node, axis=c_params[:3], center=c_params[3:6], n_fold=-1) if len(c_params) >= 6 else Box(-1)
                elif ChildClass == SymTrans:
                    child_node = SymTrans(grandchild_node, end_point=c_params[:3], n_fold=-1) if len(c_params) >= 3 else Box(-1)
                elif ChildClass == Rotate:
                    child_node = Rotate(grandchild_node, c_params[:4]) if len(c_params) >= 4 else Box(-1)
                elif ChildClass == Translate:
                    child_node = Translate(grandchild_node, c_params[:3]) if len(c_params) >= 3 else Box(-1)
                elif ChildClass == Scale:
                    child_node = Scale(grandchild_node, c_params[:3]) if len(c_params) >= 3 else Box(-1)
                else:
                    child_node = ChildClass(grandchild_node, c_params)

            # Build parent node
            if ParentClass == Union:
                return Union(child_node, children[1]) if len(children) >= 2 else Box(-1)
            elif ParentClass == SymRef:
                return SymRef(child_node, plane_normal=p_params[:3], point_on_plane=p_params[3:6]) if len(p_params) >= 6 else Box(-1)
            elif ParentClass == SymRot:
                return SymRot(child_node, axis=p_params[:3], center=p_params[3:6], n_fold=-1) if len(p_params) >= 6 else Box(-1)
            elif ParentClass == SymTrans:
                return SymTrans(child_node, end_point=p_params[:3], n_fold=-1) if len(p_params) >= 3 else Box(-1)
            elif ParentClass == Rotate:
                return Rotate(child_node, p_params[:4]) if len(p_params) >= 4 else Box(-1)
            elif ParentClass == Translate:
                return Translate(child_node, p_params[:3]) if len(p_params) >= 3 else Box(-1)
            elif ParentClass == Scale:
                return Scale(child_node, p_params[:3]) if len(p_params) >= 3 else Box(-1)
            else:
                return ParentClass(child_node, p_params)
                
        except KeyError as e:
            debug_error(f"Unknown class in pattern {pattern_name}: {e}")
            return Box(-1)

    debug_error(f"[UNKNOWN PATTERN] '{pattern_name}', defaulting to Box(-1)")
    return Box(-1)

def expand_l2_to_l1(l2_dsl_node, singleton_models_L1, pair_models_L1, singleton_models_L2, pair_models_L2):
    """
    Expand L2 abstractions to L1 abstractions using pre-loaded models. (FIXED)
    """
    
    def _expand_node(node):
        """Recursively expand a node using the appropriate models."""
        if not hasattr(node, 'serialize') and not isinstance(node, Abstraction):
            return node
            
        if isinstance(node, Abstraction):
            pattern_name = node.pattern_name
            
            # Case 1: This is already an L1 Abstraction (wasn't abstracted by L2)
            if pattern_name in pair_models_L1 or pattern_name in singleton_models_L1:
                debug_info(f"Node is already L1 abstraction: {pattern_name}. Recursing.")
                expanded_children = [_expand_node(child) for child in node.children]
                l1_model = pair_models_L1.get(pattern_name) or singleton_models_L1.get(pattern_name)
                return Abstraction(pattern_name, node.compressed_params, l1_model, expanded_children)

            # Case 2: This is an L2 Abstraction. Find L2 model.
            model = None
            if pattern_name in pair_models_L2:
                model = pair_models_L2[pattern_name]
            elif pattern_name in singleton_models_L2:
                model = singleton_models_L2[pattern_name]
            
            if not model:
                debug_error(f"No L1 or L2 model found for: {pattern_name}")
                if node.children:
                    return _expand_node(node.children[0]) # Expand first child
                else:
                    return Box(-1)

            # It's an L2 model, so decode it
            debug_info(f"Expanding L2 abstraction: {pattern_name}")
            model.eval()
            with torch.no_grad():
                params_tensor = t(torch.tensor(node.compressed_params, dtype=torch.float32)).unsqueeze(0)
                
                # 1. Decoder outputs *normalized* parameters
                normalized_reconstruction = model.decoder(params_tensor)
                
                # 2. Un-normalize
                reconstructed_params_tensor = (normalized_reconstruction * model.data_std_) + model.data_mean_

                reconstructed_params = reconstructed_params_tensor.squeeze().tolist()
            
            # Recursively expand children first
            expanded_children = [_expand_node(child) for child in node.children]
            
            # Manually instantiate the L1 structure based on the L2 pattern
            
            # --- L2 PAIR PATTERNS ---
            if pattern_name == "Translate(Abs(Rotate(Scale)))":
                # L0 Translate params = 3
                translate_params = reconstructed_params[:3]
                # Compressed L1 Abs(Rotate(Scale)) params = 6 (from 7-1)
                l1_compressed_params = reconstructed_params[3:]
                l1_model = pair_models_L1.get("Rotate(Scale)")
                
                if l1_model:
                    l1_abs = Abstraction("Rotate(Scale)", l1_compressed_params, l1_model, expanded_children)
                    return Translate(l1_abs, translate_params)
                else:
                    debug_error("Missing L1 model for Rotate(Scale)")
                    return Box(-1)
                
            elif pattern_name == "Translate(Abs(SymRef))":
                # L0 Translate params = 3
                translate_params = reconstructed_params[:3]
                # Compressed L1 Abs(SymRef) params = 5 (from 6-1)
                l1_compressed_params = reconstructed_params[3:]
                l1_model = singleton_models_L1.get("SymRef")
                
                if l1_model:
                    l1_abs = Abstraction("SymRef", l1_compressed_params, l1_model, expanded_children)
                    return Translate(l1_abs, translate_params)
                else:
                    debug_error("Missing L1 model for SymRef")
                    return Box(-1)

            # --- L2 SINGLETON PATTERNS ---
            elif pattern_name == "Abs(Rotate(Scale))":
                l1_compressed_params = reconstructed_params # All params are for the L1 node
                l1_model = pair_models_L1.get("Rotate(Scale)")
                
                if l1_model:
                    return Abstraction("Rotate(Scale)", l1_compressed_params, l1_model, expanded_children)
                else:
                    debug_error("Missing L1 model for Rotate(Scale)")
                    return Box(-1)
                    
            elif pattern_name == "Abs(SymRef)":
                l1_compressed_params = reconstructed_params # All params are for the L1 node
                l1_model = singleton_models_L1.get("SymRef")
                
                if l1_model:
                    return Abstraction("SymRef", l1_compressed_params, l1_model, expanded_children)
                else:
                    debug_error("Missing L1 model for SymRef")
                    return Box(-1)

            else:
                debug_info(f"Unhandled L2 pattern: {pattern_name}. Falling back.")
                if expanded_children:
                    return expanded_children[0]
                else:
                    return Box(-1)

        else:
            # Regular DSL node - recurse and rebuild
            if isinstance(node, Box):
                return node
            elif isinstance(node, Translate):
                return Translate(_expand_node(node.child), node.center)
            elif isinstance(node, Rotate):
                return Rotate(_expand_node(node.child), node.quaternion)
            elif isinstance(node, Scale):
                return Scale(_expand_node(node.child), node.lengths)
            elif isinstance(node, Union):
                return Union(_expand_node(node.left), _expand_node(node.right))
            elif isinstance(node, SymRef):
                return SymRef(_expand_node(node.child), node.plane, node.point_on_plane)
            elif isinstance(node, SymRot):
                return SymRot(_expand_node(node.child), node.axis, node.center, node.n)
            elif isinstance(node, SymTrans):
                return SymTrans(_expand_node(node.child), node.end_point, node.n)
            else:
                return node
    
    debug_info("Starting L2 to L1 expansion with pre-loaded models...")
    result = _expand_node(l2_dsl_node)
    debug_success("L2 to L1 expansion completed")
    return result

def expand_l1_to_l0(l1_dsl_node, singleton_models_L1, pair_models_L1):
    """
    Expand L1 abstractions to L0 (concrete DSL) using pre-loaded L1 models. (FIXED)
    """
    
    def _expand_node(node):
        if not hasattr(node, 'serialize') and not isinstance(node, Abstraction):
            return node
            
        if isinstance(node, Abstraction):
            debug_info(f"Expanding L1 abstraction: {node.pattern_name}")
            
            # Get the appropriate L1 model
            model = None
            if node.pattern_name in pair_models_L1:
                model = pair_models_L1[node.pattern_name]
            elif node.pattern_name in singleton_models_L1:
                model = singleton_models_L1[node.pattern_name]
            
            if not model:
                debug_error(f"No L1 model found for: {node.pattern_name}")
                if node.children:
                    return _expand_node(node.children[0]) # Expand first child
                else:
                    return Box(-1)
            
            # Reconstruct parameters using L1 model
            model.eval()
            with torch.no_grad():
                # Handle empty/None params
                if not node.compressed_params:
                    reconstructed_params = []
                else:
                    params_tensor = t(torch.tensor(node.compressed_params, dtype=torch.float32)).unsqueeze(0)
                    
                    # 1. Decoder outputs *normalized* parameters
                    normalized_reconstruction = model.decoder(params_tensor)
                    
                    # 2. Un-normalize
                    reconstructed_params_tensor = (normalized_reconstruction * model.data_std_) + model.data_mean_
                    
                    reconstructed_params = reconstructed_params_tensor.squeeze().tolist()

            debug_info(f"Reconstructed params for {node.pattern_name}")
            
            # Expand children
            expanded_children = [_expand_node(child) for child in node.children]
            
            # *** --- THIS IS THE FIX --- ***
            # Use the robust instantiate_pattern function to build the L0 nodes
            try:
                concrete_node = instantiate_pattern(node.pattern_name, reconstructed_params, expanded_children)
                debug_success(f"Successfully instantiated L0 node for {node.pattern_name}")
                return concrete_node
            except Exception as e:
                debug_error(f"instantiate_pattern FAILED for {node.pattern_name}: {e}")
                if expanded_children:
                    return expanded_children[0]
                else:
                    return Box(-1)
            # *** --- END OF FIX --- ***
        
        else:
            # Regular DSL node - recurse and rebuild
            if isinstance(node, Box):
                return node
            elif isinstance(node, Translate):
                return Translate(_expand_node(node.child), node.center)
            elif isinstance(node, Rotate):
                return Rotate(_expand_node(node.child), node.quaternion)
            elif isinstance(node, Scale):
                return Scale(_expand_node(node.child), node.lengths)
            elif isinstance(node, Union):
                return Union(_expand_node(node.left), _expand_node(node.right))
            elif isinstance(node, SymRef):
                return SymRef(_expand_node(node.child), node.plane, node.point_on_plane)
            elif isinstance(node, SymRot):
                return SymRot(_expand_node(node.child), node.axis, node.center, node.n)
            elif isinstance(node, SymTrans):
                return SymTrans(_expand_node(node.child), node.end_point, node.n)
            else:
                return node
    
    debug_info("Starting L1 to L0 expansion...")
    result = _expand_node(l1_dsl_node)
    debug_success("L1 to L0 expansion completed")
    return result

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
        *** MODIFIED to un-normalize the reconstructed parameters. ***

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
            
            # 1. Decoder outputs *normalized* parameters
            normalized_reconstruction = self.model.decoder(params_tensor)
            
            # 2. Un-normalize the parameters
            reconstructed_params_tensor = (normalized_reconstruction * self.model.data_std_) + self.model.data_mean_
            
            reconstructed_params = reconstructed_params_tensor.squeeze().tolist()
            debug_success(f"Reconstructed (un-normalized) params: {reconstructed_params}")

        rebuilt_node = instantiate_pattern(self.pattern_name, reconstructed_params, self.children)
        debug_success(f"Successfully rebuilt node: {rebuilt_node}")
        return rebuilt_node.expand()


def prepare_autoencoder_train_data(parameters, mask, data_mean, data_std, batch_size=BATCH_SIZE):
    """Creates a DataLoader from parameters and a boolean mask.
    *** MODIFIED to normalize data using provided stats. ***

    Args:
        parameters (list): Parameter vectors.
        mask (torch.Tensor): Boolean mask selecting examples.
        data_mean (torch.Tensor): Mean vector for normalization.
        data_std (torch.Tensor): Standard deviation vector for normalization.
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.

    Returns:
        DataLoader: PyTorch DataLoader for training.
    """
    tensor = t(torch.tensor(parameters, dtype=torch.float32))
    if mask.shape[0] != tensor.shape[0]:
        raise ValueError(f"Mask size {mask.shape[0]} != data size {tensor.shape[0]}")
    
    # Select data first, then normalize
    masked_tensor = tensor[mask]
    
    # Move mean/std to the same device as the tensor
    data_mean_dev = data_mean.to(masked_tensor.device)
    data_std_dev = data_std.to(masked_tensor.device)
    
    normalized_tensor = (masked_tensor - data_mean_dev) / data_std_dev
    
    dataloader = DataLoader(TensorDataset(normalized_tensor), batch_size=batch_size, shuffle=True)
    return dataloader


def is_well_explained(model, parameters_tensor, error_threshold=ERROR_THRESHOLD):
    """Determines if a parameter set is well reconstructed by an autoencoder.
    *** MODIFIED to use model's mean/std for normalization. ***

    Args:
        model (Autoencoder): Trained autoencoder (must have .data_mean_ and .data_std_).
        parameters_tensor (torch.Tensor): Input parameter vectors (un-normalized).
        error_threshold (float, optional): Maximum reconstruction error allowed.

    Returns:
        torch.BoolTensor: Boolean mask of well-explained examples.
    """
    model.eval()
    with torch.no_grad():
        # Normalize the input data using the model's stored stats
        normalized_input = (parameters_tensor - model.data_mean_) / model.data_std_
        
        # Model takes normalized input and produces normalized output
        _, reconstructions = model(normalized_input)
        
        # Compare normalized reconstruction to normalized input
        error, _ = torch.max(torch.abs(reconstructions - normalized_input), dim=-1)
        
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
    *** MODIFIED to calculate and store normalization stats (mean/std) on the model. ***

    Args:
        structures (dict): Mapping pattern names to parameter lists.
        structure_type (str, optional): Type description for logging.
        min_examples (int, optional): Minimum examples to train a model.
        retrain_iterations (int, optional): Number of retraining passes.
        error_threshold (float, optional): Maximum reconstruction error allowed.
        epochs (int, optional): Training epochs.

    Returns:
        dict: Trained models (with .data_mean_ and .data_std_) keyed by pattern name.
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
        mask = torch.ones(len(parameters), dtype=torch.bool, device=DEVICE)
        
        model = None # Define model in outer scope

        for iteration in range(retrain_iterations):
            current_params = params_tensor[mask]
            if not current_params.any():
                debug_info(f"No more data for {name} in iteration {iteration+1}.")
                break
            
            # 1. Calculate stats from the *current* subset of data
            data_mean = torch.mean(current_params, dim=0)
            data_std = torch.std(current_params, dim=0)
            # Prevent division by zero for constant features
            data_std[data_std == 0] = 1.0 
            
            # 2. Prepare dataloader with normalized data
            dataloader = prepare_autoencoder_train_data(parameters, mask, data_mean, data_std)
            if len(dataloader.dataset) == 0:
                 debug_info(f"Empty dataloader for {name} in iteration {iteration+1}.")
                 break

            # 3. Create model and attach stats
            model = Autoencoder(num_params, max(1, num_params - 1)).to(DEVICE)
            model.data_mean_ = data_mean.to(DEVICE)
            model.data_std_ = data_std.to(DEVICE)
            
            # 4. Train
            model = train_autoencoder(model, dataloader, model_name=name, epochs=epochs)
            
            # 5. Re-evaluate mask on *all* data using the new model
            # is_well_explained will use the stats stored on the model
            mask = is_well_explained(model, params_tensor, error_threshold)
            
            debug_info(f"[{name} Iter {iteration+1}] Kept {mask.sum().item()}/{len(parameters)} examples.")

        # Store the final model from the last iteration
        if model is not None and mask.any():
            trained_models[name] = model
            
    return trained_models

# === In: abstraction_utils.py ===
# (Place this after the find_abstractions function)

def find_abstractions_pca(structures, structure_type="PATTERNS", min_examples=MIN_EXAMPLES_FOR_ABSTRACTION,
                          retrain_iterations=RETRAIN_ITERATIONS, error_threshold=ERROR_THRESHOLD):
    """
    Finds abstractions using PCA, as a drop-in for find_abstractions.
    It "trains" by fitting PCA components.
    
    Args:
        structures (dict): Mapping pattern names to parameter lists.
        structure_type (str, optional): Type description for logging.
        min_examples (int, optional): Minimum examples to train a model.
        retrain_iterations (int, optional): Number of retraining passes.
        error_threshold (float, optional): Maximum reconstruction error allowed.

    Returns:
        dict: Trained PCAModels (with .data_mean_ and .data_std_) keyed by pattern name.
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
        mask = torch.ones(len(parameters), dtype=torch.bool, device=DEVICE)
        
        model = None # Define model in outer scope
        
        for iteration in range(retrain_iterations):
            current_params = params_tensor[mask]
            
            if not current_params.any() or len(current_params) < 2:
                debug_info(f"No more data for {name} in iteration {iteration+1}.")
                break
            
            # 1. Calculate stats (mean/std)
            data_mean = torch.mean(current_params, dim=0)
            data_std = torch.std(current_params, dim=0)
            # Prevent division by zero for constant features
            data_std[data_std == 0] = 1.0 
            
            # 2. Normalize the current data subset
            normalized_current_params = (current_params - data_mean) / data_std
            
            # 3. Create PCA model and attach stats
            # This is your "drop 1 axis" logic
            hidden_dim = max(1, num_params - 1) 
            model = PCAModel(num_params, hidden_dim).to(DEVICE)
            model.data_mean_ = data_mean.to(DEVICE)
            model.data_std_ = data_std.to(DEVICE)
            
            # 4. "Train" (fit) the PCA model on the normalized data
            debug_info(f"Fitting PCA for {name} (Iter {iteration+1}) on {len(normalized_current_params)} samples...")
            model.fit(normalized_current_params)
            
            # 5. Re-evaluate mask on *all* data using the new model's stats
            # is_well_explained will use the .data_mean_ and .data_std_
            mask = is_well_explained(model, params_tensor, error_threshold)
            
            debug_info(f"[{name} Iter {iteration+1}] Kept {mask.sum().item()}/{len(parameters)} examples.")

        # Store the final model from the last iteration
        if model is not None and mask.any():
            trained_models[name] = model
            
    return trained_models


def integrate_abstractions(node, singleton_models, pair_models, error_threshold=ERROR_THRESHOLD, depth=0, detailed_debug=False):
    """
    Recursively abstracts a DSL tree with trained models.
    *** UPDATED: All detailed logging is now behind the detailed_debug flag. ***
    *** MODIFIED to normalize data before encoding and checking error. ***
    *** MODIFIED to check model.input_dim for compatibility. ***
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
                
                # ==========================================================
                # --- START OF FIX 1 (PAIR) ---
                # ==========================================================
                elif combined.shape[1] != model.input_dim:
                # ==========================================================
                # --- END OF FIX 1 (PAIR) ---
                # ==========================================================

                    # --- Always log critical errors ---
                    debug_error(f"{indent}[{depth}]    Shape mismatch for {pair_sig}: Model expects {model.input_dim}, got {combined.shape[1]}")
                else:
                    # ==========================================================
                    # --- START NORMALIZATION CHANGE (PAIR) ---
                    # ==========================================================
                    
                    # 1. Normalize the combined parameters
                    normalized_combined = (combined - model.data_mean_) / model.data_std_
                    
                    # 2. Get reconstruction of normalized data
                    _, reconstruction = model(normalized_combined)
                    
                    # 3. Calculate error against normalized data
                    error = torch.max(torch.abs(reconstruction - normalized_combined)).item()
                    
                    # ========================================================
                    # --- END NORMALIZATION CHANGE (PAIR) ---
                    # ========================================================
                    
                    if detailed_debug: debug_info(f"{indent}[{depth}]    Reconstruction error: {error:.4f} (Threshold: {error_threshold})")
                    
                    if error < error_threshold:
                        # --- Log success only if detailed ---
                        if detailed_debug: debug_success(f"{indent}[{depth}]    SUCCESS: Applying PAIR abstraction for {pair_sig}!")
                        
                        # ==========================================================
                        # --- START NORMALIZATION CHANGE (PAIR ENCODING) ---
                        # ==========================================================
                        
                        # 4. Encode the *normalized* data
                        encoding, _ = model(normalized_combined)
                        
                        # ========================================================
                        # --- END NORMALIZATION CHANGE (PAIR ENCODING) ---
                        # ========================================================
                        
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
        
        # ==========================================================
        # --- START OF FIX 2 (SINGLETON) ---
        # ==========================================================
        if params_tensor.shape[1] != model.input_dim:
        # ==========================================================
        # --- END OF FIX 2 (SINGLETON) ---
        # ==========================================================

            # --- Always log critical errors ---
            debug_error(f"{indent}[{depth}]    Shape mismatch for {name}: Model expects {model.input_dim}, got {params_tensor.shape[1]}")
        else:
            # ==========================================================
            # --- START NORMALIZATION CHANGE (SINGLETON) ---
            # ==========================================================
            
            # 1. Normalize the parameters
            normalized_params = (params_tensor - model.data_mean_) / model.data_std_
            
            # 2. Get reconstruction of normalized data
            _, reconstruction = model(normalized_params)
            
            # 3. Calculate error against normalized data
            error = torch.max(torch.abs(reconstruction - normalized_params)).item()
            
            # ========================================================
            # --- END NORMALIZATION CHANGE (SINGLETON) ---
            # ========================================================
            
            if detailed_debug: debug_info(f"{indent}[{depth}]    Reconstruction error: {error:.4f} (Threshold: {error_threshold})")
            
            if error < error_threshold:
                # --- Log success only if detailed ---
                if detailed_debug: debug_success(f"{indent}[{depth}]    SUCCESS: Applying SINGLETON abstraction for {name}!")
                
                # ==========================================================
                # --- START NORMALIZATION CHANGE (SINGLETON ENCODING) ---
                # ==========================================================
                
                # 4. Encode the *normalized* data
                encoding, _ = model(normalized_params)
                
                # ========================================================
                # --- END NORMALIZATION CHANGE (SINGLETON ENCODING) ---
                # ========================================================
                
                if detailed_debug: debug_info(f"{indent}[{depth}] Returning NEW Abstraction node: {name}")
                return Abstraction(name, encoding.squeeze().tolist(), model, children=child_nodes)
    else:
        if detailed_debug: debug_info(f"{indent}[{depth}]    No model found for {name}.")

    # --- EXIT (No Abstraction Applied) ---
    if detailed_debug: debug_info(f"{indent}[{depth}] No abstraction applied. Returning rebuilt {name}.")
    return current_node