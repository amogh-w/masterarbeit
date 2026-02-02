"""
abstraction_utils.py

This module provides the core logic for finding, training, and applying
hierarchical abstractions to Domain-Specific Language (DSL) trees. It
introduces the `Abstraction` node, which acts as a "subroutine" or
compressed representation of a common DSL pattern.

It supports two main methods for parameter compression:
1.  **Autoencoder (AE):** A neural network trained to find a low-dimensional
    latent representation of parameters.
2.  **PCA (Principal Component Analysis):** A linear dimensionality reduction
    technique.

Key components:
-   `Autoencoder` & `PCAModel`: PyTorch modules for data compression.
-   `Abstraction`: A DSL node-like class that holds compressed parameters.
-   `find_abstractions`: The main function to *train* models (AE or PCA) on
    collected DSL parameter data.
-   `integrate_abstractions`: The main function to *apply* trained models,
    traversing a DSL tree and replacing concrete patterns with
    `Abstraction` nodes.
-   `expand_l1_to_l0` / `expand_l2_to_l1`: Functions to reverse the
    abstraction, expanding compressed nodes back into concrete DSL trees.
-   `instantiate_pattern`: A factory function to rebuild concrete DSL
    nodes from their name and (reconstructed) parameters.
"""
from __future__ import annotations

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
"""torch.device: The computation device (CUDA or CPU) to use."""

AUTOENCODER_LAYER_1_SIZE = 32
"""int: Size of the first hidden layer in the Autoencoder."""

AUTOENCODER_LAYER_2_SIZE = 16
"""int: Size of the second hidden layer in the Autoencoder."""

EPOCHS = 50
"""int: Default number of training epochs for autoencoders."""

BATCH_SIZE = 64
"""int: Default batch size for training."""

LEARNING_RATE = 1e-3
"""float: Default learning rate for the AdamW optimizer."""

MIN_EXAMPLES_FOR_ABSTRACTION = 100
"""int: Minimum number of data points required to train an abstraction model."""

RETRAIN_ITERATIONS = 1
"""int: Number of iterative retraining passes to filter outliers."""

ERROR_THRESHOLD = 0.05
"""float: Max normalized reconstruction error to be considered 'well-explained'."""

# ==============================================================================
# --- CORE LOGIC ---
# ==============================================================================


def t(tensor):
    """Move a tensor to the configured global DEVICE.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Tensor moved to the configured DEVICE.
    """
    return tensor.to(DEVICE)


class Autoencoder(nn.Module):
    """Simple fully-connected autoencoder with configurable hidden dimension.

    This model learns to compress `input_dim` features into a `hidden_dim`
    latent space and then reconstruct them.

    Attributes
    ----------
    input_dim : int
        Dimension of the input features.
    hidden_dim : int
        Dimension of the latent space (compressed representation).
    encoder : nn.Sequential
        The encoder network.
    decoder : nn.Sequential
        The decoder network.
    data_mean_ : torch.Tensor
        Mean vector of the training data. Attached by `find_abstractions`.
    data_std_ : torch.Tensor
        Std dev vector of the training data. Attached by `find_abstractions`.
    """

    def __init__(self, input_dim, hidden_dim):
        """Initialize the encoder and decoder layers.

        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        hidden_dim : int
            Dimension of latent space.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        debug_info(
            f"Initializing Autoencoder: "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}"
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
        """Perform a forward pass through the encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (expected to be normalized).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            -   `latent`: The latent representation.
            -   `decoded`: The reconstructed output (still normalized).
        """
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded


class PCAModel(nn.Module):
    """A model that uses PCA for encoding/decoding.

    This class is designed as a drop-in replacement for the `Autoencoder`.
    It implements the same `encoder`, `decoder`, and `forward` methods
    using linear projections based on Principal Component Analysis.
    It expects *normalized* data for fitting and forward passes.

    Attributes
    ----------
    input_dim : int
        Dimension of the input features.
    hidden_dim : int
        Dimension of the latent space (number of principal components).
    components : torch.Tensor
        The principal components (eigenvectors), registered as a buffer.
    data_mean_ : torch.Tensor
        Mean vector of the training data. Attached by `find_abstractions`.
    data_std_ : torch.Tensor
        Std dev vector of the training data. Attached by `find_abstractions`.
    """
    def __init__(self, input_dim, hidden_dim):
        """Initialize the PCA model structure.

        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        hidden_dim : int
            Dimension of latent space (K principal components).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Register 'components' as a buffer. It's part of the model's
        # state and moves with .to(DEVICE), but is not a trainable parameter.
        self.register_buffer('components', torch.randn(input_dim, hidden_dim))
        
        # These will be attached by the find_abstractions function
        self.data_mean_ = None
        self.data_std_ = None

    def fit(self, normalized_data_tensor):
        """'Train' the PCA model by calculating the principal components.

        This method computes the covariance matrix of the normalized data
        and stores the top `hidden_dim` eigenvectors as the components.

        Parameters
        ----------
        normalized_data_tensor : torch.Tensor
            The *normalized* training data.
        """
        if normalized_data_tensor.shape[0] < self.input_dim:
            debug_error(
                f"PCA fit error: Not enough samples "
                f"({normalized_data_tensor.shape[0]}) to compute covariance "
                f"for {self.input_dim} features."
            )
            # Keep random components as a fallback
            return
            
        # 1. Calculate covariance matrix
        cov_matrix = torch.cov(normalized_data_tensor.T)
        
        # 2. Get eigenvalues and eigenvectors
        try:
            _, eigenvectors = torch.linalg.eigh(cov_matrix)
        except Exception as e:
            debug_error(f"PCA linalg.eigh failed: {e}")
            return  # Keep random components

        # 3. Store the top 'hidden_dim' eigenvectors
        # Eigenvectors are sorted by eigenvalue (ascending), so take the last ones.
        self.components = eigenvectors[:, -self.hidden_dim:]
        debug_success(f"PCA fit complete. Components shape: {self.components.shape}")
        
    def encoder(self, normalized_x):
        """Project normalized data onto the principal components.

        Parameters
        ----------
        normalized_x : torch.Tensor
            Normalized input data (N, D).

        Returns
        -------
        torch.Tensor
            Latent representation (N, K).
        """
        # (N, D) @ (D, K) -> (N, K)
        return normalized_x @ self.components

    def decoder(self, z):
        """Reconstruct normalized data from the latent representation.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation (N, K).

        Returns
        -------
        torch.Tensor
            Reconstructed normalized data (N, D).
        """
        # (N, K) @ (K, D) -> (N, D)  (where (K, D) is components.T)
        return z @ self.components.T

    def forward(self, normalized_x):
        """Perform a forward pass: encode -> decode.

        Parameters
        ----------
        normalized_x : torch.Tensor
            Normalized input data (N, D).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            -   `latent`: The latent representation (N, K).
            -   `decoded`: The reconstructed normalized data (N, D).
        """
        latent = self.encoder(normalized_x)
        decoded = self.decoder(latent)
        return latent, decoded

# class VariationalAutoencoder(nn.Module):
#     """
#     Variational Autoencoder (VAE) for probabilistic compression.
    
#     Instead of mapping to a single point, it maps to a probability distribution 
#     (mean and variance) in the latent space.
#     """
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         # Shared encoder parts
#         self.encoder_shared = nn.Sequential(
#             nn.Linear(input_dim, AUTOENCODER_LAYER_1_SIZE),
#             nn.ReLU(),
#             nn.Linear(AUTOENCODER_LAYER_1_SIZE, AUTOENCODER_LAYER_2_SIZE),
#             nn.ReLU()
#         )
        
#         # Split into two heads: Mean (mu) and Log-Variance (logvar)
#         self.fc_mu = nn.Linear(AUTOENCODER_LAYER_2_SIZE, hidden_dim)
#         self.fc_logvar = nn.Linear(AUTOENCODER_LAYER_2_SIZE, hidden_dim)

#         # Decoder (Standard)
#         self.decoder_net = nn.Sequential(
#             nn.Linear(hidden_dim, AUTOENCODER_LAYER_2_SIZE),
#             nn.ReLU(),
#             nn.Linear(AUTOENCODER_LAYER_2_SIZE, AUTOENCODER_LAYER_1_SIZE),
#             nn.ReLU(),
#             nn.Linear(AUTOENCODER_LAYER_1_SIZE, input_dim),
#         )

#         # Placeholders for normalization stats (same as AE)
#         self.data_mean_ = None
#         self.data_std_ = None

#     def reparameterize(self, mu, logvar):
#         """
#         The 'Reparameterization Trick':
#         Sample z = mu + std * epsilon
#         """
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return mu + eps * std
#         else:
#             # During inference (integration), just return the mean
#             return mu

#     def encoder(self, x):
#         """Returns the mean (mu) latent vector. Used for deterministic compression."""
#         shared = self.encoder_shared(x)
#         return self.fc_mu(shared)

#     def decoder(self, z):
#         """Standard decoding from latent space."""
#         return self.decoder_net(z)

#     def forward(self, x):
#         """
#         Returns:
#             recon_x: Reconstructed input
#             mu: Mean of latent distribution
#             logvar: Log variance of latent distribution
#         """
#         shared = self.encoder_shared(x)
#         mu = self.fc_mu(shared)
#         logvar = self.fc_logvar(shared)
        
#         z = self.reparameterize(mu, logvar)
#         recon_x = self.decoder_net(z)
        
#         return recon_x, mu, logvar

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=2):
        super(VariationalAutoencoder, self).__init__()
        
        # Store input_dim so helper functions can check it
        self.input_dim = input_dim 
        
        self.register_buffer('data_mean_', torch.zeros(input_dim))
        self.register_buffer('data_std_', torch.ones(input_dim))
        
        # --- ENCODER (Lipschitz/Spectral Norm) ---
        self.encoder_shared = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_mu = spectral_norm(nn.Linear(hidden_dim, latent_dim))
        self.fc_logvar = spectral_norm(nn.Linear(hidden_dim, latent_dim))
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim) 
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 1. Encode
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 2. Sample or Inference
        if self.training:
            z = self.reparameterize(mu, logvar)
            recon_x = self.decoder(z)
            return recon_x, mu, logvar, h  # Return encoder features for Lipschitz loss
        else:
            # Inference: return just reconstruction and latent mean
            recon_x = self.decoder(mu)
            return recon_x, mu
         
def vae_loss_fn(recon_x, x, mu, logvar, h, h2=None, x2=None, kl_weight=0.001, lipschitz_weight=0.1):
    # Standard VAE loss
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Lipschitz regularization term
    lipschitz_loss = torch.tensor(0.0, device=x.device)
    if h2 is not None and x2 is not None:
        # Calculate |f(x1) - f(x2)| / |x1 - x2|
        h_diff = torch.norm(h - h2, dim=1)
        x_diff = torch.norm(x - x2, dim=1)
        
        # Avoid division by zero
        x_diff = torch.clamp(x_diff, min=1e-8)
        
        # Lipschitz ratio: h_diff / x_diff
        lip_ratio = h_diff / x_diff
        
        # Your desired term: |1 - (|f(x1) - f(x2)|) / (|x1 - x2|)|
        lipschitz_loss = torch.abs(1 - lip_ratio).sum()
    
    return mse_loss + (kl_weight * kld_loss) + (lipschitz_weight * lipschitz_loss)

def instantiate_pattern(pattern_name, params, children):
    """Rebuild a concrete DSL node (or subtree) from its components.

    This factory function is the reverse of `serialize()`. It takes a
    pattern name (e.g., "Scale", "Translate(Rotate)", "Abs(Scale)"), a
    list of *un-normalized* parameters, and a list of child nodes,
    and constructs the corresponding concrete DSL node object.

    Parameters
    ----------
    pattern_name : str
        The name of the pattern to instantiate.
    params : list
        The full, un-normalized parameter list for the pattern.
    children : list
        A list of child nodes to be attached to the new node(s).

    Returns
    -------
    object
        A concrete DSL node (e.g., `Scale`, `Translate`, `Union`) or
        `Box(-1)` if instantiation fails.
    """
    debug_info(
        f"[INSTANTIATE] Pattern: '{pattern_name}' | "
        f"params={params[:10]}{'...' if len(params) > 10 else ''} | "
        f"#children={len(children)}"
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
        # Handle nested abstraction patterns
        "Translate(Abs(Rotate(Scale)))": 3,
        "Rotate(Abs(Scale))": 4,
        "Scale(Abs(Rotate))": 3,
    }

    # --- SINGLETON RECONSTRUCTION ---
    singleton_patterns = [
        "Scale", "Rotate", "Translate", "SymRef", "SymRot", "SymTrans"
    ]
    if pattern_name in singleton_patterns:
        debug_info(f"  Handling singleton pattern: {pattern_name}")
        NodeClass = globals()[pattern_name]
        child_node = children[0] if children else Box(-1)

        if NodeClass == SymRef:
            if len(params) >= 6:
                return SymRef(
                    child_node, plane_normal=params[:3], point_on_plane=params[3:6]
                )
            else:
                debug_error(f"SymRef needs 6 params, got {len(params)}")
                return Box(-1)
        elif NodeClass == SymRot:
            if len(params) >= 6:
                return SymRot(
                    child_node, axis=params[:3], center=params[3:6], n_fold=-1
                )
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
        
        # Extract parent and child names
        if '(' in pattern_name and ')' in pattern_name:
            first_paren = pattern_name.find('(')
            last_paren = pattern_name.rfind(')')
            
            parent_name = pattern_name[:first_paren]
            child_name = pattern_name[first_paren+1:last_paren]
        else:
            debug_error(f"Invalid pair pattern: {pattern_name}")
            return Box(-1)
        
        debug_info(f"  Parent: {parent_name} with params {p_params}")
        debug_info(f"  Child: {child_name} with params {c_params}")

        try:
            ParentClass = globals()[parent_name]
            
            # Handle child node instantiation
            if child_name.startswith("Abs("):
                # Child is an Abstraction, reconstruct it
                child_node = Abstraction(
                    child_name, c_params, model=None, children=children
                )
            else:
                # Child is a regular DSL class
                ChildClass = globals()[child_name]
                grandchild_node = children[0] if children else Box(-1)
                
                if ChildClass == Box:
                    child_node = Box(-1)
                elif ChildClass == Union:
                    child_node = Union(
                        children[0], children[1]
                    ) if len(children) >= 2 else Box(-1)
                elif ChildClass == SymRef:
                    child_node = SymRef(
                        grandchild_node,
                        plane_normal=c_params[:3],
                        point_on_plane=c_params[3:6]
                    ) if len(c_params) >= 6 else Box(-1)
                elif ChildClass == SymRot:
                    child_node = SymRot(
                        grandchild_node,
                        axis=c_params[:3],
                        center=c_params[3:6],
                        n_fold=-1
                    ) if len(c_params) >= 6 else Box(-1)
                elif ChildClass == SymTrans:
                    child_node = SymTrans(
                        grandchild_node, end_point=c_params[:3], n_fold=-1
                    ) if len(c_params) >= 3 else Box(-1)
                elif ChildClass == Rotate:
                    child_node = Rotate(
                        grandchild_node, c_params[:4]
                    ) if len(c_params) >= 4 else Box(-1)
                elif ChildClass == Translate:
                    child_node = Translate(
                        grandchild_node, c_params[:3]
                    ) if len(c_params) >= 3 else Box(-1)
                elif ChildClass == Scale:
                    child_node = Scale(
                        grandchild_node, c_params[:3]
                    ) if len(c_params) >= 3 else Box(-1)
                else:
                    child_node = ChildClass(grandchild_node, c_params)

            # Build parent node
            if ParentClass == Union:
                return Union(
                    child_node, children[1]
                ) if len(children) >= 2 else Box(-1)
            elif ParentClass == SymRef:
                return SymRef(
                    child_node,
                    plane_normal=p_params[:3],
                    point_on_plane=p_params[3:6]
                ) if len(p_params) >= 6 else Box(-1)
            elif ParentClass == SymRot:
                return SymRot(
                    child_node,
                    axis=p_params[:3],
                    center=p_params[3:6],
                    n_fold=-1
                ) if len(p_params) >= 6 else Box(-1)
            elif ParentClass == SymTrans:
                return SymTrans(
                    child_node, end_point=p_params[:3], n_fold=-1
                ) if len(p_params) >= 3 else Box(-1)
            elif ParentClass == Rotate:
                return Rotate(
                    child_node, p_params[:4]
                ) if len(p_params) >= 4 else Box(-1)
            elif ParentClass == Translate:
                return Translate(
                    child_node, p_params[:3]
                ) if len(p_params) >= 3 else Box(-1)
            elif ParentClass == Scale:
                return Scale(
                    child_node, p_params[:3]
                ) if len(p_params) >= 3 else Box(-1)
            else:
                return ParentClass(child_node, p_params)
                
        except KeyError as e:
            debug_error(f"Unknown class in pattern {pattern_name}: {e}")
            return Box(-1)

    debug_error(f"[UNKNOWN PATTERN] '{pattern_name}', defaulting to Box(-1)")
    return Box(-1)

def expand_l2_to_l1(
    l2_dsl_node,
    singleton_models_L1,
    pair_models_L1,
    singleton_models_L2,
    pair_models_L2
):
    """Expand a DSL tree from L2 abstractions down to L1 abstractions.
    
    This function traverses a tree and, upon finding an L2 `Abstraction`
    node, uses the corresponding L2 model to decode its parameters.
    These decoded parameters are then used to construct the underlying
    L1 structure (which may include new L1 `Abstraction` nodes).

    Parameters
    ----------
    l2_dsl_node : object
        The root of the DSL tree (potentially containing L2 Abstractions).
    singleton_models_L1 : dict
        Pre-loaded dictionary of trained L1 singleton models.
    pair_models_L1 : dict
        Pre-loaded dictionary of trained L1 pair models.
    singleton_models_L2 : dict
        Pre-loaded dictionary of trained L2 singleton models.
    pair_models_L2 : dict
        Pre-loaded dictionary of trained L2 pair models.

    Returns
    -------
    object
        The expanded DSL tree, now containing only L0 nodes and
        L1 `Abstraction` nodes.
    """
    
    def _expand_node(node):
        """Recursively expand a node using the appropriate models."""
        if not hasattr(node, 'serialize') and not isinstance(node, Abstraction):
            return node
            
        if isinstance(node, Abstraction):
            pattern_name = node.pattern_name
            
            # Case 1: This is already an L1 Abstraction
            if pattern_name in pair_models_L1 or \
               pattern_name in singleton_models_L1:
                debug_info(f"Node is already L1 abstraction: {pattern_name}.")
                expanded_children = [_expand_node(child) for child in node.children]
                l1_model = pair_models_L1.get(pattern_name) or \
                           singleton_models_L1.get(pattern_name)
                return Abstraction(
                    pattern_name,
                    node.compressed_params,
                    l1_model,
                    expanded_children
                )

            # Case 2: This is an L2 Abstraction. Find L2 model.
            model = pair_models_L2.get(pattern_name) or \
                    singleton_models_L2.get(pattern_name)
            
            if not model:
                debug_error(f"No L1 or L2 model found for: {pattern_name}")
                return _expand_node(node.children[0]) if node.children else Box(-1)

            # It's an L2 model, so decode it
            debug_info(f"Expanding L2 abstraction: {pattern_name}")
            model.eval()
            with torch.no_grad():
                params_tensor = t(
                    torch.tensor(node.compressed_params, dtype=torch.float32)
                ).unsqueeze(0)
                
                # 1. Decoder outputs *normalized* parameters
                normalized_reconstruction = model.decoder(params_tensor)
                
                # 2. Un-normalize
                reconstructed_params_tensor = (
                    (normalized_reconstruction * model.data_std_) + model.data_mean_
                )
                reconstructed_params = reconstructed_params_tensor.squeeze().tolist()
            
            # Recursively expand children first
            expanded_children = [_expand_node(child) for child in node.children]
            
            # Manually instantiate the L1 structure based on L2 pattern name
            
            # --- L2 PAIR PATTERNS ---
            if pattern_name == "Translate(Abs(Rotate(Scale)))":
                translate_params = reconstructed_params[:3]
                l1_compressed_params = reconstructed_params[3:]
                l1_model = pair_models_L1.get("Rotate(Scale)")
                
                if l1_model:
                    l1_abs = Abstraction(
                        "Rotate(Scale)",
                        l1_compressed_params,
                        l1_model,
                        expanded_children
                    )
                    return Translate(l1_abs, translate_params)
                else:
                    debug_error("Missing L1 model for Rotate(Scale)")
                    return Box(-1)
                    
            elif pattern_name == "Translate(Abs(SymRef))":
                translate_params = reconstructed_params[:3]
                l1_compressed_params = reconstructed_params[3:]
                l1_model = singleton_models_L1.get("SymRef")
                
                if l1_model:
                    l1_abs = Abstraction(
                        "SymRef",
                        l1_compressed_params,
                        l1_model,
                        expanded_children
                    )
                    return Translate(l1_abs, translate_params)
                else:
                    debug_error("Missing L1 model for SymRef")
                    return Box(-1)

            # --- L2 SINGLETON PATTERNS ---
            elif pattern_name == "Abs(Rotate(Scale))":
                l1_compressed_params = reconstructed_params
                l1_model = pair_models_L1.get("Rotate(Scale)")
                
                if l1_model:
                    return Abstraction(
                        "Rotate(Scale)",
                        l1_compressed_params,
                        l1_model,
                        expanded_children
                    )
                else:
                    debug_error("Missing L1 model for Rotate(Scale)")
                    return Box(-1)
                        
            elif pattern_name == "Abs(SymRef)":
                l1_compressed_params = reconstructed_params
                l1_model = singleton_models_L1.get("SymRef")
                
                if l1_model:
                    return Abstraction(
                        "SymRef",
                        l1_compressed_params,
                        l1_model,
                        expanded_children
                    )
                else:
                    debug_error("Missing L1 model for SymRef")
                    return Box(-1)

            else:
                debug_info(f"Unhandled L2 pattern: {pattern_name}. Falling back.")
                return expanded_children[0] if expanded_children else Box(-1)

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
                return SymRef(
                    _expand_node(node.child), node.plane, node.point_on_plane
                )
            elif isinstance(node, SymRot):
                return SymRot(
                    _expand_node(node.child), node.axis, node.center, node.n
                )
            elif isinstance(node, SymTrans):
                return SymTrans(
                    _expand_node(node.child), node.end_point, node.n
                )
            else:
                return node
    
    debug_info("Starting L2 to L1 expansion with pre-loaded models...")
    result = _expand_node(l2_dsl_node)
    debug_success("L2 to L1 expansion completed")
    return result

def expand_l1_to_l0(l1_dsl_node, singleton_models_L1, pair_models_L1):
    """Expand a DSL tree from L1 abstractions down to L0 (concrete) nodes.
    
    This function traverses a tree and, upon finding an L1 `Abstraction`
    node, uses the corresponding L1 model to decode its parameters.
    It then uses `instantiate_pattern` to build the concrete L0
    DSL node (or subtree) from the reconstructed parameters.

    Parameters
    ----------
    l1_dsl_node : object
        The root of the DSL tree (potentially containing L1 Abstractions).
    singleton_models_L1 : dict
        Pre-loaded dictionary of trained L1 singleton models.
    pair_models_L1 : dict
        Pre-loaded dictionary of trained L1 pair models.

    Returns
    -------
    object
        The fully expanded, concrete (L0) DSL tree.
    """
    
    def _expand_node(node):
        if not hasattr(node, 'serialize') and not isinstance(node, Abstraction):
            return node
            
        if isinstance(node, Abstraction):
            debug_info(f"Expanding L1 abstraction: {node.pattern_name}")
            
            # Get the appropriate L1 model
            model = pair_models_L1.get(node.pattern_name) or \
                    singleton_models_L1.get(node.pattern_name)
            
            if not model:
                debug_error(f"No L1 model found for: {node.pattern_name}")
                return _expand_node(node.children[0]) if node.children else Box(-1)
            
            # Reconstruct parameters using L1 model
            model.eval()
            with torch.no_grad():
                if not node.compressed_params:
                    reconstructed_params = []
                else:
                    params_tensor = t(
                        torch.tensor(node.compressed_params, dtype=torch.float32)
                    ).unsqueeze(0)
                    
                    # 1. Decoder outputs *normalized* parameters
                    normalized_reconstruction = model.decoder(params_tensor)
                    
                    # 2. Un-normalize
                    reconstructed_params_tensor = (
                        (normalized_reconstruction * model.data_std_) +
                        model.data_mean_
                    )
                    reconstructed_params = (
                        reconstructed_params_tensor.squeeze().tolist()
                    )

            debug_info(f"Reconstructed params for {node.pattern_name}")
            
            # Expand children
            expanded_children = [_expand_node(child) for child in node.children]
            
            # Use the robust instantiate_pattern function to build the L0 nodes
            try:
                concrete_node = instantiate_pattern(
                    node.pattern_name, reconstructed_params, expanded_children
                )
                debug_success(
                    f"Successfully instantiated L0 node for {node.pattern_name}"
                )
                return concrete_node
            except Exception as e:
                debug_error(
                    f"instantiate_pattern FAILED for {node.pattern_name}: {e}"
                )
                return expanded_children[0] if expanded_children else Box(-1)
        
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
                return SymRef(
                    _expand_node(node.child), node.plane, node.point_on_plane
                )
            elif isinstance(node, SymRot):
                return SymRot(
                    _expand_node(node.child), node.axis, node.center, node.n
                )
            elif isinstance(node, SymTrans):
                return SymTrans(
                    _expand_node(node.child), node.end_point, node.n
                )
            else:
                return node
    
    debug_info("Starting L1 to L0 expansion...")
    result = _expand_node(l1_dsl_node)
    debug_success("L1 to L0 expansion completed")
    return result

class Abstraction:
    def __init__(self, pattern_name, compressed_params, model, children=None):
        self.pattern_name = pattern_name
        self.compressed_params = compressed_params
        self.model = model
        self.children = children if children is not None else []

    def __str__(self):
        header = f"Abs({self.pattern_name}, dim={len(self.compressed_params)})"
        return f"{header}"

    __repr__ = __str__

    def expand(self):
        if not self.compressed_params: return Box(-1).expand()
        self.model.eval()
        with torch.no_grad():
            params_tensor = t(torch.tensor(self.compressed_params, dtype=torch.float32)).unsqueeze(0)
            output = self.model.decoder(params_tensor)
            # FIX: VAE might return tuple? Safely check
            normalized_reconstruction = output[0] if isinstance(output, tuple) else output
            
            reconstructed_params_tensor = ((normalized_reconstruction * self.model.data_std_) + self.model.data_mean_)
            reconstructed_params = reconstructed_params_tensor.squeeze().tolist()
        return instantiate_pattern(self.pattern_name, reconstructed_params, self.children).expand()


def prepare_autoencoder_train_data(
    parameters,
    mask,
    data_mean,
    data_std,
    batch_size=BATCH_SIZE
):
    """Create a DataLoader, normalizing data using provided stats.

    Parameters
    ----------
    parameters : list
        A list of *all* parameter vectors (un-normalized).
    mask : torch.Tensor
        Boolean mask selecting which examples to include in the DataLoader.
    data_mean : torch.Tensor
        Mean vector (pre-calculated) for normalization.
    data_std : torch.Tensor
        Standard deviation vector (pre-calculated) for normalization.
    batch_size : int, optional
        Batch size for the DataLoader.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader yielding batches of normalized data.
    
    Raises
    ------
    ValueError
        If the mask size does not match the number of parameters.
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
    
    dataset = TensorDataset(normalized_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def is_well_explained(model, parameters_tensor, error_threshold=ERROR_THRESHOLD):
    model.eval()
    with torch.no_grad():
        normalized_input = (parameters_tensor - model.data_mean_) / model.data_std_
        output = model(normalized_input)
        
        # --- FIX: Handle VAE tuple output ---
        if isinstance(output, tuple):
            if len(output) == 3: # Training mode signature (recon, mu, logvar)
                 reconstructions = output[0]
            else: # Inference mode signature (recon, mu)
                 reconstructions = output[0]
        else:
            # AE/PCA case: output is (latent, reconstructions) or just reconstructions
            reconstructions = output[1]

        error, _ = torch.max(torch.abs(reconstructions - normalized_input), dim=-1)
    
    well_explained = error < error_threshold
    return well_explained, error


def make_safe_filename(name: str, suffix: str = "") -> str:
    """Create a safe, sanitized filename from a string.
    
    e.g., ("Translate(Rotate)", "pth") -> "translate_rotate.pth"

    Parameters
    ----------
    name : str
        The input string (e.g., a pattern name).
    suffix : str, optional
        A suffix to append, which will be treated as a file extension
        (a '.' will be added).

    Returns
    -------
    str
        The sanitized, lower-case filename.
    """
    safe = re.sub(r'[^\w\-]+', '_', name)
    safe = re.sub(r'_+', '_', safe).strip('_').lower()
    if suffix:
        # Add a dot before the suffix
        return f"{safe}.{suffix}"
    return safe


def train_autoencoder(
    model, 
    dataloader, 
    model_name, 
    epochs=EPOCHS, 
    lr=LEARNING_RATE,
    save_dir=None  # <-- NEW PARAMETER
):
    """Train an autoencoder, showing progress and plotting loss.

    Parameters
    ----------
    model : Autoencoder
        The autoencoder model to train.
    dataloader : DataLoader
        A DataLoader yielding batches of *normalized* training data.
    model_name : str
        The name of the model (e.g., "Translate(Rotate)") for
        logging and saving the loss chart.
    epochs : int, optional
        Number of epochs to train.
    lr : float, optional
        Learning rate for the optimizer.
    save_dir : str or Path, optional
        The directory to save the loss chart PNG. If None,
        the chart is only displayed.
    
    Returns
    -------
    Autoencoder
        The trained autoencoder model.
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
            num_samples = 0
            for batch in dataloader:
                x = batch[0].to(DEVICE)
                optimizer.zero_grad()
                _, x_rec = model(x)
                loss = loss_fn(x_rec, x)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item() * x.size(0)
                epoch_loss += batch_loss
                num_samples += x.size(0)
                global_batch += 1

                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs}",
                    "batch_loss": f"{loss.item():.6f}",
                    "avg_epoch_loss": f"{epoch_loss / num_samples:.6f}"
                })
                pbar.update(1)

            avg_epoch_loss = epoch_loss / len(dataloader.dataset)
            epoch_losses.append(avg_epoch_loss)

    # Plot epoch losses
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(
        range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', label='Training Loss'
    )
    ax.set_title(f"Training Loss for Model: {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average MSE Loss")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    if epochs <= 25:
        ax.set_xticks(range(1, epochs + 1))
    fig.tight_layout()

    # --- MODIFIED SAVE LOGIC ---
    # Save loss chart if a directory is provided
    if save_dir:
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_filename = make_safe_filename(model_name, suffix="loss_chart") + ".png"
            fig.savefig(save_path / safe_filename)
            debug_success(f"Saved loss chart to {save_path / safe_filename}")
        except Exception as e:
            debug_error(f"Failed to save loss chart for {model_name} to {save_dir}: {e}")
    # --- END MODIFIED LOGIC ---

    plt.show()
    plt.close(fig)

    return model

def train_vae(model, dataloader, model_name, epochs=EPOCHS, lr=LEARNING_RATE, save_dir=None, 
              lipschitz_weight=0.1, gradient_penalty=False):
    """
    Train a Variational Autoencoder with tqdm progress tracking and Lipschitz regularization.
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    epoch_losses = []
    
    model.train()
    
    total_batches = epochs * len(dataloader)
    
    with tqdm(total=total_batches, desc=f"Training VAE {model_name}", unit="batch") as pbar:
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_samples = 0
            
            for batch_idx, batch in enumerate(dataloader):
                x = batch[0].to(DEVICE)
                batch_size = x.size(0)
                
                optimizer.zero_grad()
                
                # Forward pass - returns 4 values now (recon_x, mu, logvar, h)
                recon_x, mu, logvar, h = model(x)
                
                # Compute standard VAE loss first
                mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = mse_loss + (0.01 * kld_loss)
                
                # Add Lipschitz regularization if enabled and batch is large enough
                if batch_size >= 2 and lipschitz_weight > 0:
                    # Create pairs with SAME size for both groups
                    pair_size = min(batch_size, 32)  # Use smaller size to ensure matching
                    idx1 = torch.randperm(batch_size)[:pair_size]
                    idx2 = torch.randperm(batch_size)[:pair_size]
                    
                    x1 = x[idx1]
                    x2 = x[idx2]
                    h1 = h[idx1]
                    h2 = h[idx2]
                    
                    # Calculate Lipschitz loss
                    h_diff = torch.norm(h1 - h2, dim=1)
                    x_diff = torch.norm(x1 - x2, dim=1)
                    
                    # Avoid division by zero
                    x_diff = torch.clamp(x_diff, min=1e-8)
                    
                    # Lipschitz ratio: h_diff / x_diff
                    lip_ratio = h_diff / x_diff
                    
                    # Your desired term: |1 - (|f(x1) - f(x2)|) / (|x1 - x2|)|
                    lipschitz_loss = torch.abs(1 - lip_ratio).sum()
                    
                    # Add to total loss
                    loss = loss + (lipschitz_weight * lipschitz_loss)
                
                loss.backward()
                optimizer.step()
                
                # Track metrics
                batch_loss = loss.item() * batch_size
                epoch_loss += batch_loss
                num_samples += batch_size
                
                pbar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs}",
                    "batch_loss": f"{loss.item()/batch_size:.6f}",
                    "lip_weight": f"{lipschitz_weight:.4f}"
                })
                pbar.update(1)
            
            avg_epoch_loss = epoch_loss / num_samples if num_samples > 0 else 0
            epoch_losses.append(avg_epoch_loss)
    
    # Plot epoch losses
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', label='Training Loss')
    ax.set_title(f"Training Loss for VAE Model: {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss (ELBO + Lipschitz)")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if save_dir:
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_filename = make_safe_filename(model_name, suffix="loss_chart") + ".png"
            fig.savefig(save_path / safe_filename)
        except Exception as e:
            debug_error(f"Failed to save loss chart: {e}")
    
    plt.show()
    plt.close(fig)
    
    return model

def compute_gradient_penalty(model, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN-GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Get encoder output for interpolates
    h = model.encoder_shared(interpolates)
    
    # Calculate gradients of outputs with respect to inputs
    gradients = torch.autograd.grad(
        outputs=h,
        inputs=interpolates,
        grad_outputs=torch.ones_like(h),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

# ==============================================================================
# --- NEW MERGED ABSTRACTION FINDER ---
# ==============================================================================

def find_abstractions(
    structures,
    method='ae',
    structure_type="PATTERNS",
    min_examples=MIN_EXAMPLES_FOR_ABSTRACTION,
    retrain_iterations=RETRAIN_ITERATIONS,
    error_threshold=ERROR_THRESHOLD,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    save_dir=None,  # <-- NEW PARAMETER
    plot_error_distribution=False
):
    """Find abstractions for DSL patterns using either Autoencoder or PCA.

    This is the main *training* function. It iterates through all
    patterns ("structures") and trains a compression model (AE or PCA)
    for each one that has enough examples.

    It handles:
    -   Calculating normalization stats (mean/std) for each pattern.
    -   Iterative retraining to filter out outliers.
    -   Attaching the `data_mean_` and `data_std_` to the final
        trained model for later use in normalization/un-normalization.

    Parameters
    ----------
    structures : dict
        A dictionary mapping pattern names (str) to lists of
        parameter vectors (list[list[float]]).
    method : str, optional
        The abstraction method: 'ae' (Autoencoder) or 'pca' (PCA).
        Defaults to 'ae'.
    structure_type : str, optional
        A descriptive name for the type of structures being processed
        (e.g., "L1 PATTERNS"), used for logging.
    min_examples : int, optional
        Minimum examples needed to train a model.
    retrain_iterations : int, optional
        Number of iterative retraining passes.
    error_threshold : float, optional
        Reconstruction error threshold for filtering outliers.
    epochs : int, optional
        Number of training epochs (used by 'ae' only).
    lr : float, optional
        Learning rate (used by 'ae' only).

    Returns
    -------
    dict
        A dictionary mapping pattern names (str) to their successfully
        trained models (Autoencoder or PCAModel).
    """
    debug_info(f"Starting abstraction search for {len(structures)} {structure_type}...")
    trained_models = {}

    # Sort by frequency
    sorted_structures = sorted(
        structures.items(), key=lambda item: len(item[1]), reverse=True
    )
    debug_info("Structures sorted by sample count (descending).")

    method_name = method.lower()
    if method_name not in ['ae', 'pca', 'vae']:
        debug_error(f"Unknown method '{method}'. Defaulting to 'ae'.")
        method_name = 'ae'
    else:
        debug_info(f"Using abstraction method: {method_name.upper()}")

    # ----------------------------------------------------------------------
    # MAIN LOOP: per-structure training
    # ----------------------------------------------------------------------
    for name, parameters in sorted_structures:
        debug_info(f"\n--- Processing pattern '{name}' ({len(parameters)} samples) ---")

        # Skip small patterns
        if len(parameters) < min_examples:
            debug_info(f"Skipping '{name}': only {len(parameters)} examples (< {min_examples}).")
            continue

        num_params = len(parameters[0])
        if num_params <= 1:
            debug_info(f"Skipping '{name}': only 1 parameter dimension.")
            continue

        debug_info(f"Parameter dimension for '{name}': {num_params}")

        # Tensor preparation
        params_tensor = t(torch.tensor(parameters, dtype=torch.float32))
        debug_info(f"Loaded tensor for '{name}' with shape {params_tensor.shape}")

        mask = torch.ones(len(parameters), dtype=torch.bool, device=DEVICE)
        model = None

        # ------------------------------------------------------------------
        # ITERATIVE RE-TRAINING LOOP
        # ------------------------------------------------------------------
        for iteration in range(retrain_iterations):

            debug_info(f"-- Iteration {iteration+1} for '{name}' --")
            debug_info(f"Current mask keeps {mask.sum().item()} / {len(mask)} samples.")

            current_params = params_tensor[mask]

            if not current_params.any() or len(current_params) < 2:
                debug_info(f"No more valid data for '{name}' in iteration {iteration+1}. Breaking.")
                break

            # ---- Compute normalization stats ----
            data_mean = torch.mean(current_params, dim=0)
            data_std = torch.std(current_params, dim=0)
            data_std[data_std == 0] = 1.0

            debug_info(f"{name} Mean: {data_mean.tolist()}")
            debug_info(f"{name} Std: {data_std.tolist()}")

            # ---- Hidden dimension decision ----
            hidden_dim = max(1, num_params - 1)
            debug_info(f"Hidden dimension for '{name}': {hidden_dim}")

            # ------------------------------------------------------------------
            # PCA METHOD
            # ------------------------------------------------------------------
            if method_name == 'pca':
                debug_info(f"Initializing PCA model for '{name}'...")

                normalized_current_params = (current_params - data_mean) / data_std

                model = PCAModel(num_params, hidden_dim).to(DEVICE)
                model.data_mean_ = data_mean.to(DEVICE)
                model.data_std_ = data_std.to(DEVICE)

                debug_info(
                    f"Fitting PCA for '{name}' (iteration {iteration+1}) "
                    f"on {len(normalized_current_params)} normalized samples..."
                )

                model.fit(normalized_current_params)

                debug_info(f"PCA training complete for '{name}'.")

            # ------------------------------------------------------------------
            # AUTOENCODER METHOD
            # ------------------------------------------------------------------
            elif method_name == 'ae':
                debug_info(f"Preparing Autoencoder training data for '{name}'...")

                dataloader = prepare_autoencoder_train_data(
                    parameters, mask, data_mean, data_std
                )
                
                if len(dataloader.dataset) == 0:
                    debug_info(f"Dataloader empty for '{name}' in iteration {iteration+1}. Breaking.")
                    break

                debug_info(f"Training AE for '{name}' with {len(dataloader.dataset)} samples.")

                model = Autoencoder(num_params, hidden_dim).to(DEVICE)
                model.data_mean_ = data_mean.to(DEVICE)
                model.data_std_ = data_std.to(DEVICE)

                model = train_autoencoder(
                    model, dataloader, model_name=name, epochs=epochs, lr=lr, save_dir=save_dir
                )
                debug_info(f"AE training complete for '{name}'.")

            # ------------------------------------------------------------------
            # VARIATIONAL AUTOENCODER METHOD
            # ------------------------------------------------------------------
            # elif method_name == 'vae':
            #     # --- FIX: Define dataloader for VAE block correctly ---
            #     dataloader = prepare_autoencoder_train_data(parameters, mask, data_mean, data_std)
            #     if len(dataloader.dataset) == 0: break
                
            #     model = VariationalAutoencoder(num_params, hidden_dim).to(DEVICE)
            #     model.data_mean_ = data_mean.to(DEVICE)
            #     model.data_std_ = data_std.to(DEVICE)
            #     model = train_vae(model, dataloader, model_name=name, epochs=epochs, lr=lr, save_dir=save_dir)
            elif method_name == 'vae':
                dataloader = prepare_autoencoder_train_data(parameters, mask, data_mean, data_std)
                if len(dataloader.dataset) == 0: break
                
                model = VariationalAutoencoder(num_params, hidden_dim).to(DEVICE)
                model.data_mean_ = data_mean.to(DEVICE)
                model.data_std_ = data_std.to(DEVICE)
                
                # Add lipschitz_weight parameter
                model = train_vae(model, dataloader, model_name=name, epochs=epochs, 
                                lr=lr, save_dir=save_dir, lipschitz_weight=0.1)


            # ------------------------------------------------------------------
            # RE-EVALUATE MASK (OUTLIER FILTERING)
            # ------------------------------------------------------------------
            debug_info(f"Recomputing mask for '{name}' using error threshold {error_threshold}...")
            mask, errors = is_well_explained(model, params_tensor, error_threshold)

            debug_info(
                f"[{name} Iter {iteration+1}] Kept "
                f"{mask.sum().item()}/{len(parameters)} examples."
            )

            # --- NEW PLOTTING LOGIC ---
            if plot_error_distribution:
                errors_np = errors.cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
                # Plot histogram of all errors
                ax.hist(
                    errors_np, 
                    bins=50, 
                    alpha=0.7, 
                    label='All Sample Errors'
                )
                
                # Plot histogram of errors *below* the threshold (the ones we keep)
                # ax.hist(
                #     errors_np[mask.cpu().numpy()], 
                #     bins=25, 
                #     alpha=0.9, 
                #     label=f'Kept (Error < {error_threshold})'
                # )
                
                # Draw the threshold line
                ax.axvline(
                    error_threshold, 
                    color='red', 
                    linestyle='--', 
                    linewidth=2, 
                    label=f'Error Threshold ({error_threshold})'
                )
                
                ax.set_title(
                    f'"{name}" - Error Distribution (Iter {iteration+1})'
                )
                ax.set_xlabel("Max Reconstruction Error (Normalized Space)")
                ax.set_ylabel("Sample Count")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                fig.tight_layout()

                # --- NEW SAVE LOGIC ---
                if save_dir:
                    try:
                        save_path = Path(save_dir)
                        save_path.mkdir(parents=True, exist_ok=True)
                        # Include iteration in filename
                        safe_filename = make_safe_filename(
                            name, suffix=f"error_iter_{iteration+1}"
                        ) + ".png"
                        fig.savefig(save_path / safe_filename)
                        debug_success(f"Saved error plot to {save_path / safe_filename}")
                    except Exception as e:
                        debug_error(f"Failed to save error plot for {name}: {e}")
                # --- END NEW SAVE LOGIC ---

                plt.show()
                plt.close(fig)
                
            # --- END NEW PLOTTING LOGIC ---

        # ----------------------------------------------------------------------
        # STORE FINAL MODEL
        # ----------------------------------------------------------------------
        if model is None:
            debug_error(f"No model produced for '{name}'. Skipping.")
            continue

        if not mask.any():
            debug_error(f"All samples rejected for '{name}'. Not storing model.")
            continue

        trained_models[name] = model
        debug_success(f"Stored final model for '{name}'.")

    debug_success(f"Finished abstraction search. Created {len(trained_models)} models.")
    return trained_models


# ==============================================================================
# --- FIXED INTEGRATION FUNCTION ---
# ==============================================================================

def debug_abstraction(msg):
    debug_success(f"[ABSTRACT] {msg}")


# ==============================================================================
# --- INTEGRATION FUNCTION ---
# ==============================================================================

def debug_abstraction(msg):
    debug_success(f"[ABSTRACT] {msg}")

# ==============================================================================
# --- EGGLOG CONFIGURATION ---
# ==============================================================================

import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import textwrap
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

# Added for Egglog Integration
from egglog import (
    Expr, f64Like, StringLike, Vec, f64, String, 
    rewrite, ruleset, method, function, EGraph
)

from abstractionssymh.dsl_nodes import (
    Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans,
)
from abstractionssymh.debug_utils import debug_info, debug_error, debug_success

# --- Updated Schema with Adjusted Costs ---
class Shape(Expr):
    @method(cost=100) # Unions are now very expensive
    def __add__(self, other: Shape) -> Shape: ...

@function(cost=100) # High cost makes raw boxes undesirable
def box(label: f64Like) -> Shape: ...

@function(cost=0) # Abstractions are "Free" to ensure they are kept
def abs_node(name: StringLike, params: Vec[f64], children: Vec[Shape]) -> Shape: ...

@function(cost=10) # Lower than Union, but high enough to prune no-ops
def translate(s: Shape, x: f64Like, y: f64Like, z: f64Like) -> Shape: ...

@function(cost=10)
def rotate(s: Shape, x: f64Like, y: f64Like, z: f64Like, w: f64Like) -> Shape: ...

@function(cost=10)
def scale(s: Shape, sx: f64Like, sy: f64Like, sz: f64Like) -> Shape: ...

@function(cost=1) # Symmetries are cheap to encourage folding
def sym_ref(s: Shape, nx: f64Like, ny: f64Like, nz: f64Like, px: f64Like, py: f64Like, pz: f64Like) -> Shape: ...

@function(cost=1)
def sym_rot(s: Shape, ax: f64Like, ay: f64Like, az: f64Like, cx: f64Like, cy: f64Like, cz: f64Like, n: f64Like) -> Shape: ...

@function(cost=1)
def sym_trans(s: Shape, dx: f64Like, dy: f64Like, dz: f64Like, n: f64Like) -> Shape: ...

# --- Conversion Helpers ---

def to_egglog(node) -> Shape:
    """Lifts Python DSL/Abstraction objects into Egglog Shape."""
    from abstractionssymh.dsl_nodes import Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans
    from abstractionssymh.abstraction_utils import Abstraction
    
    if isinstance(node, Abstraction):
        # FIX: Use capital Vec constructor
        p_vec = Vec(*[f64(float(p)) for p in node.compressed_params])
        c_vec = Vec(*[to_egglog(c) for c in node.children])
        return abs_node(node.pattern_name, p_vec, c_vec)
    
    elif isinstance(node, Box):
        return box(f64(float(node.label)))
    
    elif isinstance(node, Scale):
        sx, sy, sz = map(lambda v: f64(float(v)), node.lengths)
        return scale(to_egglog(node.child), sx, sy, sz)
    
    elif isinstance(node, Rotate):
        x, y, z, w = map(lambda v: f64(float(v)), node.quaternion)
        return rotate(to_egglog(node.child), x, y, z, w)
    
    elif isinstance(node, Translate):
        x, y, z = map(lambda v: f64(float(v)), node.center)
        return translate(to_egglog(node.child), x, y, z)
    
    elif isinstance(node, Union):
        return to_egglog(node.left) + to_egglog(node.right)
    
    elif isinstance(node, SymRef):
        nx, ny, nz = map(lambda v: f64(float(v)), node.plane)
        px, py, pz = map(lambda v: f64(float(v)), node.point_on_plane)
        return sym_ref(to_egglog(node.child), nx, ny, nz, px, py, pz)
    
    elif isinstance(node, SymRot):
        ax, ay, az = map(lambda v: f64(float(v)), node.axis)
        cx, cy, cz = map(lambda v: f64(float(v)), node.center)
        return sym_rot(to_egglog(node.child), ax, ay, az, cx, cy, cz, f64(float(node.n)))
    
    elif isinstance(node, SymTrans):
        dx, dy, dz = map(lambda v: f64(float(v)), node.end_point)
        return sym_trans(to_egglog(node.child), dx, dy, dz, f64(float(node.n)))
    
    return box(f64(-1.0))

def from_egglog_string(egg_expr, singleton_models, pair_models):
    """Lowers Egglog string back to Python DSL/Abstraction objects."""
    from abstractionssymh.dsl_nodes import Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans
    from abstractionssymh.abstraction_utils import Abstraction
    import re

    # Patch Union operator
    def dsl_add(self, other): return Union(self, other)
    for cls in [Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans, Abstraction]:
        cls.__add__ = dsl_add

    namespace = {
        "box": lambda l: Box(label=int(l)),
        "scale": lambda c, x, y, z: Scale(c, [x, y, z]),
        "rotate": lambda c, x, y, z, w: Rotate(c, [x, y, z, w]),
        "translate": lambda c, x, y, z: Translate(c, [x, y, z]),
        "sym_ref": lambda c, nx, ny, nz, px, py, pz: SymRef(c, [nx, ny, nz], [px, py, pz]),
        "sym_rot": lambda c, ax, ay, az, cx, cy, cz, n: SymRot(c, [ax, ay, az], [cx, cy, cz], int(n)),
        "sym_trans": lambda c, dx, dy, dz, n: SymTrans(c, [dx, dy, dz], int(n)),
        "abs_node": lambda name, params, children: Abstraction(
            pattern_name=name, compressed_params=params,
            model=pair_models.get(name) or singleton_models.get(name),
            children=children
        ),
        "Vec": lambda *args: list(args), 
        "f64": lambda x: float(x),       
        "String": lambda x: str(x),      
    }

    try:
        expr_str = str(egg_expr).strip()
        
        # 1. Strip Type Subscripts (Vec[f64] -> Vec)
        expr_str = re.sub(r"Vec\[.*?\]", "Vec", expr_str)
        
        # 2. Extract assignments vs final expression
        lines = expr_str.splitlines()
        assignment_block = ""
        final_expression = ""
        
        for line in lines:
            if "=" in line and line.strip().startswith("_"):
                assignment_block += line + "\n"
            else:
                final_expression += line + " " # Keep multi-line expressions together
        
        # 3. Execute assignments to populate namespace
        if assignment_block:
            exec(assignment_block, {"__builtins__": {}}, namespace)
        
        # 4. Evaluate the final consolidated expression
        return eval(final_expression.strip(), {"__builtins__": {}}, namespace)
        
    except Exception as e:
        debug_error(f"Parser Error: {e} | String: {expr_str}")
        # return None

# --- Updated Ruleset ---
sem_rules = ruleset()

@sem_rules.register
def _rules_def(
    s: Shape, s2: Shape, s3: Shape, name: String, p: Vec[f64], c: Vec[Shape],
    x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64, sx: f64, sy: f64, sz: f64, w: f64
):
    # 1. Abstraction mirrored symmetry 
    # This is the "Identity" check for VAE nodes - folds two identical VAE blocks into one SymRef
    yield rewrite(
        translate(abs_node(name, p, c), x1, y1, z1) + 
        translate(abs_node(name, p, c), f64(-1.0) * x1, y1, z1)
    ).to(
        sym_ref(translate(abs_node(name, p, c), x1, y1, z1), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    
    # 2. L0 mirrored symmetry
    yield rewrite(translate(s, x1, y1, z1) + translate(s, f64(-1.0) * x1, y1, z1)).to(
        sym_ref(translate(s, x1, y1, z1), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    # 3. Factor common translations (Works for Abstractions too!)
    yield rewrite(translate(s, x1, y1, z1) + translate(s2, x1, y1, z1)).to(
        translate(s + s2, x1, y1, z1)
    )

    # 4. Combine nested translations (Flattening)
    yield rewrite(translate(translate(s, x1, y1, z1), x2, y2, z2)).to(
        translate(s, x1 + x2, y1 + y2, z1 + z2)
    )

    # 5. Remove no-ops
    yield rewrite(translate(s, 0.0, 0.0, 0.0)).to(s)
    yield rewrite(scale(s, 1.0, 1.0, 1.0)).to(s)
    yield rewrite(rotate(s, 0.0, 0.0, 0.0, 1.0)).to(s)

def integrate_abstractions(
    node,
    singleton_models,
    pair_models,
    error_threshold=ERROR_THRESHOLD,
    geometric_threshold=0.05,
    points_per_check=50,
    depth=0,
    detailed_debug=False
):
    from abstractionssymh.abstraction_compare_utils import get_point_cloud_from_dsl, calculate_chamfer_distance
    from abstractionssymh.dsl_nodes import Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans
    from abstractionssymh.abstraction_utils import Abstraction, t, Shape, to_egglog, from_egglog_string, sem_rules
    import torch
    from egglog import EGraph, Vec

    indent = "  " * depth
    
    # ------------------------------------------------------------------
    # 1. RECURSIVE STEP (Bottom-Up Traversal)
    # ------------------------------------------------------------------
    if isinstance(node, Abstraction):
        node.children = [integrate_abstractions(c, singleton_models, pair_models, 
                                               error_threshold, geometric_threshold, 
                                               points_per_check, depth+1, detailed_debug) 
                         for c in node.children]
        current_node = node
    elif not hasattr(node, "serialize"):
        return node
    else:
        try:
            _, (_, children) = node.serialize()
            rebuilt_children = []
            for c in children:
                if hasattr(c, "serialize") or isinstance(c, Abstraction):
                    rebuilt_children.append(integrate_abstractions(c, singleton_models, pair_models, 
                                                                   error_threshold, geometric_threshold, 
                                                                   points_per_check, depth+1, detailed_debug))
                else:
                    rebuilt_children.append(c)
            
            if isinstance(node, Union): current_node = Union(rebuilt_children[0], rebuilt_children[1])
            elif isinstance(node, SymRef): current_node = SymRef(rebuilt_children[0], plane_normal=node.plane, point_on_plane=node.point_on_plane)
            elif isinstance(node, SymRot): current_node = SymRot(rebuilt_children[0], axis=node.axis, center=node.center, n_fold=node.n)
            elif isinstance(node, SymTrans): current_node = SymTrans(rebuilt_children[0], end_point=node.end_point, n_fold=node.n)
            elif hasattr(node, "child"):
                kwargs = {k: v for k, v in node.__dict__.items() if k != "child"}
                current_node = type(node)(rebuilt_children[0], **kwargs)
            else:
                current_node = type(node)(*rebuilt_children)
        except Exception:
            return node

    # ------------------------------------------------------------------
    # 2. SYMBOLIC REFACTORING (Egglog Layer)
    # ------------------------------------------------------------------
    # try:
    print(f"{indent}[EGGLOG] Attempting refactor on subtree...")

    egraph = EGraph()
    egg_shape = egraph.let("subtree", to_egglog(current_node))
    egraph.run(sem_rules.saturate())
    extracted, cost = egraph.extract(egg_shape, include_cost=True)
    print("extracted", extracted)
    refined = from_egglog_string(extracted, singleton_models, pair_models)
    print("refined", refined)

    print(refined)
    
    if refined is not None and str(refined) != str(current_node):
        if detailed_debug: print(f"{indent}[EGGLOG] Improved subtree cost (Extracted Cost: {cost})")
        current_node = refined
    # except Exception as e:
    #     if detailed_debug: print(f"{indent}[WARN] Egglog refactor skipped: {e}")

    # ------------------------------------------------------------------
    # 3. VAE COMPRESSION LAYER (With Detailed Debug)
    # ------------------------------------------------------------------
    
    def check_geometric_fidelity(original_node, candidate_abstraction, label):
        try:
            pc_orig = get_point_cloud_from_dsl(original_node, points_per_box=points_per_check)
            pc_cand = get_point_cloud_from_dsl(candidate_abstraction, points_per_box=points_per_check)
            if len(pc_orig) == 0 or len(pc_cand) == 0: return False
            dist = calculate_chamfer_distance(pc_orig, pc_cand)
            
            if detailed_debug:
                status = "PASS" if dist <= geometric_threshold else "FAIL"
                print(f"{indent}  [GEO-CHECK] {label} | Chamfer: {dist:.6f} (Limit: {geometric_threshold}) -> {status}")
            
            return dist <= geometric_threshold
        except Exception as e:
            if detailed_debug: print(f"{indent}  [GEO-ERROR] {e}")
            return False

    child_nodes = []
    if isinstance(current_node, Abstraction):
        child_nodes = current_node.children
    elif hasattr(current_node, "serialize"):
        _, (_, raw_children) = current_node.serialize()
        child_nodes = [c for c in raw_children if hasattr(c, "serialize") or isinstance(c, Abstraction)]

    # --- VAE PAIR LOGIC ---
    if len(child_nodes) == 1:
        child = child_nodes[0]
        if isinstance(child, Abstraction):
            child_name, child_params, grandchildren = f"Abs({child.pattern_name})", child.compressed_params, child.children
        else:
            child_name = type(child).__name__
            child_params, gc_raw = child.serialize()[1]
            grandchildren = [gc for gc in gc_raw if hasattr(gc, "serialize") or isinstance(gc, Abstraction)]

        pair_sig = f"{type(current_node).__name__}({child_name})"
        if pair_sig in pair_models:
            model = pair_models[pair_sig]
            p_params, _ = current_node.serialize()[1]
            combined = t(torch.tensor(list(p_params or []) + list(child_params or []), dtype=torch.float32)).unsqueeze(0)
            
            if hasattr(model, 'input_dim') and combined.shape[1] == model.input_dim:
                normalized = (combined - model.data_mean_) / model.data_std_
                output = model(normalized)
                recon, encoding = (output[1], output[0]) if not isinstance(output, tuple) or len(output) != 2 else (output[0], output[1])
                
                err = torch.max(torch.abs(recon - normalized)).item()
                if detailed_debug:
                    print(f"{indent}[VAE-TRY] PAIR {pair_sig} | Param Error: {err:.6f} (Limit: {error_threshold})")
                
                if err < error_threshold:
                    candidate = Abstraction(pair_sig, encoding.squeeze().tolist(), model, children=grandchildren)
                    if check_geometric_fidelity(current_node, candidate, pair_sig):
                        if detailed_debug: print(f"{indent}[VAE-SUCCESS] Abstracted PAIR: {pair_sig}")
                        return candidate

    # --- VAE SINGLETON LOGIC ---
    name = type(current_node).__name__
    if name in singleton_models:
        model = singleton_models[name]
        params, _ = current_node.serialize()[1]
        if params:
            params_tensor = t(torch.tensor(params, dtype=torch.float32)).unsqueeze(0)
            if hasattr(model, 'input_dim') and params_tensor.shape[1] == model.input_dim:
                normalized = (params_tensor - model.data_mean_) / model.data_std_
                output = model(normalized)
                recon, encoding = (output[1], output[0]) if not isinstance(output, tuple) or len(output) != 2 else (output[0], output[1])

                err = torch.max(torch.abs(recon - normalized)).item()
                if detailed_debug:
                    print(f"{indent}[VAE-TRY] SINGLETON {name} | Param Error: {err:.6f} (Limit: {error_threshold})")

                if err < error_threshold:
                    candidate = Abstraction(name, encoding.squeeze().tolist(), model, children=child_nodes)
                    if check_geometric_fidelity(current_node, candidate, name):
                        if detailed_debug: print(f"{indent}[VAE-SUCCESS] Abstracted SINGLETON: {name}")
                        return candidate

    return current_node

# def integrate_abstractions(
#     node,
#     singleton_models,
#     pair_models,
#     error_threshold=ERROR_THRESHOLD,
#     geometric_threshold=0.05,  # <--- NEW: Geometric threshold (default 5cm if units=m)
#     points_per_check=50,       # <--- NEW: Low point count for speed
#     depth=0,
#     detailed_debug=False
# ):
#     """Recursively abstract a DSL tree using both Parameter and Geometric metrics.

#     1. Checks Parametric MSE (Fast).
#     2. If passed, checks Geometric Chamfer Distance (Slow but accurate).
#     """
#     # --- Local Imports to avoid circular dependencies ---
#     from abstractionssymh.abstraction_compare_utils import (
#         get_point_cloud_from_dsl, 
#         calculate_chamfer_distance
#     )
#     from abstractionssymh.abstraction_utils import Abstraction, t, debug_abstraction

#     indent = "  " * depth
#     node_name = f"Abs({node.pattern_name})" if isinstance(node, Abstraction) else type(node).__name__

#     if detailed_debug:
#         debug_info(f"{indent}[{depth}] Processing: {node_name}")

#     # ------------------------------------------------------------------
#     # ALREADY ABSTRACTED
#     # ------------------------------------------------------------------
#     if isinstance(node, Abstraction):
#         if detailed_debug:
#             debug_info(f"{indent}[{depth}] Already abstracted; recursing into children.")
#         node.children = [
#             integrate_abstractions(c, singleton_models, pair_models,
#                                    error_threshold, geometric_threshold, points_per_check, depth+1, detailed_debug)
#             for c in node.children
#         ]
#         return node

#     # ------------------------------------------------------------------
#     # NODE WITHOUT serialize()
#     # ------------------------------------------------------------------
#     if not hasattr(node, "serialize"):
#         if detailed_debug:
#             debug_error(f"{indent}[{depth}] Node {type(node).__name__} lacks serialize().")
#         return node

#     # ------------------------------------------------------------------
#     # RECURSE (Bottom-Up)
#     # ------------------------------------------------------------------
#     try:
#         _, (_, children) = node.serialize()
#     except Exception as e:
#         if detailed_debug:
#             debug_error(f"{indent}[{depth}] Failed to serialize {node_name}: {e}")
#         return node

#     valid_children = [
#         c for c in children if hasattr(c, "serialize") or isinstance(c, Abstraction)
#     ]
#     if detailed_debug:
#         debug_info(f"{indent}[{depth}] Recursing into {len(valid_children)} children.")

#     rebuilt_children = []
#     for c in children:
#         if hasattr(c, "serialize") or isinstance(c, Abstraction):
#             rebuilt_children.append(
#                 integrate_abstractions(c, singleton_models, pair_models,
#                                        error_threshold, geometric_threshold, points_per_check, depth+1, detailed_debug)
#             )
#         else:
#             rebuilt_children.append(c)

#     if detailed_debug:
#         debug_info(f"{indent}[{depth}] Finished children for {node_name}")

#     # ------------------------------------------------------------------
#     # REBUILD ORIGINAL NODE
#     # ------------------------------------------------------------------
#     try:
#         if isinstance(node, Union):
#             current_node = Union(rebuilt_children[0], rebuilt_children[1])
#         elif isinstance(node, SymRef):
#             current_node = SymRef(
#                 rebuilt_children[0],
#                 plane_normal=node.plane,
#                 point_on_plane=node.point_on_plane
#             )
#         elif isinstance(node, SymRot):
#             current_node = SymRot(
#                 rebuilt_children[0],
#                 axis=node.axis,
#                 center=node.center,
#                 n_fold=node.n
#             )
#         elif isinstance(node, SymTrans):
#             current_node = SymTrans(
#                 rebuilt_children[0], end_point=node.end_point, n_fold=node.n
#             )
#         elif hasattr(node, "child"):
#             kwargs = {k: v for k, v in node.__dict__.items() if k != "child"}
#             current_node = type(node)(rebuilt_children[0], **kwargs)
#         else:
#             current_node = type(node)(*rebuilt_children)

#         if detailed_debug:
#             debug_info(f"{indent}[{depth}] Rebuilt node {type(current_node).__name__}")
#     except Exception as e:
#         if detailed_debug:
#             debug_error(f"{indent}[{depth}] Failed rebuild: {e}")
#         return node

#     # --- HELPER: Geometric Verification ---
#     def check_geometric_fidelity(original_node, candidate_abstraction):
#         """Returns True if the abstraction geometrically resembles the original."""
#         try:
#             # Generate Local Point Clouds (relative to this node's frame)
#             pc_orig = get_point_cloud_from_dsl(original_node, points_per_box=points_per_check)
#             pc_cand = get_point_cloud_from_dsl(candidate_abstraction, points_per_box=points_per_check)
            
#             # If either is empty, fail safely (unless both are empty, which implies match)
#             if len(pc_orig) == 0 and len(pc_cand) == 0: return True
#             if len(pc_orig) == 0 or len(pc_cand) == 0: return False

#             # Calculate Chamfer
#             dist = calculate_chamfer_distance(pc_orig, pc_cand)
            
#             if detailed_debug:
#                 if dist > geometric_threshold:
#                     debug_info(f"{indent}[GEO REJECT] Chamfer {dist:.4f} > {geometric_threshold}")
#                 else:
#                     debug_info(f"{indent}[GEO PASS] Chamfer {dist:.4f} <= {geometric_threshold}")
            
#             return dist <= geometric_threshold
#         except Exception as e:
#             if detailed_debug: debug_error(f"Geometry check failed: {e}")
#             return False

#     # ------------------------------------------------------------------
#     # PAIR ABSTRACTION ATTEMPT
#     # ------------------------------------------------------------------
#     child_nodes = [
#         c for c in rebuilt_children if hasattr(c, "serialize") or isinstance(c, Abstraction)
#     ]

#     if len(child_nodes) == 1:
#         child = child_nodes[0]

#         if isinstance(child, Abstraction):
#             child_name = f"Abs({child.pattern_name})"
#             child_params = child.compressed_params
#             grandchildren = child.children
#         else:
#             child_name = type(child).__name__
#             child_params, gc_raw = child.serialize()[1]
#             grandchildren = [
#                 gc for gc in gc_raw if hasattr(gc, "serialize") or isinstance(gc, Abstraction)
#             ]

#         pair_sig = f"{type(current_node).__name__}({child_name})"

#         if pair_sig in pair_models:
#             if detailed_debug:
#                 debug_info(f"{indent}[{depth}] Checking PAIR: {pair_sig}")

#             model = pair_models[pair_sig]
#             p_params, _ = current_node.serialize()[1]

#             combined = t(torch.tensor(
#                 list(p_params or []) + list(child_params or []),
#                 dtype=torch.float32
#             )).unsqueeze(0)

#             if combined.shape[1] == model.input_dim:
#                 normalized = (combined - model.data_mean_) / model.data_std_
#                 _, recon = model(normalized)
#                 param_error = torch.max(torch.abs(recon - normalized)).item()

#                 if detailed_debug:
#                     debug_info(f"{indent}[{depth}] Pair error {param_error:.4f}")

#                 # 1. FAST CHECK: Parameter Error
#                 if param_error < error_threshold:
#                     encoding, _ = model(normalized)
                    
#                     # Create Candidate
#                     candidate = Abstraction(
#                         pair_sig,
#                         encoding.squeeze().tolist(),
#                         model,
#                         children=grandchildren
#                     )
                    
#                     # 2. SLOW CHECK: Geometric Fidelity
#                     if check_geometric_fidelity(current_node, candidate):
#                         if detailed_debug: debug_abstraction(f"Applied PAIR: {pair_sig}")
#                         return candidate

#     # ------------------------------------------------------------------
#     # SINGLETON ABSTRACTION ATTEMPT
#     # ------------------------------------------------------------------
#     name = type(current_node).__name__
#     if name in singleton_models:
#         if detailed_debug:
#             debug_info(f"{indent}[{depth}] Checking SINGLETON: {name}")

#         model = singleton_models[name]
#         params, _ = current_node.serialize()[1]

#         if params:
#             params_tensor = t(torch.tensor(params, dtype=torch.float32)).unsqueeze(0)

#             if params_tensor.shape[1] == model.input_dim:
#                 normalized = (params_tensor - model.data_mean_) / model.data_std_
#                 _, recon = model(normalized)
#                 param_error = torch.max(torch.abs(recon - normalized)).item()

#                 if detailed_debug:
#                     debug_info(f"{indent}[{depth}] Singleton error {param_error:.4f}")

#                 # 1. FAST CHECK
#                 if param_error < error_threshold:
#                     encoding, _ = model(normalized)
                    
#                     # Create Candidate
#                     candidate = Abstraction(
#                         name,
#                         encoding.squeeze().tolist(),
#                         model,
#                         children=child_nodes
#                     )
                    
#                     # 2. SLOW CHECK
#                     if check_geometric_fidelity(current_node, candidate):
#                         if detailed_debug: debug_abstraction(f"Applied SINGLETON: {name}")
#                         return candidate

#     # ------------------------------------------------------------------
#     # NO ABSTRACTION
#     # ------------------------------------------------------------------
#     if detailed_debug:
#         debug_info(f"{indent}[{depth}] No abstraction applied.")
#     return current_node

# def integrate_abstractions(
#     node,
#     singleton_models,
#     pair_models,
#     error_threshold=ERROR_THRESHOLD,
#     depth=0,
#     detailed_debug=False  # Added verbosity control flag
# ):
#     """Recursively abstract a DSL tree by replacing concrete patterns.

#     This is the main *application* function. It performs a post-order
#     traversal (bottom-up) of the DSL tree. At each node, it attempts
#     to match the node (and its immediate child) against the trained
#     `pair_models` and `singleton_models`.

#     If a match is found:
#     1.  It extracts the parameters.
#     2.  It normalizes them using the model's `data_mean_` and `data_std_`.
#     3.  It checks the reconstruction error.
#     4.  If error < `error_threshold`, it *encodes* the normalized
#         parameters to get a latent vector.
#     5.  It replaces the concrete node(s) with a new `Abstraction` node
#         containing the pattern name, latent vector, and model.

#     Parameters
#     ----------
#     node : object
#         The root node of the DSL tree to abstract.
#     singleton_models : dict
#         Dictionary of trained models for singleton patterns.
#     pair_models : dict
#         Dictionary of trained models for pair patterns.
#     error_threshold : float, optional
#         The max normalized error to allow for an abstraction.
#     depth : int, optional
#         Internal tracking of recursion depth for logging.
#     detailed_debug : bool, optional
#         If True, prints verbose step-by-step logging. (Default: False)

#     Returns
#     -------
#     object
#         The new root node of the (potentially) abstracted DSL tree.
#     """
#     indent = "  " * depth
#     node_name = f"Abs({node.pattern_name})" if isinstance(node, Abstraction) else type(node).__name__

#     if detailed_debug:
#         debug_info(f"{indent}[{depth}] Processing: {node_name}")

#     # ------------------------------------------------------------------
#     # ALREADY ABSTRACTED
#     # ------------------------------------------------------------------
#     if isinstance(node, Abstraction):
#         if detailed_debug:
#             debug_info(f"{indent}[{depth}] Already abstracted; recursing into children.")
#         node.children = [
#             integrate_abstractions(c, singleton_models, pair_models,
#                                    error_threshold, depth+1, detailed_debug)
#             for c in node.children
#         ]
#         return node

#     # ------------------------------------------------------------------
#     # NODE WITHOUT serialize()
#     # ------------------------------------------------------------------
#     if not hasattr(node, "serialize"):
#         if detailed_debug:
#             debug_error(f"{indent}[{depth}] Node {type(node).__name__} lacks serialize().")
#         return node

#     # ------------------------------------------------------------------
#     # RECURSE
#     # ------------------------------------------------------------------
#     try:
#         _, (_, children) = node.serialize()
#     except Exception as e:
#         if detailed_debug:
#             debug_error(f"{indent}[{depth}] Failed to serialize {node_name}: {e}")
#         return node

#     valid_children = [
#         c for c in children if hasattr(c, "serialize") or isinstance(c, Abstraction)
#     ]
#     if detailed_debug:
#         debug_info(f"{indent}[{depth}] Recursing into {len(valid_children)} children.")

#     rebuilt_children = []
#     for c in children:
#         if hasattr(c, "serialize") or isinstance(c, Abstraction):
#             rebuilt_children.append(
#                 integrate_abstractions(c, singleton_models, pair_models,
#                                        error_threshold, depth+1, detailed_debug)
#             )
#         else:
#             rebuilt_children.append(c)

#     if detailed_debug:
#         debug_info(f"{indent}[{depth}] Finished children for {node_name}")

#     # ------------------------------------------------------------------
#     # REBUILD ORIGINAL NODE
#     # ------------------------------------------------------------------
#     try:
#         if isinstance(node, Union):
#             current_node = Union(rebuilt_children[0], rebuilt_children[1])
#         elif isinstance(node, SymRef):
#             current_node = SymRef(
#                 rebuilt_children[0],
#                 plane_normal=node.plane,
#                 point_on_plane=node.point_on_plane
#             )
#         elif isinstance(node, SymRot):
#             current_node = SymRot(
#                 rebuilt_children[0],
#                 axis=node.axis,
#                 center=node.center,
#                 n_fold=node.n
#             )
#         elif isinstance(node, SymTrans):
#             current_node = SymTrans(
#                 rebuilt_children[0], end_point=node.end_point, n_fold=node.n
#             )
#         elif hasattr(node, "child"):
#             kwargs = {k: v for k, v in node.__dict__.items() if k != "child"}
#             current_node = type(node)(rebuilt_children[0], **kwargs)
#         else:
#             current_node = type(node)(*rebuilt_children)

#         if detailed_debug:
#             debug_info(f"{indent}[{depth}] Rebuilt node {type(current_node).__name__}")
#     except Exception as e:
#         if detailed_debug:
#             debug_error(f"{indent}[{depth}] Failed rebuild: {e}")
#         return node

#     # ------------------------------------------------------------------
#     # PAIR ABSTRACTION ATTEMPT
#     # ------------------------------------------------------------------
#     child_nodes = [
#         c for c in rebuilt_children if hasattr(c, "serialize") or isinstance(c, Abstraction)
#     ]

#     if len(child_nodes) == 1:
#         child = child_nodes[0]

#         if isinstance(child, Abstraction):
#             child_name = f"Abs({child.pattern_name})"
#             child_params = child.compressed_params
#             grandchildren = child.children
#         else:
#             child_name = type(child).__name__
#             child_params, gc_raw = child.serialize()[1]
#             grandchildren = [
#                 gc for gc in gc_raw if hasattr(gc, "serialize") or isinstance(gc, Abstraction)
#             ]

#         pair_sig = f"{type(current_node).__name__}({child_name})"

#         if pair_sig in pair_models:
#             if detailed_debug:
#                 debug_info(f"{indent}[{depth}] Checking PAIR: {pair_sig}")

#             model = pair_models[pair_sig]
#             p_params, _ = current_node.serialize()[1]

#             combined = t(torch.tensor(
#                 list(p_params or []) + list(child_params or []),
#                 dtype=torch.float32
#             )).unsqueeze(0)

#             if combined.shape[1] == model.input_dim:
#                 normalized = (combined - model.data_mean_) / model.data_std_
#                 _, recon = model(normalized)
#                 error = torch.max(torch.abs(recon - normalized)).item()

#                 if detailed_debug:
#                     debug_info(f"{indent}[{depth}] Pair error {error:.4f}")

#                 if error < error_threshold:
#                     if detailed_debug:
#                         debug_abstraction(f"Applied PAIR: {pair_sig}")

#                     encoding, _ = model(normalized)

#                     return Abstraction(
#                         pair_sig,
#                         encoding.squeeze().tolist(),
#                         model,
#                         children=grandchildren
#                     )

#     # ------------------------------------------------------------------
#     # SINGLETON ABSTRACTION
#     # ------------------------------------------------------------------
#     name = type(current_node).__name__
#     if name in singleton_models:
#         if detailed_debug:
#             debug_info(f"{indent}[{depth}] Checking SINGLETON: {name}")

#         model = singleton_models[name]
#         params, _ = current_node.serialize()[1]

#         if params:
#             params_tensor = t(torch.tensor(params, dtype=torch.float32)).unsqueeze(0)

#             if params_tensor.shape[1] == model.input_dim:
#                 normalized = (params_tensor - model.data_mean_) / model.data_std_
#                 _, recon = model(normalized)
#                 error = torch.max(torch.abs(recon - normalized)).item()

#                 if detailed_debug:
#                     debug_info(f"{indent}[{depth}] Singleton error {error:.4f}")

#                 if error < error_threshold:
#                     if detailed_debug:
#                         debug_abstraction(f"Applied SINGLETON: {name}")

#                     encoding, _ = model(normalized)
#                     return Abstraction(
#                         name,
#                         encoding.squeeze().tolist(),
#                         model,
#                         children=child_nodes
#                     )

#     # ------------------------------------------------------------------
#     # NO ABSTRACTION
#     # ------------------------------------------------------------------
#     if detailed_debug:
#         debug_info(f"{indent}[{depth}] No abstraction applied.")
#     return current_node