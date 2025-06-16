"""
utils.py

Utility functions to prepare training data and analyze reconstruction
performance for neural models in the abstraction discovery pipeline.

Functions include data loaders for MLPs and autoencoders, evaluation of
well-explained instances, and helpers for managing symbolic structure
datasets.
"""

from collections import defaultdict
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Move, SymRef, SymTrans, Union
from abstractions.learning.models import Autoencoder


def prepare_mlp_train_data(parameters, output_index):
    """
    Prepares training data for an MLP by excluding one parameter as the target output.

    Parameters
    ----------
    parameters : list[tuple]
        A list of parameter tuples (N samples × D parameters).
    output_index : int
        The index of the parameter to predict.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader containing input-output training pairs.
    """
    tensor = torch.tensor(parameters, dtype=torch.float32)  # shape: [N, D]

    # Separate input and output columns
    input_mask = [i for i in range(tensor.shape[1]) if i != output_index]
    input_data = tensor[:, input_mask]  # shape: [N, D-1]
    output_data = tensor[:, output_index].unsqueeze(1)  # shape: [N, 1]

    train_dataset = TensorDataset(input_data, output_data)
    return DataLoader(train_dataset, batch_size=64, shuffle=True)


def prepare_autoencoder_train_data(parameters, mask):
    """
    Prepares training data for an autoencoder based on a boolean mask.

    Parameters
    ----------
    parameters : list[tuple]
        A list of parameter tuples (N samples × D parameters).
    mask : Tensor
        A 1D boolean mask (length N) indicating which parameter rows to include.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader for the masked training dataset.
    """
    tensor = torch.tensor(parameters, dtype=torch.float32).T  # transpose to [N, D]
    masked_data = tensor[mask]  # apply boolean mask to rows
    dataset = TensorDataset(masked_data)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def is_well_explained(
    parameters: Tensor, model: Autoencoder, threshold: float
) -> Tensor:
    """
    Determines which parameter vectors are well explained by an autoencoder model.

    Parameters
    ----------
    parameters : Tensor
        A batch of parameter vectors of shape (N, D).
    model : Autoencoder
        The trained autoencoder model.
    threshold : float
        Maximum allowed reconstruction error for a point to be considered well explained.

    Returns
    -------
    Tensor
        A 1D boolean tensor of shape (N,) indicating which vectors are well explained.
    """
    # parameters.shape = (batch_size, num_parameters)
    model.eval()

    with torch.no_grad():
        encodings, reconstructions = model(parameters)
        error, _ = torch.max(torch.abs(reconstructions - parameters), dim=-1)
        well_explained = error < threshold

    return well_explained


def add(structures1, structures2):
    """
    Merges two defaultdict(list) structures by appending values for matching keys.

    Parameters
    ----------
    structures1 : defaultdict
        The primary dictionary to which values will be added.
    structures2 : defaultdict
        The secondary dictionary whose values will be appended to structures1.

    Returns
    -------
    defaultdict
        The updated structures1 dictionary with merged values.
    """

    for key in structures2.keys():
        structures1[key] += structures2[key]

    return structures1


def get_singletons(shapes: Shape | list[Shape]):
    """
    Recursively extracts all singleton shape structures (non-composite shapes) from a shape or list of shapes.

    Parameters
    ----------
    shapes : Shape or list of Shape
        A single Shape instance or a list of Shape instances.

    Returns
    -------
    defaultdict
        A dictionary mapping shape type names to lists of their parameter tuples.
    """
    singletons = defaultdict(list)

    if isinstance(shapes, list):
        for shape in shapes:
            singletons = add(singletons, get_singletons(shape))
        return singletons

    shape_type, parameters = shapes.param_tuple()
    singletons[shape_type.__name__].append(parameters)

    match shapes:
        case Move() | SymTrans() | SymRef():
            singletons = add(singletons, get_singletons(shapes.children[0]))
        case Union():
            for child in shapes.children:
                singletons = add(singletons, get_singletons(child))
        case _:
            pass

    return singletons


def get_pairs(shapes: Shape | list[Shape]):
    """
    Recursively extracts all binary composite structures (e.g., Union of two shapes) from a shape or list of shapes.

    Parameters
    ----------
    shapes : Shape or list of Shape
        A single Shape instance or a list of Shape instances.

    Returns
    -------
    defaultdict
        A dictionary mapping composite structure type descriptions to lists of combined child parameter tuples.
    """
    pairs = defaultdict(list)

    if isinstance(shapes, list):
        for shape in shapes:
            pairs = add(pairs, get_pairs(shape))
        return pairs

    match shapes:
        case Move() | SymTrans() | SymRef():
            return get_pairs(shapes.children[0])
        case Union():
            type_, (child1, child2) = shapes.param_tuple()
            type1, params1 = child1.param_tuple()
            type2, params2 = child2.param_tuple()
            type_str = f"{type_.__name__}({type1.__name__}, {type2.__name__})"
            current_structures = defaultdict(list)
            current_structures[type_str].append(params1 + params2)
            current_structures = add(get_pairs(shapes.children[0]), current_structures)
            current_structures = add(get_pairs(shapes.children[1]), current_structures)
            return current_structures
        case _:
            return defaultdict(list)
