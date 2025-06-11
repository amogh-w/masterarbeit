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

    Args:
        parameters (list): A list of parameter tuples.
        output_index (int): The index of the parameter to predict.

    Returns:
        DataLoader: A DataLoader for the training dataset.
    """
    tensor = torch.tensor(parameters).swapaxes(0, 1)
    input_data = tensor[
        :, tuple([i for i in range(tensor.shape[1]) if i != output_index])
    ]
    output_data = tensor[:, (output_index,)]
    train_data = TensorDataset(input_data, output_data)
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    return train_dl


def prepare_autoencoder_train_data(parameters, mask):
    """
    Prepares training data for an autoencoder based on a boolean mask.

    Args:
        parameters (list): A list of parameter tuples.
        mask (Tensor): A boolean mask indicating which data points to include.

    Returns:
        DataLoader: A DataLoader for the masked training dataset.
    """
    tensor = torch.tensor(parameters).swapaxes(0, 1)
    train_data = TensorDataset(tensor[mask, :])
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    return train_dl


def is_well_explained(
    parameters: Tensor, model: Autoencoder, threshold: float
) -> Tensor:
    """
    Determines which parameter vectors are well explained by an autoencoder model.

    Args:
        parameters (Tensor): A batch of parameter vectors.
        model (Autoencoder): The trained autoencoder model.
        threshold (float): The maximum reconstruction error allowed for a point to be considered well explained.

    Returns:
        Tensor: A boolean tensor indicating which parameter vectors are well explained.
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
    Adds values from structures2 into structures1 by appending lists for matching keys.

    Args:
        structures1 (defaultdict): The primary dictionary to which values will be added.
        structures2 (defaultdict): The secondary dictionary whose values will be appended.

    Returns:
        defaultdict: The updated structures1 dictionary.
    """

    for key in structures2.keys():
        structures1[key] += structures2[key]

    return structures1


# In[9]:


def get_singletons(shapes: Shape | list[Shape]):
    """
    Recursively extracts all singleton shape structures (non-composite shapes) from a shape or list of shapes.

    Args:
        shapes (Shape | list[Shape]): A single Shape or a list of Shapes.

    Returns:
        defaultdict: A dictionary mapping shape type names to lists of their parameter tuples.
    """
    if isinstance(shapes, list):
        singletons = defaultdict(list)

        for shape in shapes:
            singletons = add(singletons, get_singletons(shape))

        return singletons

    type, parameters = shapes.param_tuple()
    type_str = type.__name__
    current_structures = defaultdict(list)
    current_structures[type_str].append(parameters)

    if (
        isinstance(shapes, Move)
        or isinstance(shapes, SymTrans)
        or isinstance(shapes, SymRef)
    ):
        current_structures = add(get_singletons(shapes.children[0]), current_structures)
    elif isinstance(shapes, Union):
        current_structures = add(get_singletons(shapes.children[0]), current_structures)
        current_structures = add(get_singletons(shapes.children[1]), current_structures)

    return current_structures


# In[10]:


def get_pairs(shapes: Shape | list[Shape]):
    """
    Recursively extracts all binary composite structures (e.g., Union of two shapes) from a shape or list of shapes.

    Args:
        shapes (Shape | list[Shape]): A single Shape or a list of Shapes.

    Returns:
        defaultdict: A dictionary mapping composite type descriptions to lists of parameter tuples.
    """
    if isinstance(shapes, list):
        pairs = defaultdict(list)

        for shape in shapes:
            pairs = add(pairs, get_pairs(shape))

        return pairs

    if (
        isinstance(shapes, Move)
        or isinstance(shapes, SymTrans)
        or isinstance(shapes, SymRef)
    ):
        return get_pairs(shapes.children[0])
    elif isinstance(shapes, Union):
        type, (child1, child2) = shapes.param_tuple()
        type1, parameters1 = child1.param_tuple()
        type2, parameters2 = child2.param_tuple()
        type_str = f"{type.__name__}({type1.__name__}, {type2.__name__})"
        current_structures = defaultdict(list)
        current_structures[type_str].append(parameters1 + parameters2)
        current_structures = add(get_pairs(shapes.children[0]), current_structures)
        current_structures = add(get_pairs(shapes.children[1]), current_structures)
        return current_structures
    else:
        return defaultdict(list)
