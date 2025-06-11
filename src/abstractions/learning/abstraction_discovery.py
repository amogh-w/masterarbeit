from collections import defaultdict
import torch
from torch.optim import AdamW

from abstractions.dsl.abstraction import Abstraction
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Union
from abstractions.learning.models import Autoencoder
from abstractions.learning.utils import (
    is_well_explained,
    prepare_autoencoder_train_data,
)


def find_abstractions(structures, retrain_iterations=2, error_threshold=0.01):
    """
    Trains an autoencoder for each structure type to detect low-dimensional latent representations (abstractions).

    Args:
        structures (dict): A dictionary of structure names to lists of parameter tuples.
        retrain_iterations (int): Number of times to retrain the autoencoder per structure.
        error_threshold (float): Maximum reconstruction error to consider a structure well explained.

    Returns:
        tuple: A tuple containing the dictionary of trained models and their corresponding training losses.
    """
    losses = defaultdict(list)
    models = {}

    for name, parameters in structures.items():
        # for now, only consider float parameters
        valid_indices = [
            i for i in range(len(parameters[0])) if isinstance(parameters[0][i], float)
        ]
        num_float_parameters = len(valid_indices)

        if num_float_parameters <= 0:
            continue

        float_parameters = [
            [p[valid_index] for p in parameters] for valid_index in valid_indices
        ]
        well_explained = torch.ones(len(float_parameters[0]), dtype=torch.bool)

        for i in range(retrain_iterations):
            losses[name].append([])
            train_dl = prepare_autoencoder_train_data(
                float_parameters, mask=well_explained
            )
            model = Autoencoder(num_float_parameters, num_float_parameters - 1)
            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
            loss_fn = lambda pred, target: torch.mean(
                torch.max(torch.abs(pred - target), dim=-1)[0], dim=0
            )
            epochs = 100
            model.train()
            print(
                f"Iteration {i+1}/{retrain_iterations}: training model for {name} on {len(train_dl.dataset)} structures."
            )

            for epoch in range(epochs):
                running_train_loss = 0.0

                for x in train_dl:
                    x = x[0]
                    optimizer.zero_grad()
                    encoding, x_rec = model(x)
                    loss = loss_fn(x_rec, x)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()

                epoch_train_loss = running_train_loss / len(train_dl)
                losses[name][i].append(epoch_train_loss)
                # print(f"{name}, P{output_index}, Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.3f}")

            well_explained = is_well_explained(
                torch.tensor(float_parameters).swapaxes(0, 1),
                model,
                threshold=error_threshold,
            )

            if well_explained.sum() <= 0:
                # fix for debugging
                # if no structures are explained well, just train on the first one
                well_explained[0] = True

        models[name] = model
        print(f"Trained model for {name}. Final train loss: {losses[name][-1][-1]}")

    return models, losses


def integrate_abstractions(
    shape: list[Shape] | Shape, models: dict[str, Autoencoder], error_threshold: float
) -> list[Shape] | Shape:
    """
    Recursively replaces parts of a shape (or list of shapes) with abstraction nodes if they are well explained by the learned models.

    Args:
        shape (Shape | list[Shape]): A shape or list of shapes to abstract.
        models (dict): A dictionary mapping structure names to trained Autoencoder models.
        error_threshold (float): Maximum reconstruction error to qualify for abstraction.

    Returns:
        Shape | list[Shape]: The input shape(s) with abstractions integrated.
    """
    if isinstance(shape, list):
        return [integrate_abstractions(s, models, error_threshold) for s in shape]

    if isinstance(shape, Union):
        # check pairs
        parent_type, (child1, child2) = shape.param_tuple()
        type1, parameters1 = child1.param_tuple()
        type2, parameters2 = child2.param_tuple()
        type_str = f"{parent_type.__name__}({type1.__name__}, {type2.__name__})"
        all_parameters = list(parameters1) + list(parameters2)
        float_parameters = [p for p in all_parameters if isinstance(p, float)]
        other_parameters = [
            p
            for p in all_parameters
            if isinstance(p, int) or isinstance(p, str) or isinstance(p, Shape)
        ]

        if len(float_parameters) > 0 and type_str in models:
            # check if the appropriate abstraction fits well
            model_input = torch.tensor(float_parameters)[None, :]
            encoding, reconstruction = models[type_str](model_input)
            encoding = encoding[0, :]
            reconstruction = reconstruction[0, :]
            error, _ = torch.max(torch.abs(reconstruction - model_input), dim=-1)
            fits_well = error.item() < error_threshold
        else:
            # if there are no float parameters or there is no model for the current subtree, there is no abstraction that can be integrated
            fits_well = False
            encoding = None

        if fits_well:
            # integrate abstraction
            # the children of the abstraction have to be declared as "Shape" and not their specific type, so the instantiate function does not try to instantiate them further
            type_list = (
                [parent_type, type1]
                + [Shape if isinstance(p, Shape) else type(p) for p in parameters1]
                + [type2]
                + [Shape if isinstance(p, Shape) else type(p) for p in parameters2]
            )
            new_shape = Abstraction(
                type_list, encoding.tolist(), other_parameters, models[type_str]
            )
        else:
            # dont integrate abstraction
            new_shape = Union(child1, child2)

        # continue integrating abstractions in children
        for i in range(len(new_shape.children)):
            new_shape.children[i] = integrate_abstractions(
                new_shape.children[i], models, error_threshold
            )

        return new_shape
    else:
        # check singletons
        parent_type, parameters = shape.param_tuple()
        type_str = type.__name__
        float_parameters = [p for p in parameters if isinstance(p, float)]
        other_parameters = [
            p
            for p in parameters
            if isinstance(p, int) or isinstance(p, str) or isinstance(p, Shape)
        ]

        if len(float_parameters) > 0 and type_str in models:
            # check if the appropriate abstraction fits well
            model_input = torch.tensor(float_parameters)[None, :]
            encoding, reconstruction = models[type_str](model_input)
            encoding = encoding[0, :]
            reconstruction = reconstruction[0, :]
            error, _ = torch.max(torch.abs(reconstruction - model_input), dim=-1)
            fits_well = error.item() < error_threshold
        else:
            # if there are no float parameters or there is no model for the current subtree, there is no abstraction that can be integrated
            fits_well = False
            encoding = None

        if fits_well:
            # integrate abstraction
            # the children of the abstraction have to be declared as "Shape" and not their specific type, so the instantiate function does not try to instantiate them further
            type_list = [parent_type] + [
                Shape if isinstance(p, Shape) else type(p) for p in parameters
            ]
            new_shape = Abstraction(
                type_list, encoding.tolist(), other_parameters, models[type_str]
            )
        else:
            # dont integrate abstraction
            new_shape = parent_type(*parameters)

        # continue integrating abstractions in children
        for i in range(len(new_shape.children)):
            new_shape.children[i] = integrate_abstractions(
                new_shape.children[i], models, error_threshold
            )

        return new_shape
