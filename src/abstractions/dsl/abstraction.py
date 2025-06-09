"""
Abstraction class for DSL-driven neural shape expansion.

This module defines the `Abstraction` class, a subclass of `Shape`, which uses a neural
model (typically a decoder network) to convert a set of latent float parameters into a
concrete shape. The `expand()` method applies the decoder and uses the `instantiate` function
to build the resulting shape from decoded values and structural type information.

Key Features:
- Combines symbolic type structure (`type_list`) with neural representations.
- Integrates PyTorch models to support differentiable shape expansion.
- Supports downstream rendering and analysis through `get_box_list()`.

Typical use case:
    abstraction = Abstraction(type_list, float_params, other_params, model)
    concrete_shape = abstraction.expand()
"""

import torch
from torch import nn

from abstractions.dsl.core import Shape, left_pad
from abstractions.dsl.instantiation import instantiate
from abstractions.primitives.shapes import Box


class Abstraction(Shape):
    """
    A shape abstraction that can expand itself using a neural model decoder.

    Args:
        type_list (list[type]): List of types describing the shape structure.
        float_parameters (list[float]): List of float parameters used for expansion.
        other_parameters (list): List of other parameters including shapes and strings.
        model (nn.Module): Neural model with a decoder to expand float parameters.

    Returns:
        Expanded concrete shape after decoding.
    """

    def __init__(
        self,
        type_list: list[type],
        float_parameters: list[float],
        other_parameters: list,
        model: nn.Module,
    ):
        super().__init__(
            children=[child for child in other_parameters if isinstance(child, Shape)]
        )
        self.type_list = type_list
        self.float_parameters = float_parameters
        self.other_parameters = other_parameters
        self.model = model

    def __str__(self):
        output = "Abstraction(\n"

        for child in self.children:
            output += f"    {left_pad(str(child), '    ', 1)},\n"

        for p in self.float_parameters:
            output += f"    {p:.3f},\n"

        for p in self.other_parameters:
            if not isinstance(p, Shape):
                output += f"    {p},\n"

        output += ")"
        return output

    def expand(self) -> Shape:
        """
        Expands the abstract parameters into a concrete shape using the decoder from the neural model.

        Returns:
            Shape: The expanded, instantiated shape.
        """
        self.model.eval()
        decoder = self.model.decoder
        decoder_input = torch.tensor(self.float_parameters, dtype=torch.float32)[
            None, :
        ]
        decoder_output = decoder(decoder_input)
        expanded_float_parameters = decoder_output[0, :].tolist()
        full_parameter_list = []
        i_floats = 0
        i_others = 0

        for type in self.type_list:
            if type == float:
                full_parameter_list.append(expanded_float_parameters[i_floats])
                i_floats += 1
            elif type == int or type == str or type == Shape:
                full_parameter_list.append(self.other_parameters[i_others])
                i_others += 1

        expanded_shape = instantiate(self.type_list, full_parameter_list)
        return expanded_shape

    def get_box_list(self) -> list[Box]:
        return self.expand().get_box_list()

    def param_tuple(self):
        return Abstraction, (
            self.type_list,
            self.float_parameters,
            self.other_parameters,
            self.model,
        )
