primitives/shapes.py

class Box:
    def __init__(self, center: Tensor, scale: Tensor):

        self.center = center
        self.scale = scale

dsl/abstraction.py

import torch
from torch import nn

from abstractions.dsl.core import Shape, left_pad
from abstractions.dsl.instantiation import instantiate
from abstractions.primitives.shapes import Box


class Abstraction(Shape):
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


dsl/core.py

class Shape:
    def __init__(self, children: list[Shape]):
        self.children = children

    def __str__(self):
        return "Shape"

    def get_box_list(self) -> list[Box]:
        pass

    def param_tuple(self):
        pass

dsl/instantiation.py

from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Rect, Move, Union, SymTrans, SymRef


def instantiate(type_list: list[type], param_list: list):

    def _instantiate(type_list: list[type], param_list: list):
        token = type_list.pop(0)

        if issubclass(token, Rect):
            s_x = _instantiate(type_list, param_list)
            s_y = _instantiate(type_list, param_list)
            return Rect(s_x, s_y)
        elif issubclass(token, Move):
            child = _instantiate(type_list, param_list)
            t_x = _instantiate(type_list, param_list)
            t_y = _instantiate(type_list, param_list)
            return Move(child, t_x, t_y)
        elif issubclass(token, Union):
            child1 = _instantiate(type_list, param_list)
            child2 = _instantiate(type_list, param_list)
            return Union(child1, child2)
        elif issubclass(token, SymTrans):
            child = _instantiate(type_list, param_list)
            axis = _instantiate(type_list, param_list)
            dist = _instantiate(type_list, param_list)
            degree = _instantiate(type_list, param_list)
            return SymTrans(child, axis, dist, degree)
        elif issubclass(token, SymRef):
            child = _instantiate(type_list, param_list)
            axis = _instantiate(type_list, param_list)
            return SymRef(child, axis)
        elif issubclass(token, float):
            return float(param_list.pop(0))
        elif issubclass(token, int):
            return int(param_list.pop(0))
        elif issubclass(token, str):
            return str(param_list.pop(0))
        elif issubclass(token, Shape):
            return param_list.pop(0)
        else:
            raise ValueError(f"Unknown token: {token}")

    return _instantiate(type_list.copy(), param_list.copy())


dsl/nodes.py

class Rect(Shape):

    def __init__(self, s_x: float, s_y: float):
        super().__init__(children=[])
        self.s_x = s_x
        self.s_y = s_y

    def __str__(self):
        params = f"{self.s_x:.3f},\n{self.s_y:.3f}"
        indented = textwrap.indent(params, "    ")
        return f"Rect(\n{indented}\n)"

    def get_box_list(self) -> list[Box]:
        return [
            Box(
                center=torch.tensor([0.0, 0.0]),
                scale=torch.tensor([self.s_x, self.s_y]),
            )
        ]

    def param_tuple(self):
        return Rect, (self.s_x, self.s_y)


class Move(Shape):

    def __init__(self, child: Shape, t_x: float, t_y: float):
        super().__init__(children=[child])
        self.t_x = t_x
        self.t_y = t_y

    def __str__(self):
        child_str = textwrap.indent(str(self.children[0]), "    ")
        params = f"{self.t_x:.3f},\n{self.t_y:.3f}"
        indented_params = textwrap.indent(params, "    ")
        return f"Move(\n{child_str},\n{indented_params}\n)"

    def get_box_list(self) -> list[Box]:
        child_boxes = self.children[0].get_box_list()

        for box in child_boxes:
            box.center[0] += self.t_x
            box.center[1] += self.t_y

        return child_boxes

    def param_tuple(self):
        return Move, (self.children[0], self.t_x, self.t_y)


class Union(Shape):


    def __init__(self, child1: Shape, child2: Shape):
        if not isinstance(child1, Shape) or not isinstance(child2, Shape):
            raise TypeError("Union expects two Shape instances.")
        super().__init__(children=[child1, child2])

    def __str__(self):
        child1_str = textwrap.indent(str(self.children[0]), "    ")
        child2_str = textwrap.indent(str(self.children[1]), "    ")
        return f"Union(\n{child1_str},\n{child2_str}\n)"

    def get_box_list(self) -> list[Box]:
        child_boxes1 = self.children[0].get_box_list()
        child_boxes2 = self.children[1].get_box_list()
        return child_boxes1 + child_boxes2

    def param_tuple(self):
        return Union, (self.children[0], self.children[1])


class SymTrans(Shape):

    def __init__(self, child: Shape, axis: str, dist: float, degree: int):
        super().__init__(children=[child])
        self.axis = axis
        self.dist = dist
        self.degree = degree

    def __str__(self):
        child_str = textwrap.indent(str(self.children[0]), "    ")
        params = f"{self.axis},\n{self.dist:.3f},\n{self.degree}"
        indented_params = textwrap.indent(params, "    ")
        return f"SymTrans(\n{child_str},\n{indented_params}\n)"

    def get_box_list(self) -> list[Box]:
        child_boxes = self.children[0].get_box_list()
        copies = []
        dt = (self.dist / (self.degree - 1)) * (
            torch.tensor([1.0, 0.0]) if self.axis == "x" else torch.tensor([0.0, 1.0])
        )

        for box in child_boxes:
            for _ in range(self.degree - 1):
                copies.append(Box(center=box.center + dt, scale=box.scale))

        return child_boxes + copies

    def param_tuple(self):
        return SymTrans, (self.children[0], self.axis, self.dist, self.degree)


class SymRef(Shape):
    def __init__(self, child: Shape, axis: str):
        super().__init__(children=[child])
        self.axis = axis

    def __str__(self):
        child_str = textwrap.indent(str(self.children[0]), "    ")
        return f"SymRef(\n{child_str},\n    {self.axis}\n)"

    def get_box_list(self) -> list[Box]:
        child_boxes = self.children[0].get_box_list()
        copies = []
        dt = (
            torch.tensor([-1.0, 1.0]) if self.axis == "x" else torch.tensor([1.0, -1.0])
        )

        for box in child_boxes:
            copies.append(Box(center=dt * box.center, scale=box.scale))

        return child_boxes + copies

    def param_tuple(self):
        return SymRef, (self.children[0], self.axis)

learning/abstraction_discovery.py

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

learning / models.py


from torch import nn


class MLP(nn.Module):


    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = nn.ReLU()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            self.activation,
            nn.Linear(32, 32),
            self.activation,
            nn.Linear(32, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            self.activation,
            nn.Linear(32, 32),
            self.activation,
            nn.Linear(32, input_dim),
        )

    def forward(self, x):

        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return encoding, reconstruction

learning / utils.py

from collections import defaultdict
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Move, SymRef, SymTrans, Union
from abstractions.learning.models import Autoencoder


def prepare_mlp_train_data(parameters, output_index):
    tensor = torch.tensor(parameters, dtype=torch.float32)  # shape: [N, D]

    # Separate input and output columns
    input_mask = [i for i in range(tensor.shape[1]) if i != output_index]
    input_data = tensor[:, input_mask]  # shape: [N, D-1]
    output_data = tensor[:, output_index].unsqueeze(1)  # shape: [N, 1]

    train_dataset = TensorDataset(input_data, output_data)
    return DataLoader(train_dataset, batch_size=64, shuffle=True)


def prepare_autoencoder_train_data(parameters, mask):

    tensor = torch.tensor(parameters, dtype=torch.float32).T  # transpose to [N, D]
    masked_data = tensor[mask]  # apply boolean mask to rows
    dataset = TensorDataset(masked_data)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def is_well_explained(
    parameters: Tensor, model: Autoencoder, threshold: float
) -> Tensor:

    # parameters.shape = (batch_size, num_parameters)
    model.eval()

    with torch.no_grad():
        encodings, reconstructions = model(parameters)
        error, _ = torch.max(torch.abs(reconstructions - parameters), dim=-1)
        well_explained = error < threshold

    return well_explained


def add(structures1, structures2):


    for key in structures2.keys():
        structures1[key] += structures2[key]

    return structures1


def get_singletons(shapes: Shape | list[Shape]):

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
