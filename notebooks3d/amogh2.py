#!/usr/bin/env python
# coding: utf-8

# In[42]:


# Cell 1: Imports and Setup
import sys
import os
import random
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../src"))

from abstractions3d.primitives.shapes import Box3D
from abstractions3d.primitives.visualization import visualize_boxes_3d
from abstractions3d.dsl.core import Shape3D
from abstractions3d.dsl.nodes import Rect3D, Move3D, Union3D, SymTrans3D, SymRef3D
from abstractions3d.dsl.instantiation import instantiate_3d
from abstractions3d.data.blueprint import Table3D, Chair3D, Bench3D

# Set random seeds
random.seed(42)
torch.manual_seed(42)


# In[43]:


import random
from typing import List
from abstractions3d.data.blueprint import Table3D, Chair3D, Bench3D
from abstractions3d.dsl.nodes import Union3D

def sample_uniform(low: float, high: float) -> float:
    return random.uniform(low, high)

def generate_dataset(num_samples: int = 50) -> List[Union3D]:
    dataset = []
    for _ in range(num_samples):
        dataset.append(Table3D(
            top_length=sample_uniform(3.0, 6.0),
            top_depth=sample_uniform(1.5, 3.0),
            top_thickness=sample_uniform(0.1, 0.3),
            leg_length=sample_uniform(0.2, 0.4),
            leg_depth=sample_uniform(0.2, 0.4),
            leg_height=sample_uniform(1.0, 2.0)
        ))
        dataset.append(Chair3D(
            seat_length=sample_uniform(1.0, 2.0),
            seat_depth=sample_uniform(1.0, 2.0),
            seat_thickness=sample_uniform(0.2, 0.5),
            leg_length=sample_uniform(0.1, 0.3),
            leg_depth=sample_uniform(0.1, 0.3),
            leg_height=sample_uniform(0.5, 1.0),
            backrest_height=sample_uniform(1.0, 2.0),
            backrest_thickness=sample_uniform(0.1, 0.3)
        ))
        backrest_height = sample_uniform(0.0, 1.5)
        dataset.append(Bench3D(
            seat_length=sample_uniform(2.0, 5.0),
            seat_depth=sample_uniform(0.5, 1.0),
            seat_thickness=sample_uniform(0.2, 0.5),
            leg_length=sample_uniform(0.1, 0.3),
            leg_depth=sample_uniform(0.1, 0.3),
            leg_height=sample_uniform(0.5, 1.0),
            backrest_height=backrest_height,
            backrest_thickness=sample_uniform(0.0, 0.3) if backrest_height > 0 else 0.0
        ))
    return dataset


# In[44]:


# Cell 3: Utilities

def get_singletons(shapes):
    singletons = defaultdict(list)
    if isinstance(shapes, list):
        for s in shapes:
            res = get_singletons(s)
            for k, v in res.items():
                singletons[k] += v
        return singletons

    cls, params = shapes.param_tuple()
    singletons[cls.__name__].append(params)

    if hasattr(shapes, 'children'):
        for child in getattr(shapes, 'children', []):
            res = get_singletons(child)
            for k, v in res.items():
                singletons[k] += v
    return singletons

def get_pairs(shapes):
    pairs = defaultdict(list)
    if isinstance(shapes, list):
        for s in shapes:
            res = get_pairs(s)
            for k, v in res.items():
                pairs[k] += v
        return pairs

    if hasattr(shapes, 'children') and len(shapes.children) == 2:
        cls, (child1, child2) = shapes.param_tuple()
        type_str = f"{cls.__name__}({child1.param_tuple()[0].__name__},{child2.param_tuple()[0].__name__})"
        pairs[type_str].append(child1.param_tuple()[1] + child2.param_tuple()[1])
        for child in shapes.children:
            res = get_pairs(child)
            for k, v in res.items():
                pairs[k] += v
    return pairs

def prepare_autoencoder_train_data(parameters, mask):
    tensor = torch.tensor(parameters, dtype=torch.float32)
    masked_data = tensor[mask]
    if len(masked_data) == 0:
        return None
    dataset = TensorDataset(masked_data)
    return DataLoader(dataset, batch_size=64, shuffle=True)

def is_well_explained(parameters, model, threshold=0.01):
    with torch.no_grad():
        recon = model(parameters)
        error = torch.max(torch.abs(recon - parameters), dim=1).values
        return error < threshold


# In[45]:


class Abstraction(Shape3D):
    def __init__(self, type_list, float_parameters, other_parameters, model):
        self.type_list = type_list
        self.float_parameters = float_parameters
        self.other_parameters = other_parameters
        self.model = model
        self.children = []  # empty to reduce node count

        # Expand immediately to store geometry for visualization
        try:
            expanded_shape = self.expand()
            self.boxes = expanded_shape.get_box3d_list()
        except Exception:
            self.boxes = []

    def __str__(self):
        s = "Abstraction(\n"
        for f in self.float_parameters:
            s += f"  {f}\n"
        for o in self.other_parameters:
            s += f"  {o}\n"
        s += ")"
        return s

    def expand(self):
        self.model.eval()
        float_tensor = torch.tensor(self.float_parameters, dtype=torch.float32).unsqueeze(0)
        decoder_output = self.model(float_tensor)
        expanded_float_parameters = decoder_output.squeeze(0).tolist()

        full_parameter_list = []
        i_floats = 0
        i_others = 0
        for t in self.type_list:
            if t == float:
                full_parameter_list.append(expanded_float_parameters[i_floats])
                i_floats += 1
            else:
                full_parameter_list.append(self.other_parameters[i_others])
                i_others += 1

        return instantiate_3d(self.type_list, full_parameter_list)

    def get_box3d_list(self):
        # Return the precomputed boxes
        return self.boxes


# In[46]:


# Cell 5: Train Autoencoders

def train_abstractions(structures, retrain_iterations=2, error_threshold=0.01):
    losses = defaultdict(list)
    models = {}

    for structure_name, parameters in structures.items():
        float_indices = [i for i, t in enumerate(parameters[0]) if isinstance(t, float)]
        if not float_indices:
            continue

        float_params = [[p[i] for i in float_indices] for p in parameters]
        float_tensor = torch.tensor(float_params, dtype=torch.float32)
        well_explained = torch.ones(len(float_tensor), dtype=torch.bool)

        for _ in range(retrain_iterations):
            train_loader = prepare_autoencoder_train_data(float_tensor, well_explained)
            if train_loader is None:
                print(f"Skipping {structure_name}: no well-explained samples")
                break

            model = nn.Sequential(
                nn.Linear(len(float_indices), 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, len(float_indices))
            )
            optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
            loss_fn = lambda recon, x: torch.mean(torch.max(torch.abs(recon - x), dim=1).values)

            model.train()
            for epoch in range(100):
                running_loss = 0.0
                for batch in train_loader:
                    x = batch[0]
                    optimizer.zero_grad()
                    recon = model(x)
                    loss = loss_fn(recon, x)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                losses[structure_name].append(running_loss / len(train_loader))

            model.eval()
            well_explained = is_well_explained(float_tensor, model, threshold=error_threshold)
            if torch.sum(well_explained) == 0:
                well_explained[0] = True

        models[structure_name] = model
    return models, losses


# In[47]:


# Cell 6: integrate_abstractions (new version)

def integrate_abstractions(shape_or_list, models, error_threshold=0.01):
    if isinstance(shape_or_list, list):
        return [integrate_abstractions(s, models, error_threshold) for s in shape_or_list]

    if shape_or_list is None or not isinstance(shape_or_list, Shape3D):
        return shape_or_list

    shape = shape_or_list
    try:
        cls, parameters = shape.param_tuple()
    except Exception:
        return shape

    type_str = cls.__name__
    float_params = [p for p in parameters if isinstance(p, float)]
    other_params = [p for p in parameters if not isinstance(p, float)]

    fits_well = False
    if float_params and type_str in models:
        float_tensor = torch.tensor([float_params], dtype=torch.float32)
        with torch.no_grad():
            recon = models[type_str](float_tensor)
            error = torch.max(torch.abs(recon - float_tensor))
            fits_well = error < error_threshold

    if fits_well:
        type_list = [cls] + [type(p) for p in parameters]
        new_shape = Abstraction(type_list, float_params, other_params, models[type_str])
        new_shape.children = []  # remove children to reduce nodes
        return new_shape
    else:
        new_shape = cls(*parameters)
        if hasattr(new_shape, 'children') and isinstance(new_shape.children, list):
            new_shape.children = [integrate_abstractions(c, models, error_threshold)
                                  for c in new_shape.children]
        return new_shape


# In[48]:


# Cell 7: Full Pipeline

def abstraction_pipeline(dataset, error_threshold=0.01):
    singletons = get_singletons(dataset)
    pairs = get_pairs(dataset)

    singleton_models, _ = train_abstractions(singletons, error_threshold=error_threshold)
    pair_models, _ = train_abstractions(pairs, error_threshold=error_threshold)

    dataset_singleton_abstracted = integrate_abstractions(dataset, singleton_models, error_threshold)
    dataset_fully_abstracted = integrate_abstractions(dataset_singleton_abstracted, pair_models, error_threshold)

    return dataset_fully_abstracted


# In[49]:


# Cell 8: Run Pipeline and Visualize

dataset = generate_dataset(50)
print(f"Generated {len(dataset)} shapes.")

abstracted_dataset = abstraction_pipeline(dataset)
print("Abstraction pipeline completed.")

# Count nodes before and after
def count_nodes(shape):
    if isinstance(shape, list):
        return sum(count_nodes(s) for s in shape)
    return 1 + sum(count_nodes(c) for c in getattr(shape, 'children', []))

total_before = count_nodes(dataset)
total_after = count_nodes(abstracted_dataset)
print("Total nodes before abstraction:", total_before)
print("Total nodes after abstraction:", total_after)

# # Visualize first 5 shapes
# for shape in abstracted_dataset[:5]:
#     boxes = shape.get_box3d_list()
#     visualize_boxes_3d(boxes)


# In[50]:


dataset[0]


# In[51]:


print(dataset[0])


# In[40]:


print(abstracted_dataset[0])


# In[52]:


visualize_boxes_3d(abstracted_dataset[0].get_box3d_list())


# In[54]:


visualize_boxes_3d(dataset[0].get_box3d_list())


# In[57]:


visualize_boxes_3d(abstracted_dataset[95].get_box3d_list())


# In[59]:


print(abstracted_dataset[95])


# In[ ]:




