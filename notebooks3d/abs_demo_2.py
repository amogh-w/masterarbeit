#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath("../src"))


# In[9]:


import random
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from typing import List, Tuple, Type

from abstractions3d.primitives.shapes import Box3D
from abstractions3d.primitives.visualization import visualize_boxes_3d
from abstractions3d.dsl.core import Shape3D
from abstractions3d.dsl.nodes import Rect3D, Move3D, Union3D, SymTrans3D, SymRef3D
from abstractions3d.dsl.instantiation import instantiate_3d
from abstractions3d.data.blueprint import Table3D, Chair3D, Bench3D


# In[10]:


def sample_uniform(low, high):
    return random.uniform(low, high)

dataset = []

num_samples = 10  # per shape type

for _ in range(num_samples):
    # Tables
    table = Table3D(
        top_length=sample_uniform(3.0, 6.0),
        top_depth=sample_uniform(1.5, 3.0),
        top_thickness=sample_uniform(0.1, 0.3),
        leg_length=sample_uniform(0.2, 0.4),
        leg_depth=sample_uniform(0.2, 0.4),
        leg_height=sample_uniform(1.0, 2.0)
    )
    dataset.append(table)

    # Chairs
    chair = Chair3D(
        seat_length=sample_uniform(1.0, 2.0),
        seat_depth=sample_uniform(1.0, 2.0),
        seat_thickness=sample_uniform(0.2, 0.5),
        leg_length=sample_uniform(0.1, 0.3),
        leg_depth=sample_uniform(0.1, 0.3),
        leg_height=sample_uniform(0.5, 1.0),
        backrest_height=sample_uniform(1.0, 2.0),
        backrest_thickness=sample_uniform(0.1, 0.3)
    )
    dataset.append(chair)

    # Benches
    bench = Bench3D(
        seat_length=sample_uniform(2.0, 5.0),
        seat_depth=sample_uniform(0.5, 1.0),
        seat_thickness=sample_uniform(0.2, 0.5),
        leg_length=sample_uniform(0.1, 0.3),
        leg_depth=sample_uniform(0.1, 0.3),
        leg_height=sample_uniform(0.5, 1.0),
        backrest_height=sample_uniform(0.5, 1.5),
        backrest_thickness=sample_uniform(0.1, 0.3)
    )
    dataset.append(bench)

print(f"Generated {len(dataset)} 3D shapes")


# In[11]:


shape = random.choice(dataset)
boxes = shape.get_box3d_list()
visualize_boxes_3d(boxes)


# In[12]:


def add(d1, d2):
    for k in d2:
        d1[k] += d2[k]
    return d1

def get_singletons(shapes):
    singletons = defaultdict(list)
    if isinstance(shapes, list):
        for s in shapes:
            singletons = add(singletons, get_singletons(s))
        return singletons

    t, params = shapes.param_tuple()
    floats = tuple(p for p in params if isinstance(p,(float,int)))
    if floats:
        singletons[t.__name__].append(floats)

    for c in getattr(shapes, 'children', []):
        singletons = add(singletons, get_singletons(c))
    return singletons

def get_pairs(shapes):
    pairs = defaultdict(list)
    if isinstance(shapes, list):
        for s in shapes:
            pairs = add(pairs, get_pairs(s))
        return pairs

    if isinstance(shapes, Union3D):
        t, (c1, c2) = shapes.param_tuple()
        t1, p1 = c1.param_tuple()
        t2, p2 = c2.param_tuple()
        key = f"{t.__name__}({t1.__name__},{t2.__name__})"
        pairs[key].append(tuple([p for p in p1 if isinstance(p,(float,int))] +
                                [p for p in p2 if isinstance(p,(float,int))]))
        for c in shapes.children:
            pairs = add(pairs, get_pairs(c))
    else:
        for c in getattr(shapes, 'children', []):
            pairs = add(pairs, get_pairs(c))
    return pairs


# In[13]:


singletons = defaultdict(list)
pairs = defaultdict(list)

for shape in dataset:
    singletons = add(singletons, get_singletons(shape))
    pairs = add(pairs, get_pairs(shape))

structures = add(singletons, pairs)
print("Extracted structure types:", structures.keys())


# In[31]:


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


# In[32]:


def find_abstractions(structures, retrain_iterations=2, error_threshold=0.01):
    models = {}
    for name, parameters in structures.items():
        float_indices = [i for i, p in enumerate(parameters[0]) if isinstance(p, float)]
        if len(float_indices) == 0:
            continue

        float_params = [[p[i] for i in float_indices] for p in parameters]
        num_float_parameters = len(float_indices)

        for _ in range(retrain_iterations):
            train_tensor = torch.tensor(float_params, dtype=torch.float32)
            train_dl = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)

            model = Autoencoder(input_dim=num_float_parameters, latent_dim=max(1,num_float_parameters-1))
            optimizer = AdamW(model.parameters(), lr=1e-3)
            loss_fn = lambda pred, target: torch.mean(torch.max(torch.abs(pred-target), dim=-1)[0])

            model.train()
            for batch in train_dl:
                x = batch[0]
                optimizer.zero_grad()
                _, x_rec = model(x)
                loss = loss_fn(x_rec, x)
                loss.backward()
                optimizer.step()

        models[name] = model
        print(f"Trained model for {name} with input_dim={num_float_parameters}")
    return models


# In[33]:


from abstractions3d.dsl.core import Shape3D
from abstractions3d.dsl.instantiation import instantiate_3d

class AbstractionNode(Shape3D):
    def __init__(self, type_list: list, latent: torch.Tensor, model: nn.Module):
        super().__init__(children=[])
        self.type_list = type_list
        self.latent = latent
        self.model = model

    def expand(self) -> Shape3D:
        if self.model is None:
            param_list = self.latent.tolist()
        else:
            self.model.eval()
            with torch.no_grad():
                param_list = self.model.decoder(self.latent[None,:])[0].tolist()
        return instantiate_3d(self.type_list, param_list)

    def get_box3d_list(self):
        return self.expand().get_box3d_list()

    @classmethod
    def from_shape(cls, shape: Shape3D, models: dict):
        type_list, numeric_params = [], []

        def collect(s):
            type_list.append(type(s))
            for p in s.param_tuple()[1]:
                if isinstance(p,(float,int)):
                    numeric_params.append(p)
            for c in getattr(s, 'children', []):
                collect(c)

        collect(shape)
        model = models.get(type(shape).__name__)
        latent = torch.tensor(numeric_params, dtype=torch.float32)
        return cls(type_list, latent, model)


# In[35]:


def integrate_abstractions_with_stats(shape: Shape3D, models: dict, error_threshold: float = 0.01, stats=None):
    """
    Recursively replaces sub-shapes with AbstractionNode if well explained by a model.
    Tracks the number of replaced nodes.
    """
    if stats is None:
        stats = {'replaced': 0}

    # Composite shapes
    if isinstance(shape, Union3D):
        # Recursively process children
        new_child1 = integrate_abstractions_with_stats(shape.children[0], models, error_threshold, stats)
        new_child2 = integrate_abstractions_with_stats(shape.children[1], models, error_threshold, stats)
        type_str = f"Union3D({type(new_child1).__name__},{type(new_child2).__name__})"

        # Collect numeric parameters
        params1 = [p for p in new_child1.param_tuple()[1] if isinstance(p,(float,int))]
        params2 = [p for p in new_child2.param_tuple()[1] if isinstance(p,(float,int))]
        float_params = params1 + params2

        if type_str in models and len(float_params) > 0:
            model = models[type_str]
            input_tensor = torch.tensor(float_params)[None,:]
            with torch.no_grad():
                _, recon = model(input_tensor)
                error,_ = torch.max(torch.abs(recon - input_tensor), dim=-1)
            if error.item() < error_threshold:
                type_list = [Union3D, type(new_child1), type(new_child2)]
                latent = torch.tensor(float_params, dtype=torch.float32)
                stats['replaced'] += 1
                return AbstractionNode(type_list, latent, model)

        return Union3D(new_child1, new_child2)

    # Singleton shapes
    type_, params = shape.param_tuple()
    float_params = [p for p in params if isinstance(p,(float,int))]
    type_str = type_.__name__

    if type_str in models and len(float_params) > 0:
        model = models[type_str]
        input_tensor = torch.tensor(float_params)[None,:]
        with torch.no_grad():
            _, recon = model(input_tensor)
            error,_ = torch.max(torch.abs(recon - input_tensor), dim=-1)
        if error.item() < error_threshold:
            type_list = [type_]
            latent = torch.tensor(float_params, dtype=torch.float32)
            stats['replaced'] += 1
            return AbstractionNode(type_list, latent, model)

    # Process children recursively
    new_children = [integrate_abstractions_with_stats(c, models, error_threshold, stats) for c in getattr(shape, 'children', [])]
    shape.children = new_children
    return shape


# In[36]:


abstracted_dataset = []
stats_list = []

for shape in dataset:
    stats = {'replaced': 0}
    abstracted_shape = integrate_abstractions_with_stats(shape, models, error_threshold=0.01, stats=stats)
    abstracted_dataset.append(abstracted_shape)
    stats_list.append(stats)

# Summary
total_shapes = len(dataset)
total_replaced_nodes = sum(s['replaced'] for s in stats_list)
print(f"Total shapes: {total_shapes}")
print(f"Total abstracted nodes across dataset: {total_replaced_nodes}")

# Optional per-shape statistics
for i, s in enumerate(stats_list[:5]):
    print(f"Shape {i} replaced nodes: {s['replaced']}")


# In[ ]:




