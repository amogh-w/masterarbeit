#!/usr/bin/env python
# coding: utf-8

# In[43]:


import sys
import os
import random
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Type, Union

sys.path.append(os.path.abspath("../src"))

from abstractions3d.primitives.shapes import Box3D
from abstractions3d.primitives.visualization import visualize_boxes_3d
from abstractions3d.dsl.core import Shape3D
from abstractions3d.dsl.nodes import Rect3D, Move3D, Union3D, SymTrans3D, SymRef3D
from abstractions3d.dsl.instantiation import instantiate_3d
from abstractions3d.data.blueprint import Table3D, Chair3D, Bench3D


# In[44]:


def sample_uniform(low: float, high: float) -> float:
    return random.uniform(low, high)

def generate_dataset(num_samples: int = 50) -> List[Shape3D]:
    dataset = []
    for _ in range(num_samples):
        table = Table3D(
            top_length=sample_uniform(3.0, 6.0),
            top_depth=sample_uniform(1.5, 3.0),
            top_thickness=sample_uniform(0.1, 0.3),
            leg_length=sample_uniform(0.2, 0.4),
            leg_depth=sample_uniform(0.2, 0.4),
            leg_height=sample_uniform(1.0, 2.0)
        )
        dataset.append(table)

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
    return dataset

dataset = generate_dataset()
print(f"Generated {len(dataset)} shapes")


# In[45]:


shape = random.choice(dataset)
print("Random shape from dataset:")
visualize_boxes_3d(shape.get_box3d_list())


# In[57]:


def add_dicts(d1: Dict, d2: Dict) -> Dict:
    for k in d2:
        d1[k].extend(d2[k])
    return d1

def get_singletons(shapes: Union[Shape3D, List[Shape3D]]) -> Dict[Tuple[str,int],List[Tuple[float,...]]]:
    singletons = defaultdict(list)
    if isinstance(shapes, list):
        for s in shapes:
            singletons = add_dicts(singletons,get_singletons(s))
        return singletons
    t, params = shapes.param_tuple()
    floats = tuple(p for p in params if isinstance(p,(float,int)))
    if floats:
        singletons[(t.__name__, len(floats))].append(floats)
    for c in getattr(shapes,'children',[]):
        singletons = add_dicts(singletons,get_singletons(c))
    return singletons

def get_pairs(shapes: Union[Shape3D, List[Shape3D]]) -> Dict[Tuple[str,int],List[Tuple[float,...]]]:
    pairs = defaultdict(list)
    if isinstance(shapes,list):
        for s in shapes:
            pairs = add_dicts(pairs,get_pairs(s))
        return pairs
    if isinstance(shapes, Union3D):
        t, (c1, c2) = shapes.param_tuple()
        p1 = tuple(p for p in c1.param_tuple()[1] if isinstance(p,(float,int)))
        p2 = tuple(p for p in c2.param_tuple()[1] if isinstance(p,(float,int)))
        key = (f"Union3D({type(c1).__name__},{type(c2).__name__})", len(p1)+len(p2))
        pairs[key].append(p1+p2)
        for c in shapes.children:
            pairs = add_dicts(pairs,get_pairs(c))
    else:
        for c in getattr(shapes,'children',[]):
            pairs = add_dicts(pairs,get_pairs(c))
    return pairs


# In[58]:


class AbstractionNode(Shape3D):
    def __init__(self, type_list: List[Type[Shape3D]], latent: torch.Tensor, model: Optional[nn.Module] = None):
        super().__init__(children=[])
        self.type_list = type_list
        self.latent = latent
        self.model = model

    def expand(self) -> Shape3D:
        # Replace AbstractionNode types with Shape3D placeholders
        safe_type_list = [Shape3D if t==AbstractionNode else t for t in self.type_list]
        if self.model is None:
            param_list = self.latent.tolist()
        else:
            self.model.eval()
            with torch.no_grad():
                # ensure latent matches decoder input
                if self.latent.numel() != self.model.decoder[0].in_features:
                    latent = self.latent[:self.model.decoder[0].in_features]
                else:
                    latent = self.latent
                param_list = self.model.decoder(latent[None,:])[0].tolist()
        return instantiate_3d(safe_type_list, param_list)

    def get_box3d_list(self) -> List[Box3D]:
        return self.expand().get_box3d_list()

    def __str__(self):
        return f"AbstractionNode(type_list={[t.__name__ for t in self.type_list]}, latent={self.latent.tolist()})"


# In[59]:


class Autoencoder(nn.Module):
    def __init__(self,input_dim:int,latent_dim:int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Linear(64,latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim,64), nn.ReLU(), nn.Linear(64,input_dim))
    def forward(self,x:torch.Tensor):
        z = self.encoder(x)
        return z,self.decoder(z)


# In[60]:


def find_abstractions(structures: Dict[Tuple[str,int],List[Tuple[float,...]]],
                      retrain_iterations: int = 10,
                      error_threshold: float = 0.05) -> Tuple[Dict, Dict]:
    models = {}
    losses = {}
    for key, params in structures.items():
        if not params: continue
        float_params = [list(p) for p in params if any(isinstance(x,(float,int)) for x in p)]
        if not float_params: continue
        num_float = len(float_params[0])
        if num_float < 2: continue

        train_tensor = torch.tensor(float_params,dtype=torch.float32)
        train_dl = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)
        model = Autoencoder(input_dim=num_float, latent_dim=max(1,num_float-1))
        optimizer = AdamW(model.parameters(), lr=1e-3)
        loss_fn = lambda pred,target: torch.mean(torch.max(torch.abs(pred-target),dim=-1)[0])

        model.train()
        losses[key] = []
        for _ in range(retrain_iterations):
            batch_losses = []
            for batch in train_dl:
                x = batch[0]
                optimizer.zero_grad()
                _, x_rec = model(x)
                loss = loss_fn(x_rec,x)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            losses[key].append(sum(batch_losses)/len(batch_losses))
        models[key] = model
        print(f"Trained {key}, final loss: {losses[key][-1]:.4f}")
    return models, losses


# In[62]:


def integrate_abstractions(shape: Shape3D, models: Dict, error_threshold: float = 0.05, stats: Optional[Dict] = None) -> Shape3D:
    if stats is None: stats = {'replaced':0}

    if isinstance(shape,Union3D):
        new_c1 = integrate_abstractions(shape.children[0], models, error_threshold, stats)
        new_c2 = integrate_abstractions(shape.children[1], models, error_threshold, stats)

        # Use only direct floats from children (no recursive recombination)
        float_params = [p for p in new_c1.param_tuple()[1] if isinstance(p,(float,int))] + \
                       [p for p in new_c2.param_tuple()[1] if isinstance(p,(float,int))]
        key = (f"Union3D({type(new_c1).__name__},{type(new_c2).__name__})", len(float_params))

        if key in models and float_params:
            model = models[key]
            input_tensor = torch.tensor(float_params)[None,:]
            with torch.no_grad():
                _, recon = model(input_tensor)
                error = torch.max(torch.abs(recon - input_tensor)).item()
            if error < error_threshold:
                stats['replaced'] += 1
                return AbstractionNode([Union3D,type(new_c1),type(new_c2)],
                                       torch.tensor(float_params,dtype=torch.float32),
                                       model)
        return Union3D(new_c1,new_c2)

    type_, params = shape.param_tuple()
    float_params = [p for p in params if isinstance(p,(float,int))]
    key = (type_.__name__, len(float_params))
    if key in models and float_params:
        model = models[key]
        input_tensor = torch.tensor(float_params)[None,:]
        with torch.no_grad():
            _, recon = model(input_tensor)
            error = torch.max(torch.abs(recon - input_tensor)).item()
        if error < error_threshold:
            stats['replaced'] += 1
            return AbstractionNode([type_], torch.tensor(float_params,dtype=torch.float32), model)

    # Return a fresh instance to avoid mutating original
    new_children = [integrate_abstractions(c, models, error_threshold, stats) for c in getattr(shape,'children',[])]
    return type_(*params) if not new_children else type_(*params)  # type_ must handle children internally


# In[63]:


singletons = get_singletons(dataset)
pairs = get_pairs(dataset)
structures = add_dicts(singletons,pairs)
print("Extracted structures:", structures.keys())

models, losses = find_abstractions(structures)

abstracted_dataset = []
stats_list = []
for shape in dataset:
    stats = {'replaced':0}
    abstracted = integrate_abstractions(shape, models, error_threshold=0.05, stats=stats)
    abstracted_dataset.append(abstracted)
    stats_list.append(stats)

total_replaced = sum(s['replaced'] for s in stats_list)
print(f"Total nodes replaced: {total_replaced}, avg per shape: {total_replaced/len(dataset):.2f}")

for i,s in enumerate(stats_list):
    if s['replaced']>0:
        print(f"Shape {i} contains {s['replaced']} abstraction nodes")


# In[ ]:




