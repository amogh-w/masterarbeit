#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath("../src"))


# In[4]:


from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from abstractions.dsl.abstraction import Abstraction
from abstractions.dsl.instantiation import instantiate
from abstractions.dsl.core import Shape
from abstractions.dsl.nodes import Rect, Move, SymTrans, SymRef, Union

from abstractions.primitives.visualization import show_boxes
from abstractions.data.toy import random_chairs_1, random_chairs_2, random_tables_1


# In[5]:


test_shape = instantiate([Union, Rect, float, float, Rect, float, float], [1, 2, 3, 4])
show_boxes(test_shape.get_box_list(), limits=(-5, 5))


# In[6]:


chair = random_tables_1(1)[0]

print(chair)

show_boxes(chair.get_box_list())


# In[7]:


# dataset = random_shapes(1024, max_nodes=32)
dataset = random_chairs_1(512) + random_chairs_2(512) + random_tables_1(512)


# In[8]:


# In[11]:


singletons = get_singletons(dataset)
pairs = get_pairs(dataset)
structures = add(singletons, pairs)

for key, instances in structures.items():
    print(key, len(instances), instances[0])


# In[12]:


plt.figure()
plt.scatter([s[0] for s in structures["Rect"]], [s[1] for s in structures["Rect"]])
plt.show()


# In[13]:


# for each type of structure, train a very small MLP with n inputs to predict the parameters
# if a low loss can be achieved with an n smaller than the number of parameters, then a new abstraction has been found


# In[14]:
# In[15]:
# In[18]:


# In[19]:


models, losses = find_abstractions(
    structures, retrain_iterations=4, error_threshold=0.01
)


# In[20]:


for structure_name, loss_list in losses.items():
    plt.figure()

    for i in range(len(loss_list)):
        plt.plot(loss_list[i], label=f"Iteration {i+1}")

    plt.legend()
    plt.title(structure_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

plt.show()


# In[21]:


for structure_name, model in models.items():
    # for now, only consider float parameters
    parameters = structures[structure_name]
    print(structure_name, parameters[0])
    valid_indices = [
        i for i in range(len(parameters[0])) if isinstance(parameters[0][i], float)
    ]
    num_float_parameters = len(valid_indices)

    if num_float_parameters <= 0:
        break

    float_parameters = [
        [p[valid_index] for p in parameters] for valid_index in valid_indices
    ]
    data = torch.tensor(float_parameters).swapaxes(0, 1)

    model.eval()
    encodings, reconstructions = model(data)
    error, _ = torch.max(torch.abs(reconstructions - data), dim=-1)
    well_explained = error < 0.01
    encodings = encodings.detach().cpu().numpy()
    reconstructions = reconstructions.detach().cpu().numpy()
    error = error.detach().cpu().numpy()
    well_explained = well_explained.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    if data.shape[1] == 2:
        colors = ["green" if w else "red" for w in well_explained]
        ax[0].scatter(data[:, 0], data[:, 1], label="data", c=colors)
        ax[0].scatter(
            reconstructions[:, 0],
            reconstructions[:, 1],
            c=encodings,
            label="reconstruction",
        )
        ax[0].legend()
        ax[0].set_title(structure_name)
        ax[0].set_xlabel("P0")
        ax[0].set_ylabel("P1")

    sorted_errors = np.sort(error)
    ax[1].plot(sorted_errors)
    ax[1].set_title(structure_name)
    ax[1].set_ylabel("Error")

    fig.show()


# In[22]:


# In[23]:


abs_dataset = integrate_abstractions(dataset, models, error_threshold=0.03)


# In[24]:


index = 512
print(dataset[index])
print("----------------------")
print(abs_dataset[index])

show_boxes(dataset[index].get_box_list())
show_boxes(abs_dataset[index].get_box_list())


# In[ ]:
