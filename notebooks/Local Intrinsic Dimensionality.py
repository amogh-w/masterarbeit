#!/usr/bin/env python
# coding: utf-8

# # Imports and Setup

# In[11]:


import sys
import os
from collections import defaultdict

# Add the src directory to sys.path
sys.path.append(os.path.abspath("../src"))

import torch
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, Markdown

from abstractions.dsl.abstraction import Abstraction
from abstractions.dsl.core import Shape
from abstractions.data.generator import generate_dataset
from abstractions.learning.utils import add, get_singletons, get_pairs
from abstractions.primitives.visualization import show_boxes, print_tree
from abstractions.learning.abstraction_discovery import find_abstractions, integrate_abstractions
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import hdbscan # Make sure you have this installed: pip install hdbscan
from sklearn.neighbors import NearestNeighbors


# # Utility and Helper Functions

# In[12]:


def count_nodes(shape):
    """Recursively counts the number of nodes in a shape program."""
    if not hasattr(shape, "param_tuple"):
        return 1
    _, args = shape.param_tuple()
    return 1 + sum(count_nodes(arg) for arg in args)

def contains_abstraction(shape):
    """Recursively checks if a shape program contains an Abstraction node."""
    if isinstance(shape, Abstraction):
        return True
    if not hasattr(shape, 'param_tuple'):
        return False
    _, args = shape.param_tuple()
    return any(contains_abstraction(arg) for arg in args)

def extract_float_params(param_list):
    """Extracts only float or integer parameters from a list of parameter tuples."""
    return [
        [p for p in tup if isinstance(p, (float, int))]
        for tup in param_list
    ]

def make_df_from_structure(param_list):
    """Converts a list of parameters into a Pandas DataFrame with float columns."""
    float_data = extract_float_params(param_list)
    float_data = [row for row in float_data if len(row) > 0]
    if not float_data:
        return pd.DataFrame()

    df = pd.DataFrame(float_data)
    df.columns = [f"param_{i}" for i in range(df.shape[1])]
    return df

# Helper for Autoencoder
def extract_floats(params):
    """Extracts only float or integer parameters from a list of parameter tuples for Autoencoder."""
    return [
        tuple(p for p in t if isinstance(p, (float, int)))
        for t in params
        if any(isinstance(p, (float, int)) for p in t)
    ]


# # Reusable Visualization Widgets

# In[13]:


def visualize_dataset_interactive(dataset):
    """Creates an interactive slider to view shapes in a dataset."""
    output = widgets.Output()
    index_slider = widgets.IntSlider(
        value=0, min=0, max=len(dataset) - 1, step=1,
        description='Shape Index:', continuous_update=False
    )

    def update_display(change):
        idx = change["new"]
        output.clear_output()
        with output:
            shape = dataset[idx]
            display(Markdown(f"### Shape #{idx}"))
            print_tree(shape)
            show_boxes(shape.get_box_list(), backend="plotly")

    index_slider.observe(update_display, names="value")
    display(Markdown("### Visualize Dataset"), index_slider, output)
    update_display({"new": 0}) # Initial display

def create_structure_scatterplot_widget(structures, title):
    """Creates a dropdown to generate scatter plots for different structures."""
    plot_dropdown = widgets.Dropdown(
        options=list(structures.keys()),
        description="Structure:",
        layout=widgets.Layout(width="60%"),
    )
    plot_output = widgets.Output()

    def plot_on_change(change):
        structure_name = change["new"]
        plot_output.clear_output()
        with plot_output:
            param_list = structures[structure_name]
            df = make_df_from_structure(param_list)
            display(Markdown(f"### `{structure_name}` – {len(df)} instances"))
            if df.empty or df.shape[1] == 0:
                print("No float parameters to visualize.")
                return

            fig = px.scatter(
                df, x="param_0", y="param_1" if df.shape[1] > 1 else None,
                title=f"Scatterplot of parameters for {structure_name}",
                hover_data=df.columns
            ) if df.shape[1] > 1 else px.histogram(df, x="param_0", nbins=50, title=f"Histogram for {structure_name}")
            fig.show()

    plot_dropdown.observe(plot_on_change, names="value")
    display(Markdown(f"### {title}"), plot_dropdown, plot_output)
    if list(structures.keys()):
        plot_dropdown.value = list(structures.keys())[0]

def create_reconstruction_plot_widget(models, structures, threshold):
    """Creates a widget to visualize model reconstruction error."""
    dropdown = widgets.Dropdown(options=list(models.keys()), description="Structure:")
    output = widgets.Output()

    def update_plot(change):
        name = change["new"]
        output.clear_output()
        with output:
            model = models[name]
            param_list = structures[name]
            df = make_df_from_structure(param_list)
            if df.empty:
                print("No float parameters to plot.")
                return

            data = torch.tensor(df.values, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                _, recon = model(data)
                error = torch.max(torch.abs(recon - data), dim=1)[0]
                df["well_explained"] = error < threshold

            fig = px.scatter(
                df, x="param_0", y="param_1", color="well_explained",
                color_discrete_map={True: "green", False: "red"},
                title=f"{name} — Reconstruction Quality (Error < {threshold})"
            )
            fig.show()

    dropdown.observe(update_plot, names="value")
    display(Markdown("### Well-Defined Plots (Reconstruction Quality)"), dropdown, output)
    if list(models.keys()):
        update_plot({"new": dropdown.value})

def create_comparison_widget(original_dataset, abstracted_dataset):
    """Creates a widget to compare original and abstracted shapes."""
    num_before = sum(count_nodes(s) for s in original_dataset)
    num_after = sum(count_nodes(s) for s in abstracted_dataset)
    abstracted_indices = [i for i, s in enumerate(abstracted_dataset) if contains_abstraction(s)]

    summary = f"Total nodes reduced from {num_before} to {num_after}. "
    summary += f"{len(abstracted_indices)} / {len(original_dataset)} shapes were abstracted."
    display(Markdown(f"### Before and After Abstraction\n**Summary:** {summary}"))

    options = [
        (f"Index {i} | Δ={count_nodes(original_dataset[i]) - count_nodes(s)}", i)
        for i, s in enumerate(abstracted_dataset) if i in abstracted_indices
    ]
    if not options:
        print("No shapes were successfully abstracted.")
        return

    dropdown = widgets.Dropdown(options=options, description="Shape:")
    output = widgets.Output()

    def show_comparison(idx):
        output.clear_output()
        with output:
            before, after = original_dataset[idx], abstracted_dataset[idx]
            display(Markdown(f"#### Comparing Shape Index: {idx}"))
            print(f"Node Count: {count_nodes(before)} → {count_nodes(after)}")

            print("\nOriginal DSL Tree:")
            print(before)
            print("\nAbstracted DSL Tree:")
            print(after)

            display(Markdown("##### Original Geometry"))
            show_boxes(before.get_box_list(), backend="plotly")
            display(Markdown("##### Abstracted Geometry"))
            show_boxes(after.get_box_list(), backend="plotly")

    dropdown.observe(lambda change: show_comparison(change["new"]), names="value")
    display(dropdown, output)
    show_comparison(options[0][1])


# # Phase 1 - Initial Abstraction

# In[14]:


# 1.1. Generate and Visualize Dataset
print("--- PHASE 1: INITIAL ABSTRACTION ---")
dataset = (
    generate_dataset("chair_1", 1000) +
    generate_dataset("chair_2", 1000) +
    generate_dataset("chair_3", 1000)
)
visualize_dataset_interactive(dataset)

# 1.2. Generate Structures and Visualize Parameters
print("\nGenerating structures (singletons and pairs)...")
singletons = get_singletons(dataset)
pairs = get_pairs(dataset)
structures = add(singletons, pairs)
print(f"Found {len(structures)} unique structures.")
create_structure_scatterplot_widget(structures, "Initial Parameter Distribution")


# # Explore 'Move' Parameters and Basic Clustering (KMeans & DBSCAN)

# In[15]:


# Extract float-only params for 'Move' and display/plot
move_params = structures["Move"]
float_data = extract_float_params(move_params)
float_data = [row for row in float_data if len(row) > 0]
df = pd.DataFrame(float_data)
df.columns = [f"param_{i}" for i in range(df.shape[1])]

display(Markdown(f"### Move – {len(df)} instances"))

if df.shape[1] == 1:
    fig = px.histogram(df, x="param_0", nbins=50, title="Histogram of param_0 for Move")
else:
    fig = px.scatter(
        df, x="param_0", y="param_1",
        title="Scatterplot of Move Parameters (param_0 vs param_1)",
        hover_data=df.columns
    )
fig.show()


# ## KMeans clustering

# In[16]:


display(Markdown(f"### Move – {len(df)} instances – KMeans (k=4) Clusters"))
X_scaled_kmeans = StandardScaler().fit_transform(df)
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10) # n_init added for KMeans stability
labels_kmeans = kmeans.fit_predict(X_scaled_kmeans)
df["cluster_kmeans"] = labels_kmeans

fig_kmeans = px.scatter(
    df, x="param_0", y="param_1" if "param_1" in df.columns else "param_0",
    color="cluster_kmeans",
    title="KMeans Clustering (k=4) of Move Parameters",
    hover_data=df.columns
)
fig_kmeans.show()


# ## DBSCAN clustering (eps=0.5)

# In[18]:


display(Markdown(f"### Move – {len(df)} instances – DBSCAN Clusters (eps=0.5)"))
X_scaled_dbscan_05 = StandardScaler().fit_transform(df.drop(columns=["cluster_kmeans"], errors='ignore'))
dbscan_05 = DBSCAN(eps=0.5, min_samples=10)
labels_dbscan_05 = dbscan_05.fit_predict(X_scaled_dbscan_05)
df["cluster_dbscan_05"] = labels_dbscan_05

n_clusters_dbscan_05 = len(set(labels_dbscan_05)) - (1 if -1 in labels_dbscan_05 else 0)
display(Markdown(f"**Clusters found:** {n_clusters_dbscan_05} (Noise = {sum(labels_dbscan_05 == -1)})"))

fig_dbscan_05 = px.scatter(
    df, x="param_0", y="param_1" if "param_1" in df.columns else "param_0",
    color="cluster_dbscan_05",
    title="DBSCAN Clustering of Move Parameters (eps=0.5)",
    hover_data=df.columns,
    color_continuous_scale="Viridis"
)
fig_dbscan_05.show()


# ## DBSCAN clustering (eps=0.2) - more stringent

# In[19]:


display(Markdown(f"### Move – {len(df)} instances – DBSCAN Clusters (eps=0.2)"))
X_scaled_dbscan_02 = StandardScaler().fit_transform(df.drop(columns=["cluster_kmeans", "cluster_dbscan_05"], errors='ignore'))
dbscan_02 = DBSCAN(eps=0.2, min_samples=10)
labels_dbscan_02 = dbscan_02.fit_predict(X_scaled_dbscan_02)
df["cluster_dbscan_02"] = labels_dbscan_02

n_clusters_dbscan_02 = len(set(labels_dbscan_02)) - (1 if -1 in labels_dbscan_02 else 0)
display(Markdown(f"**Clusters found:** {n_clusters_dbscan_02} (Noise = {sum(labels_dbscan_02 == -1)})"))

fig_dbscan_02 = px.scatter(
    df, x="param_0", y="param_1" if "param_1" in df.columns else "param_0",
    color="cluster_dbscan_02",
    title="DBSCAN Clustering of Move Parameters (eps=0.2)",
    hover_data=df.columns,
    color_continuous_scale="Viridis"
)
fig_dbscan_02.show()


# In[28]:


from sklearn.neighbors import NearestNeighbors
import numpy as np

def estimate_local_id(X, r1=0.1, r2=0.9):
    nbrs = NearestNeighbors(radius=r2).fit(X)
    distances, _ = nbrs.radius_neighbors(X, return_distance=True)

    lids = np.zeros(len(X))
    log_r_ratio = np.log(r2 / r1)

    for i, dist in enumerate(distances):
        k1 = np.sum(dist < r1)
        k2 = np.sum(dist < r2)
        if k1 > 0 and k2 > k1:
            lids[i] = (np.log(k2) - np.log(k1)) / log_r_ratio
    return lids

# Apply to the same scaled data
df["local_id_manual"] = estimate_local_id(X_scaled_dbscan_02, r1=0.1, r2=0.9)

# Visualize Local ID
import plotly.express as px
fig_lid = px.scatter(
    df,
    x="param_0",
    y="param_1" if "param_1" in df.columns else "param_0",
    color="local_id_manual",
    title="Local Intrinsic Dimensionality (Manual Estimate)",
    color_continuous_scale="Plasma",
    hover_data=["cluster_dbscan_02"]
)
fig_lid.show()


# In[43]:


import plotly.express as px

# Filter vertical cluster where param_0 == 0
vertical_cluster = df[df["param_0"] == 0].copy()

# Sanity check: print how many points we have
print(f"Vertical cluster contains {len(vertical_cluster)} points.")

# Sort for better visual order (optional)
vertical_cluster = vertical_cluster.sort_values("param_1")

# Scatter plot of vertical cluster with Local ID
fig_vertical = px.scatter(
    vertical_cluster,
    x="param_0",
    y="param_1",
    color="local_id_manual",
    title="Vertical Cluster (param_0 = 0) Colored by Local ID",
    color_continuous_scale="Plasma",
    hover_data=vertical_cluster.columns
)

fig_vertical.show()


# In[44]:


import numpy as np
import plotly.express as px

# Extract vertical cluster
vertical_cluster = df[df["param_0"] == 0].copy()

# Sort by param_1 so we can visualize it smoothly
vertical_cluster = vertical_cluster.sort_values("param_1").reset_index(drop=True)

# Create a synthetic y-axis for visual spacing (e.g., 0, 1, 2, ...)
vertical_cluster["visual_y"] = np.arange(len(vertical_cluster))

# Plot param_1 on X-axis and visual_y on Y
fig_horizontal_line = px.scatter(
    vertical_cluster,
    x="param_1",             # original data value (was Y)
    y="visual_y",            # artificial Y to spread vertically
    color="local_id_manual",
    title="Flattened View of Vertical Cluster (Colored by Local ID)",
    labels={"param_1": "param_1 (was Y)", "visual_y": "Visual Index"},
    color_continuous_scale="Plasma",
    hover_data=["local_id_manual", "param_1", "cluster_dbscan_02"]
)

fig_horizontal_line.update_layout(yaxis_title="Visual Order (index)", xaxis_title="param_1 value")
fig_horizontal_line.show()


# In[29]:


import hdbscan

hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels_hdb = hdb.fit_predict(X_scaled_dbscan_02)
df["cluster_hdbscan"] = labels_hdb

n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
display(Markdown(f"### HDBSCAN Clustering"))
display(Markdown(f"**Clusters found:** {n_clusters_hdb} (Noise = {sum(labels_hdb == -1)})"))

fig_hdb = px.scatter(
    df,
    x="param_0",
    y="param_1" if "param_1" in df.columns else "param_0",
    color="cluster_hdbscan",
    title="HDBSCAN Clustering of Move Parameters",
    hover_data=["local_id_manual"],
    color_continuous_scale="Viridis"
)
fig_hdb.show()


# In[30]:


from skdim.id import lPCA

lpca = lPCA()
lid_lpca = lpca.fit_transform_pw(X_scaled_dbscan_02, n_neighbors=10)
df["local_id_lpca"] = lid_lpca

fig_lpca = px.scatter(
    df,
    x="param_0",
    y="param_1" if "param_1" in df.columns else "param_0",
    color="local_id_lpca",
    title="Local Intrinsic Dimensionality (skdim.lPCA)",
    color_continuous_scale="Magma",
    hover_data=["cluster_hdbscan"]
)
fig_lpca.show()


# In[35]:


from skdim.id import MLE

mle = MLE()
lid_mle = mle.fit_transform_pw(X_scaled_dbscan_02, n_neighbors=10)
df["local_id_mle"] = lid_mle

fig_mle = px.scatter(
    df,
    x="param_0",
    y="param_1" if "param_1" in df.columns else "param_0",
    color="local_id_mle",
    title="Local Intrinsic Dimensionality (skdim.MLE)",
    color_continuous_scale="Magma",
    hover_data=["cluster_hdbscan"]
)
fig_mle.show()


# In[37]:


from skdim.id import FisherS

fisher = FisherS()
lid_fisher = fisher.fit_transform_pw(X_scaled_dbscan_02, n_neighbors=10)
df["local_id_fishers"] = lid_fisher

fig_fisher = px.scatter(
    df,
    x="param_0",
    y="param_1" if "param_1" in df.columns else "param_0",
    color="local_id_fishers",
    title="Local Intrinsic Dimensionality (skdim.FisherS)",
    color_continuous_scale="Magma",
    hover_data=["cluster_hdbscan"]
)
fig_fisher.show()


# In[38]:


from skdim.id import CorrInt

corrint = CorrInt()
lid_corrint = corrint.fit_transform_pw(X_scaled_dbscan_02, n_neighbors=10)
df["local_id_corrint"] = lid_corrint

fig_corrint = px.scatter(
    df,
    x="param_0",
    y="param_1" if "param_1" in df.columns else "param_0",
    color="local_id_corrint",
    title="Local Intrinsic Dimensionality (skdim.CorrInt)",
    color_continuous_scale="Magma",
    hover_data=["cluster_hdbscan"]
)
fig_corrint.show()

