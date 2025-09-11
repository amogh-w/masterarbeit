# Abstraction Discovery

# Setup

```
conda create -n abs python=3.12
conda activate abs
pip install -r requirements.txt
```

# Folder Structure

```
project-root/
├── notebooks2d/              # Experiments on 2D generated dataset
├── notebooks3d/              # Experiments on 3D generated dataset
├── notebookssymh/            # Experiments on PartNet_symh dataset
├── PartNet_symh/             # From github.com/FENGGENYU/PartNet_symh
│   ├── dataset/              # Dataset in JSON format
│   ├── dropbox/              # Dataset in original format
│   └── loaders/              # Parsers for `.mat`, `.obb`, `.obj` files
└── src/                      # Code for all experiments
    ├── abstractions/         # Abstractions for 2D generated dataset
    ├── abstractions3d/       # Abstractions for 3D generated dataset
    └── abstractionssymh/     # Abstractions for PartNet_symh dataset

```