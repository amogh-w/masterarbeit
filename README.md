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

# Steps

1. Copy the folders like `Bag`, `Bottle`, `Chair` etc from [github.com/FENGGENYU/PartNet_symh](github.com/FENGGENYU/PartNet_symh) to `PartNet_symh/dropbox` folder
2. Run the `PartNet_symh/dataset_to_json.ipynb` file to generate the json files in `PartNet_symh/dataset` folder
3. Copy the contents of `PartNet_symh/dataset` folder to `src/abstractionssymh/dataset` folder
4. Run `notebookssymh/abstraction_demo.ipynb`