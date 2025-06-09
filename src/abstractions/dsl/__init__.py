"""
DSL (Domain-Specific Language) module for shape abstraction and composition.

This package provides the foundational components for defining, composing,
and instantiating abstract shape structures in a symbolic DSL. Shapes can be
instantiated into geometric representations for further analysis or rendering.

Modules exposed:
- `Abstraction`: Abstraction logic for shape grouping or hierarchy.
- `Shape`: Base class for all symbolic shapes.
- `left_pad`: Utility for formatting multi-line strings.
- `instantiate`: Function to recursively build shape structures from type and parameter lists.
- Shape nodes: `Rect`, `Move`, `Union`, `SymTrans`, and `SymRef`, which represent DSL expressions.
"""

from .abstraction import Abstraction
from .core import Shape, left_pad
from .instantiation import instantiate
from .nodes import Rect, Move, Union, SymTrans, SymRef
