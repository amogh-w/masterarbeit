"""
dsl_utils.py

Utilities for analyzing DSL object trees: recursively finding all subtrees and
collecting singleton and parent-child pair parameters with debug logging.
"""

from abstractionssymh.debug_utils import debug_info, debug_error, debug_success
from collections import defaultdict


def find_all_subtrees(node):
    """Recursively finds all subtrees of a DSL node.

    Args:
        node: A DSL tree node object. The node must implement a ``serialize()``
            method that returns a tuple of the form ``(class_name, (params, children))``.

    Yields:
        Each subtree node found in the DSL tree.
    """
    name, (params, children) = type(node).__name__, node.serialize()[1]
    for child in children:
        if hasattr(child, "serialize"):
            yield from find_all_subtrees(child)
    yield node


def collect_singleton_and_pair_data(dsl_shapes):
    """Collects singleton and parent-child pair parameter data from DSL shapes.

    Args:
        dsl_shapes: A list of DSL shape root nodes. Each node must implement a
            ``serialize()`` method returning ``(class_name, (params, children))``.

    Returns:
        A tuple containing:
        - dict: A mapping of node type name → list of parameter sets (singleton data),
            e.g., {'Box': [...], 'Scale': [...], 'Rotate': [...]}.
        - dict: A mapping of "Parent(Child)" → list of combined parameter sets (pair data),
            e.g., {'Scale(Box)': [...], 'Translate(Box)': [...]}.

    Side Effects:
        Logs debug messages summarizing the collected singleton and pair data.

    Example:
        >>> s_data, p_data = collect_singleton_and_pair_data([dsl_root])
        >>> print(s_data.keys())  # {'Box', 'Scale', 'Rotate', 'Translate', ...}
        >>> print(p_data.keys())  # {'Scale(Box)', 'Rotate(Scale)', 'Translate(Rotate)', ...}
    """
    if not dsl_shapes:
        debug_error("No DSL shapes provided for analysis.")
        return {}, {}

    s_data = defaultdict(list)
    p_data = defaultdict(list)

    for shape in dsl_shapes:
        for node in find_all_subtrees(shape):
            name, (p_params, children) = type(node).__name__, node.serialize()[1]

            if p_params:
                s_data[name].append(p_params)

            for child in (c for c in children if hasattr(c, "serialize")):
                c_name, (c_params, _) = type(child).__name__, child.serialize()[1]
                combo_params = p_params + c_params
                if combo_params:
                    pair_sig = f"{name}({c_name})"
                    p_data[pair_sig].append(combo_params)

    debug_success(
        "Collected keys:", f"{s_data.keys()} singletons, {p_data.keys()} pairs"
    )
    debug_success(
        "Collected parameters:", f"{len(s_data)} singletons, {len(p_data)} pairs"
    )

    return dict(sorted(s_data.items())), dict(sorted(p_data.items()))
