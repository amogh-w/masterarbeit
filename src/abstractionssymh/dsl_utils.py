"""
dsl_utils.py

Utilities for analyzing DSL object trees: recursively finding all subtrees and collecting singleton and parent-child pair parameters with debug logging.
"""

from abstractionssymh.debug_utils import debug_info, debug_error, debug_success
from collections import defaultdict


def find_all_subtrees(node):
    name, (params, children) = type(node).__name__, node.serialize()[1]
    for child in children:
        if hasattr(child, "serialize"):
            yield from find_all_subtrees(child)
    yield node


def collect_singleton_and_pair_data(dsl_shapes):
    if not dsl_shapes:
        debug_error("No DSL shapes provided for analysis.")
        return {}, {}

    s_data, p_data = defaultdict(list), defaultdict(list)

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
