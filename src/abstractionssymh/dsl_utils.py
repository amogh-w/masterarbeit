"""
dsl_utils.py

Utilities for analyzing DSL object trees: recursively finding all subtrees and
collecting singleton and parent-child pair parameters with debug logging.

*** UPDATED to support Abstraction nodes for L2+ analysis ***
"""

from abstractionssymh.debug_utils import debug_info, debug_error, debug_success
from collections import defaultdict

# --- IMPORTANT: Must import Abstraction to check for it ---
try:
    # This import is necessary for the functions to recognize Abstraction nodes
    from abstractionssymh.abstraction_utils import Abstraction
except ImportError:
    debug_error("Could not import Abstraction node. L2+ analysis will fail.")
    # Define a dummy class to avoid NameErrors if something loads out of order
    class Abstraction:
        pass

def find_all_subtrees(node):
    """
    Recursively finds all subtrees of a DSL node.
    *** UPDATED to handle Abstraction nodes. ***
    
    Args:
        node: A DSL tree node object (e.g., Box, Scale, or Abstraction).

    Yields:
        Each subtree node found in the DSL tree.
    """
    
    children = []
    if isinstance(node, Abstraction):
        # Handle Abstraction node
        children = node.children
    elif hasattr(node, "serialize"):
        # Handle original DSL node
        _, (_, children_from_serialize) = node.serialize()
        # Filter children to only include node-like objects
        children = [c for c in children_from_serialize if hasattr(c, "serialize") or isinstance(c, Abstraction)]
    else:
        # Not a node we can process (e.g., a primitive value like 'n_fold')
        return

    # Recurse into children first (post-order traversal)
    for child in children:
        yield from find_all_subtrees(child)
    
    # Yield the parent node itself after its children
    yield node


def collect_singleton_and_pair_data(dsl_shapes):
    """
    Collects singleton and parent-child pair parameter data from DSL shapes.
    *** UPDATED to handle Abstraction nodes for L2+ analysis. ***

    Args:
        dsl_shapes: A list of DSL shape root nodes.

    Returns:
        A tuple containing (singleton_data, pair_data)
    """
    if not dsl_shapes:
        debug_error("No DSL shapes provided for analysis.")
        return {}, {}

    s_data = defaultdict(list)
    p_data = defaultdict(list)

    for shape in dsl_shapes:
        for node in find_all_subtrees(shape): # This now yields Abstraction nodes
            
            p_params, children = [], []
            name = ""
            
            # --- Get parent node info ---
            if isinstance(node, Abstraction):
                name = f"Abs({node.pattern_name})" # Special name for abstractions
                p_params = node.compressed_params
                children = node.children
            elif hasattr(node, "serialize"):
                name, (p_params, children_from_serialize) = type(node).__name__, node.serialize()[1]
                # Filter children to only include node-like objects
                children = [c for c in children_from_serialize if hasattr(c, "serialize") or isinstance(c, Abstraction)]
            else:
                continue # Should be filtered by find_all_subtrees, but as a safeguard

            # --- Store Singleton Data ---
            # Ensure p_params is a list or tuple before storing
            if p_params is not None and not isinstance(p_params, (list, tuple)):
                p_params = [p_params] # Wrap single values in a list
            
            if p_params:
                s_data[name].append(p_params)

            # --- Store Pair Data ---
            for child in children:
                c_params = []
                c_name = ""
                
                if isinstance(child, Abstraction):
                    c_name = f"Abs({child.pattern_name})"
                    c_params = child.compressed_params
                elif hasattr(child, "serialize"):
                    c_name, (c_params, _) = type(child).__name__, child.serialize()[1]
                else:
                    continue # Not a node (e.g., an int parameter)
                
                # Ensure c_params is a list or tuple
                if c_params is not None and not isinstance(c_params, (list, tuple)):
                    c_params = [c_params]

                # Use list() to handle potential empty lists/None correctly
                combo_params = list(p_params or []) + list(c_params or [])
                
                if combo_params:
                    pair_sig = f"{name}({c_name})"
                    p_data[pair_sig].append(combo_params)

    # debug_success(
    #     f"Collected keys: {list(s_data.keys())} singletons, {list(p_data.keys())} pairs"
    # )
    # debug_success(
    #     f"Collected parameters: {len(s_data)} singletons, {len(p_data)} pairs"
    # )

    return dict(sorted(s_data.items())), dict(sorted(p_data.items()))