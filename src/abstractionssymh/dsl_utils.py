"""dsl_utils.py

Utilities for analyzing DSL object trees: recursively finding all subtrees and
collecting singleton and parent-child pair parameters with debug logging.

This module is specifically updated to support `Abstraction` nodes, allowing
for "L2+" (Level 2 or higher) analysis of DSL trees that have been
partially abstracted.
"""

from abstractionssymh.debug_utils import debug_info, debug_error, debug_success
from collections import defaultdict

# --- IMPORTANT: Must import Abstraction to check for it ---
try:
    # This import is necessary for the functions to recognize Abstraction nodes
    from abstractionssymh.abstraction_utils import Abstraction
except ImportError:
    debug_error("Could not import Abstraction node. L2+ analysis will fail.")
    # Define a dummy class to avoid NameErrors if something loads out of order.
    # This allows the module to be imported, but analysis will fail
    # if Abstraction nodes are actually encountered.
    class Abstraction:
        """Dummy class to prevent NameErrors if import fails."""
        pass

def find_all_subtrees(node):
    """Recursively find all subtrees of a DSL node via post-order traversal.
    
    This function traverses the tree from the leaves up to the root,
    yielding each node. It is updated to correctly handle both standard
    DSL nodes (which have a `.serialize()` method) and new `Abstraction`
    nodes (which have a `.children` attribute).

    Parameters
    ----------
    node : object
        A DSL tree node object (e.g., Box, Scale, or Abstraction).

    Yields
    ------
    object
        Each subtree node found in the DSL tree, starting from the
        leaves and moving up to the root.
    """
    
    children = []
    if isinstance(node, Abstraction):
        # Handle Abstraction node
        children = node.children
    elif hasattr(node, "serialize"):
        # Handle original DSL node
        _, (_, children_from_serialize) = node.serialize()
        # Filter children to only include node-like objects
        children = [
            c for c in children_from_serialize 
            if hasattr(c, "serialize") or isinstance(c, Abstraction)
        ]
    else:
        # Not a node we can process (e.g., a primitive value like 'n_fold')
        return

    # Recurse into children first (post-order traversal)
    for child in children:
        yield from find_all_subtrees(child)
    
    # Yield the parent node itself after its children
    yield node


def collect_singleton_and_pair_data(dsl_shapes):
    debug_error("bruh")
    """Collect singleton and parent-child pair parameter data from DSL shapes.

    This function iterates through a list of DSL root nodes, traverses
    each tree, and collects two types of data:
    1.  **Singleton Data:** Parameters associated with a single node
        (e.g., the parameters for a `Box` or `Abs(MyPattern)` node).
    2.  **Pair Data:** Combined parameters from an immediate parent-child
        relationship (e.g., parameters from `Scale` combined with
        parameters from its child `Box`).
        
    This version is updated to correctly identify nodes and parameters
    from `Abstraction` instances.

    Parameters
    ----------
    dsl_shapes : list[object]
        A list of DSL shape root nodes (e.g., `[dsl_tree1, dsl_tree2]`).

    Returns
    -------
    tuple[dict, dict]
        A tuple of two dictionaries: `(singleton_data, pair_data)`.
        -   `singleton_data` (dict): Maps node type names (str) to a
            list of their parameter lists (list[list]).
            Example: `{"Box": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                       "Abs(LegPattern)": [[1.0, 0.5], [1.1, 0.6]]}`
        -   `pair_data` (dict): Maps parent-child signature strings (str)
            to a list of their combined parameter lists (list[list]).
            Example: `{"Scale(Box)": [[0.5, 0.1, 0.2, 0.3], ...]}`
    """
    if not dsl_shapes:
        debug_error("No DSL shapes provided for analysis.")
        return {}, {}

    s_data = defaultdict(list)
    p_data = defaultdict(list)

    for shape in dsl_shapes:
        # Use the updated subtree finder
        for node in find_all_subtrees(shape): 
            
            p_params, children = [], []
            name = ""
            
            # --- Get parent node info ---
            if isinstance(node, Abstraction):
                # Special name and params for Abstraction nodes
                name = f"Abs({node.pattern_name})"
                p_params = node.compressed_params
                children = node.children
            elif hasattr(node, "serialize"):
                # Standard DSL node
                name, (p_params, children_from_serialize) = \
                    type(node).__name__, node.serialize()[1]
                # Filter children to only include node-like objects
                children = [
                    c for c in children_from_serialize
                    if hasattr(c, "serialize") or isinstance(c, Abstraction)
                ]
            else:
                # Should be filtered by find_all_subtrees, but as a safeguard
                continue 

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
                
                # Get child node info
                if isinstance(child, Abstraction):
                    c_name = f"Abs({child.pattern_name})"
                    c_params = child.compressed_params
                elif hasattr(child, "serialize"):
                    c_name, (c_params, _) = \
                        type(child).__name__, child.serialize()[1]
                else:
                    # Not a node (e.g., an int parameter)
                    continue 
                
                # Ensure c_params is a list or tuple
                if c_params is not None and not isinstance(c_params, (list, tuple)):
                    c_params = [c_params]

                # Use list() to handle potential empty lists/None correctly
                # This creates [parent_params, child_params]
                combo_params = list(p_params or []) + list(c_params or [])
                
                if combo_params:
                    pair_sig = f"{name}({c_name})"
                    p_data[pair_sig].append(combo_params)

    # Return dictionaries sorted by key for consistent output
    return dict(sorted(s_data.items())), dict(sorted(p_data.items()))