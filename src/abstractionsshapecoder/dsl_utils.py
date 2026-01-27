"""dsl_utils.py

Utilities for analyzing DSL object trees: recursively finding all subtrees and
collecting singleton and parent-child pair parameters.
"""

from collections import defaultdict

# Dummy Abstraction class to prevent errors if you don't use L2 analysis yet
class Abstraction:
    pass

def find_all_subtrees(node):
    """Recursively find all subtrees of a DSL node via post-order traversal."""
    
    children = []
    if isinstance(node, Abstraction):
        children = node.children
    elif hasattr(node, "serialize"):
        # serialize returns: (Class, (params, children_list))
        _, (_, children_from_serialize) = node.serialize()
        
        # Filter: only keep objects that are actual Nodes (have serialize or are Abstraction)
        children = [
            c for c in children_from_serialize 
            if hasattr(c, "serialize") or isinstance(c, Abstraction)
        ]
    else:
        return

    # Recurse into children first
    for child in children:
        yield from find_all_subtrees(child)
    
    # Yield the parent node itself
    yield node


def collect_singleton_and_pair_data(dsl_shapes):
    """Collect parameter data from DSL shapes."""
    if not dsl_shapes:
        print("[ERROR] No DSL shapes provided for analysis.")
        return {}, {}

    s_data = defaultdict(list)
    p_data = defaultdict(list)

    for shape in dsl_shapes:
        # Traverse every node in the tree
        for node in find_all_subtrees(shape): 
            
            p_params, children = [], []
            name = ""
            
            # --- Identify Node Type ---
            if isinstance(node, Abstraction):
                name = f"Abs({node.pattern_name})"
                p_params = node.compressed_params
                children = node.children
            elif hasattr(node, "serialize"):
                # serialize returns (Class, (params, children))
                # We use type(node).__name__ for the string name
                name = type(node).__name__
                _, (p_params, children_from_serialize) = node.serialize()
                
                # Filter children to only process Nodes (ignore labels/ints)
                children = [
                    c for c in children_from_serialize
                    if hasattr(c, "serialize") or isinstance(c, Abstraction)
                ]
            else:
                continue 

            # --- 1. Store Singleton Data ---
            # (Parameters inherent to this specific node)
            if p_params is not None:
                # Ensure it's a list
                if not isinstance(p_params, (list, tuple)):
                    p_params = [p_params]
                
                if len(p_params) > 0:
                    s_data[name].append(p_params)

            # --- 2. Store Pair Data ---
            # (Parameters of this node + Parameters of its immediate child)
            for child in children:
                c_params = []
                c_name = ""
                
                if isinstance(child, Abstraction):
                    c_name = f"Abs({child.pattern_name})"
                    c_params = child.compressed_params
                elif hasattr(child, "serialize"):
                    c_name = type(child).__name__
                    _, (c_params, _) = child.serialize()
                else:
                    continue 
                
                if c_params is not None and not isinstance(c_params, (list, tuple)):
                    c_params = [c_params]

                # Combine: [Parent Params] + [Child Params]
                # Use "or []" to handle None safely
                combo_params = list(p_params or []) + list(c_params or [])
                
                if combo_params:
                    pair_sig = f"{name}({c_name})"
                    p_data[pair_sig].append(combo_params)

    # Sort keys for consistent output
    return dict(sorted(s_data.items())), dict(sorted(p_data.items()))