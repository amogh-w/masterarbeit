import torch
import torch.nn as nn
# Import all the DSL node classes you've defined
from abstractionssymh.dsl_nodes import Box, Scale, Rotate, Translate, Union, SymRef, SymRot, SymTrans

# ==============================================================================
# 1. THE ABSTRACTION NODE
# ==============================================================================

class Abstraction:
    """
    A special DSL node that holds a compressed representation of a sub-tree.
    It uses a trained neural network decoder to expand itself back into the
    original, full sub-tree on demand.
    """
    def __init__(self, signature, compressed_params, other_params, decoder: nn.Module):
        self.signature = signature
        self.compressed_params = torch.tensor(compressed_params, dtype=torch.float32)
        self.other_params = other_params
        self.decoder = decoder
        # Put the decoder in evaluation mode
        self.decoder.eval()

    def expand(self):
        """
        The core method: decompress parameters and instantiate the full shape.
        """
        with torch.no_grad():
            reconstructed_floats = self.decoder(self.compressed_params).cpu().numpy().tolist()
        
        # Create copies of the lists to avoid modifying them during instantiation
        type_list = signature_to_type_list(self.signature)
        other_params_copy = self.other_params[:]
        
        # Rebuild the full shape from the decompressed parameters
        full_shape = instantiate(type_list, reconstructed_floats, other_params_copy)
        return full_shape.expand()

    def serialize(self):
        """ An Abstraction node serializes to itself. """
        return (Abstraction, ([], [self]))
        
    def __str__(self):
        return f"Abstraction(sig={self.signature}, params={self.compressed_params.shape})"

# ==============================================================================
# 2. HELPER FUNCTIONS FOR SERIALIZATION AND DESERIALIZATION
# ==============================================================================

def instantiate(type_list, float_params, other_params):
    """
    Recursively builds a DSL object tree from serialized lists of types and parameters.
    This is the inverse of the .serialize() and flatten_for_params() methods.
    """
    if not type_list:
        raise ValueError("Type list is empty, cannot instantiate.")

    type_class = type_list.pop(0)

    # Base case
    if type_class == Box:
        label = other_params.pop(0)
        return Box(label=label)

    # Recursive cases for single-child nodes
    elif type_class in [Scale, Rotate, Translate, SymRef, SymRot, SymTrans]:
        if type_class == Scale:
            params = float_params[:3]; del float_params[:3]
            child = instantiate(type_list, float_params, other_params)
            return Scale(child=child, lengths=params)
        elif type_class == Rotate:
            params = float_params[:4]; del float_params[:4]
            child = instantiate(type_list, float_params, other_params)
            return Rotate(child=child, quaternion=params)
        elif type_class == Translate:
            params = float_params[:3]; del float_params[:3]
            child = instantiate(type_list, float_params, other_params)
            return Translate(child=child, center=params)
        elif type_class == SymRef:
            params = float_params[:6]; del float_params[:6]
            child = instantiate(type_list, float_params, other_params)
            return SymRef(child=child, plane_normal=params[:3], point_on_plane=params[3:])
        elif type_class == SymRot:
            params = float_params[:6]; del float_params[:6]
            n_fold = other_params.pop(0)
            child = instantiate(type_list, float_params, other_params)
            return SymRot(child=child, axis=params[:3], center=params[3:], n_fold=n_fold)
        elif type_class == SymTrans:
            params = float_params[:3]; del float_params[:3]
            n_fold = other_params.pop(0)
            child = instantiate(type_list, float_params, other_params)
            return SymTrans(child=child, end_point=params, n_fold=n_fold)

    # Recursive case for two-child node
    elif type_class == Union:
        left_child = instantiate(type_list, float_params, other_params)
        right_child = instantiate(type_list, float_params, other_params)
        return Union(left=left_child, right=right_child)

    raise ValueError(f"Unknown type class for instantiation: {type_class}")


def get_pattern_signature(dsl_node):
    """
    Recursively traverses a DSL node and returns a nested tuple of its types.
    This unique signature is used as a key to group identical structures.
    Example: (Union, (Translate, (Box,)), (Scale, (Box,)))
    """
    node_type, (_, other_params_and_children) = dsl_node.serialize()
    
    # *** FIXED LOGIC HERE ***
    # Only recurse into items that are actual DSL nodes (not integers)
    child_signatures = tuple(
        get_pattern_signature(item) 
        for item in other_params_and_children if hasattr(item, 'serialize')
    )
    
    return (node_type,) + child_signatures


def flatten_for_params(dsl_node):
    """
    Recursively traverses a DSL node and flattens its parameters into two lists:
    1. A list of all floating-point numbers.
    2. A list of all other parameters (e.g., labels, n_fold integers).
    """
    _ , (float_params, other_params_and_children) = dsl_node.serialize()
    
    all_floats = list(float_params)
    all_others = []

    for item in other_params_and_children:
        # Check if the item is a DSL node by seeing if it has a 'serialize' method
        if hasattr(item, 'serialize'):
            child_floats, child_others = flatten_for_params(item)
            all_floats.extend(child_floats)
            all_others.extend(child_others)
        else: # It's a non-node parameter (like a label or n_fold)
             all_others.append(item)
            
    return all_floats, all_others


def signature_to_type_list(signature):
    """
    Converts a nested signature tuple back into a flat list of types for instantiation.
    """
    type_list = [signature[0]]
    for child_sig in signature[1:]:
        type_list.extend(signature_to_type_list(child_sig))
    return type_list