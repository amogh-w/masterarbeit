from abstractions3d.dsl.core import Shape3D
from abstractions3d.dsl.nodes import Rect3D, Move3D, Union3D, SymTrans3D, SymRef3D


def instantiate_3d(type_list: list[type], param_list: list):
    def _instantiate(type_list: list[type], param_list: list):
        token = type_list.pop(0)

        if issubclass(token, Rect3D):
            s_x = _instantiate(type_list, param_list)
            s_y = _instantiate(type_list, param_list)
            s_z = _instantiate(type_list, param_list)
            return Rect3D(s_x, s_y, s_z)
        elif issubclass(token, Move3D):
            child = _instantiate(type_list, param_list)
            t_x = _instantiate(type_list, param_list)
            t_y = _instantiate(type_list, param_list)
            t_z = _instantiate(type_list, param_list)
            return Move3D(child, t_x, t_y, t_z)
        elif issubclass(token, Union3D):
            child1 = _instantiate(type_list, param_list)
            child2 = _instantiate(type_list, param_list)
            return Union3D(child1, child2)
        elif issubclass(token, SymTrans3D):
            child = _instantiate(type_list, param_list)
            axis = _instantiate(type_list, param_list)
            dist = _instantiate(type_list, param_list)
            degree = _instantiate(type_list, param_list)
            return SymTrans3D(child, axis, dist, degree)
        elif issubclass(token, SymRef3D):
            child = _instantiate(type_list, param_list)
            axis = _instantiate(type_list, param_list)
            return SymRef3D(child, axis)
        elif issubclass(token, float):
            return float(param_list.pop(0))
        elif issubclass(token, int):
            return int(param_list.pop(0))
        elif issubclass(token, str):
            return str(param_list.pop(0))
        elif issubclass(token, Shape3D):
            return param_list.pop(0)
        else:
            raise ValueError(f"Unknown token: {token}")

    return _instantiate(type_list.copy(), param_list.copy())


# def instantiate_3d(type_list: list[type], param_list: list):
#     def _inst(types, params):
#         if not types:
#             return None
#         token = types.pop(0)
#         if token is Rect3D or (hasattr(token, "__name__") and token.__name__ == "Rect3D"):
#             w = _inst(types, params)
#             h = _inst(types, params)
#             d = _inst(types, params)
#             return Rect3D(w, h, d)
#         if token is Move3D or (hasattr(token, "__name__") and token.__name__ == "Move3D"):
#             child = _inst(types, params)
#             t_x = _inst(types, params)
#             t_y = _inst(types, params)
#             t_z = _inst(types, params)
#             return Move3D(child, t_x, t_y, t_z)
#         if token is Union3D or (hasattr(token, "__name__") and token.__name__ == "Union3D"):
#             a = _inst(types, params)
#             b = _inst(types, params)
#             return Union3D(a, b)
#         if token is SymTrans3D or (hasattr(token, "__name__") and token.__name__ == "SymTrans3D"):
#             child = _inst(types, params)
#             axis = _inst(types, params)
#             dist = _inst(types, params)
#             degree = _inst(types, params)
#             return SymTrans3D(child, axis, dist, degree)
#         if token is SymRef3D or (hasattr(token, "__name__") and token.__name__ == "SymRef3D"):
#             child = _inst(types, params)
#             axis = _inst(types, params)
#             return SymRef3D(child, axis)
#         if token is float:
#             return float(params.pop(0))
#         if token is int:
#             return int(params.pop(0))
#         if token is str:
#             return str(params.pop(0))
#         if issubclass(token, Shape3D):
#             return params.pop(0)
#         raise ValueError(f"Unknown token: {token}")
#     if not type_list:
#         return None
#     return _inst(type_list.copy(), param_list.copy())
