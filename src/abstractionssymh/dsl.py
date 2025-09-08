class Box:
    def __init__(self, label: int, center, dims, quaternion):
        self.label = label
        self.center = center
        self.lengths = dims
        self.quaternion = quaternion

    def __str__(self):
        center_str = np.round(np.array(self.center), 2)
        lengths_str = np.round(np.array(self.lengths), 2)
        quat_str = np.round(np.array(self.quaternion), 2)
        return f"Box(label_id={self.label}, center={center_str}, dims={lengths_str}, quat={quat_str})"

class Union:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return "Union"

class Symmetry:
    def __init__(self, child):
        self.child = child

class SymRef(Symmetry):
    def __init__(self, child, plane_normal, point_on_plane):
        super().__init__(child)
        self.plane = plane_normal
        self.point_on_plane = point_on_plane

    def __str__(self):
        plane_vec = np.round(np.array(self.plane), 2)
        point_vec = np.round(np.array(self.point_on_plane), 2)
        return f"Symmetry(Reflection) across plane with normal={plane_vec} at point={point_vec}"

class SymRot(Symmetry):
    def __init__(self, child, axis, center, n_fold: int):
        super().__init__(child)
        self.axis = axis
        self.center = center
        self.n = n_fold

    def __str__(self):
        axis_vec = np.round(np.array(self.axis), 2)
        center_vec = np.round(np.array(self.center), 2)
        return f"Symmetry(Rotation) of {self.n}-fold around axis={axis_vec} at center={center_vec}"

class SymTrans(Symmetry):
    def __init__(self, child, end_point, n_fold: int):
        super().__init__(child)
        self.end_point = end_point
        self.n = n_fold

    def __str__(self):
        end_point_vec = np.round(np.array(self.end_point), 2)
        return f"Symmetry(Translation) of {self.n} items towards end_point={end_point_vec}"