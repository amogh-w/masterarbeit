import k3d
import numpy as np
from scipy.spatial.transform import Rotation

# Color map for part labels
LABEL_COLORS = {
    0: 0xFF0000,  # red    (Backrest)
    1: 0x00FF00,  # green  (Seat)
    2: 0x0000FF,  # blue   (Leg)
    3: 0xFFFF00,  # yellow (Armrest)
    -1: 0x808080, # gray   (Unknown)
}


def expand_dsl_tree(node):
    """
    Expands a DSL object tree by calling its internal expand method.
    """
    # All the complex logic is now encapsulated within each node's .expand() method.
    return node.expand()


def plot_dsl_with_k3d(dsl_root_node):
    """
    Top-level function to expand a DSL tree and plot it using k3d.
    """
    print("Expanding DSL tree for visualization...")
    # This function call is now much simpler, delegating the work to the DSL node.
    final_boxes = expand_dsl_tree(dsl_root_node)
    print(f"Found {len(final_boxes)} total boxes after expansion.")

    plot = k3d.plot(name="Reconstructed DSL Shape")
    for box in final_boxes:
        center, lengths, quaternion, label_id = (
            box["center"],
            box["lengths"],
            box["quaternion"],
            box.get("label_id", -1),
        )
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        d1, d2, d3 = [
            col * length / 2 for col, length in zip(rotation_matrix.T, lengths)
        ]
        corners = np.array(
            [
                center - d1 - d2 - d3,
                center - d1 + d2 - d3,
                center + d1 - d2 - d3,
                center + d1 + d2 - d3,
                center - d1 - d2 + d3,
                center - d1 + d2 + d3,
                center + d1 - d2 + d3,
                center + d1 + d2 + d3,
            ],
            dtype=np.float32,
        )

        faces = np.array(
            [
                [0, 1, 3], [0, 3, 2],
                [4, 6, 7], [4, 7, 5],
                [0, 2, 6], [0, 6, 4],
                [1, 5, 7], [1, 7, 3],
                [0, 4, 5], [0, 5, 1],
                [2, 3, 7], [2, 7, 6],
            ],
            dtype=np.uint32,
        )

        color = LABEL_COLORS.get(label_id, LABEL_COLORS[-1])
        plot += k3d.mesh(corners, faces, color=color)

    plot.display()