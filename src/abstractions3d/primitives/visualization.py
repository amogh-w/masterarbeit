import k3d
import numpy as np

from abstractions3d.primitives.shapes import Box3D


def visualize_boxes_3d(boxes: list[Box3D]):
    """
    Visualizes a list of 3D Box3D objects using an interactive k3d plot.

    Parameters
    ----------
    boxes : list[Box3D]
        List of Box3D instances to visualize.
    """
    plot = k3d.plot()

    for box in boxes:
        cx, cy, cz = box.center.tolist()
        sx, sy, sz = (
            box.scale / 2
        ).tolist()  # half-sizes for easier corner calculation

        # Define the 12 edges of the box
        edges = [
            # Bottom face
            [[cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz]],
            [[cx + sx, cy - sy, cz - sz], [cx + sx, cy + sy, cz - sz]],
            [[cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz]],
            [[cx - sx, cy + sy, cz - sz], [cx - sx, cy - sy, cz - sz]],
            # Top face
            [[cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz]],
            [[cx + sx, cy - sy, cz + sz], [cx + sx, cy + sy, cz + sz]],
            [[cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz]],
            [[cx - sx, cy + sy, cz + sz], [cx - sx, cy - sy, cz + sz]],
            # Vertical edges
            [[cx - sx, cy - sy, cz - sz], [cx - sx, cy - sy, cz + sz]],
            [[cx + sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz + sz]],
            [[cx + sx, cy + sy, cz - sz], [cx + sx, cy + sy, cz + sz]],
            [[cx - sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz + sz]],
        ]

        # Add edges to the plot
        for edge in edges:
            plot += k3d.line(
                np.array(edge, dtype=np.float32), color=0x0055FF, width=0.01
            )

    plot.display()
