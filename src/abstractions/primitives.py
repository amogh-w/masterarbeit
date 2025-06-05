import matplotlib.pyplot as plt
from torch import Tensor


class Box:
    def __init__(self, center: Tensor, scale: Tensor):
        self.center = center
        self.scale = scale


def show_boxes(boxes: list[Box], limits=(-1, 1)):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)

    for i, box in enumerate(boxes):
        rect = plt.Rectangle(
            (box.center[0].item() - box.scale[0].item() / 2, box.center[1].item() - box.scale[1].item() / 2),
            box.scale[0].item(),
            box.scale[1].item(),
            rotation_point="center",
            color=f"C{i}",
            fill=False
        )
        ax.add_patch(rect)

    plt.grid()
    plt.show()