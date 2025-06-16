"""
shapes.py

Defines geometric shape primitives in 2D space, including Box and Circle.
"""

from torch import Tensor


class Box:
    """
    Represents a rectangular box in 2D space using a center point and a scale (width and height).

    Attributes:
        center (Tensor): A 2D tensor representing the center coordinates (x, y).
        scale (Tensor): A 2D tensor representing the width and height of the box.
    """

    def __init__(self, center: Tensor, scale: Tensor):
        """
        Initializes a Box with a given center and scale.

        Parameters
        ----------
        center : Tensor
            The center coordinates of the box (x, y).
        scale : Tensor
            The width and height of the box.

        Returns
        -------
        None
        """
        self.center = center
        self.scale = scale


class Circle:
    """
    Represents a circular shape in 2D space using a center point and a radius.

    Attributes:
        center (Tensor): A 2D tensor representing the center coordinates (x, y).
        radius (Tensor): A scalar tensor representing the radius of the circle.
    """

    def __init__(self, center: Tensor, radius: Tensor):
        """
        Initializes a Circle with a given center and radius.

        Parameters
        ----------
        center : Tensor
            The center coordinates of the circle (x, y).
        radius : Tensor
            A scalar tensor representing the radius of the circle.

        Returns
        -------
        None
        """
        self.center = center
        self.radius = radius
