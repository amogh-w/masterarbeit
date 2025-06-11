import random


def random_quantized_uniform(low: float, high: float, steps: int) -> float:
    """
    Generates a random float value between `low` and `high`, quantized into a number of steps.

    Args:
        low (float): Lower bound of the range.
        high (float): Upper bound of the range.
        steps (int): Number of discrete steps between low and high.

    Returns:
        float: A quantized random float within the specified range.
    """
    step = random.randint(0, steps)
    return low + (high - low) * step / steps
