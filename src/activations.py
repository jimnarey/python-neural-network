"""
Activation functions for use in network layers.
"""

# Some of these are left as placeholders for now since the plan is not
# to use numpy for the final implementation.
import numpy as np

from src.constants import e


# Appears to be more common in modern neural networks than the former
# favourite the sigmoid due to lower computational cost
def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) function. Very simple but when implemented
    across several neurons, capable of describing complex non-linear functions
    """
    return np.maximum(0, x)


def linear(x: np.ndarray):
    pass


def step(x: np.ndarray):
    pass


def sigmoid(x: np.ndarray):
    pass


# With both the softmax functions we're making an assumption that the
# input is a batch of values, hence a 2D array


def softmax_np(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function for use in the output layer of a network.
    """
    # Subtract the largest value in the array from each number in the array
    # This avoids overflow errors (try np.exp(1000) in REPL) caused by
    # exploding values. All exponentiated values now fall between 0 and 1
    # axis=1 sums each row whereas axis=0 would sum
    # each column. The default (axis=None) produces a single scalar value
    # which is the sum of all values in the 2D array
    # keepdims=True produces a matrix of the same shape as the input
    # (a 2D array), even though it could be expressed as a simple vector
    safe_x = x - np.max(x, axis=1, keepdims=True)
    # Exponentiate the values (remove negatives, without losing meaning)
    exp_values = np.exp(safe_x)
    # Normalise the values.
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


def softmax_py(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function using Python operations to illustrate
    more clearly what it is doing. Slow by comparison.
    """
    result = []
    for row in x:
        max_val = max(row)
        safe_x = [value - max_val for value in row]
        exp_values = [e**value for value in safe_x]
        norm_base = sum(exp_values)
        probabilities = [value / norm_base for value in exp_values]
        result.append(probabilities)
    return np.array(result)
