"""Spiral data generator

This module includes numpy and pure Python implementations of functions
to generate three groups of points (X) with each group representing a spiral
shape when plotted. Each implementation also returns an array/list (y) which
records which point belongs to which group.

The names of the retun values reflect the convention whereby datasets are
denoted with 'X' and classifications with 'y'.

Example output data can be found in the X.json and y.json files in this
directory
"""

import math
import random
import numpy as np

np.random.seed(0)

# The datasets (X) returned by these functions are not interchangable
# because, amongst other things, Python lists of lists do not support
# slicing by dimension as numpy arrays do.

# We probably need to shuffle the output data (and classifications)
# before using this in training, otherwise the model will likely
# learn patterns specific to the ordering of the data, leading to
# poor generalisation.

# TODO - Add conditional logic to use the numpy version by default and
# and the Python version if numpy is not available.

# TODO - Move these to separate modules. We can use conditional imports
# at runtime but that won't help with the type checker


# import matplotlib.pyplot as plt
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
# ply.show()
def generate_numpy(points: int, classes: int) -> np.ndarray:
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# import matplotlib.pyplot as plt
# plt.scatter([x[0] for x in X], [x[1] for x in X], c=y, cmap="brg")
# ply.show()
def generate_py(points: int, classes: int) -> tuple[list[list[float]], list[int]]:
    X = [[0.0, 0.0] for _ in range(points * classes)]
    y = [0 for _ in range(points * classes)]
    rng = random.Random(0)
    for class_number in range(classes):
        start = points * class_number
        end = points * (class_number + 1)
        if points == 1:
            radii = [0.0]
            theta_base = [class_number * 4.0]
        else:
            radii = [index / (points - 1) for index in range(points)]
            theta_start = class_number * 4.0
            theta_step = 4.0 / (points - 1)
            theta_base = [theta_start + (theta_step * index) for index in range(points)]
        for offset, row_index in enumerate(range(start, end)):
            radius = radii[offset]
            theta = theta_base[offset] + rng.gauss(0.0, 0.2)
            X[row_index] = [
                radius * math.sin(theta * 2.5),
                radius * math.cos(theta * 2.5),
            ]
            y[row_index] = class_number
    return X, y
