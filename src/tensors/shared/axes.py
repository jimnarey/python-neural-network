"""Shared axes manipulation/handling"""


def normalise_axis(axis: int, ndim: int) -> int:
    """
    Convert a negative axis value to its positive equivalent.
    """
    if not -ndim <= axis < ndim:
        raise ValueError("axis is out of bounds")
    if axis < 0:
        return axis + ndim
    return axis


def normalise_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(normalise_axis(axis, ndim) for axis in axes)
