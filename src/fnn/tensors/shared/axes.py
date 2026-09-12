"""Shared axes manipulation/handling"""


def validate_axis_is_int_or_none(axis: object) -> None:
    if axis is not None and type(axis) is not int:
        raise TypeError("axis must be an int or None")


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
