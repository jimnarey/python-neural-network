import math

from src.tensors.axes import normalise_axes
from src.tensors.validation import validate_axes_are_unique


def normalise_axis_argument(
    axis: int | tuple[int, ...] | None, ndim: int
) -> tuple[int, ...]:
    """
    Return a tuple of axes based on various forms the axis argument
    to a reduction method might take.

    It includes a runtime type check because unsupported argument
    types would otherwise be returned unchanged and could fail later in
    application logic, making debugging harder.
    """
    if axis is None:
        return tuple(range(ndim))
    if type(axis) is int:
        return (axis,)
    if isinstance(axis, tuple):
        return axis
    raise TypeError("axis must be None, an int, or a tuple of ints")


def get_reduction_target_shape(
    shape: tuple[int, ...],
    reduced_axes: tuple[int, ...],
    keepdims: bool = False,
) -> tuple[int, ...]:
    result = []
    for axis, dimension in enumerate(shape):
        if axis in reduced_axes:
            if keepdims:
                result.append(1)
        else:
            result.append(dimension)
    return tuple(result)


def get_reduction_axes_and_target_shape(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
    keepdims: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    reduced_axes = normalise_axis_argument(axis, len(shape))
    reduced_axes = normalise_axes(reduced_axes, len(shape))
    validate_axes_are_unique(reduced_axes)
    target_shape = get_reduction_target_shape(shape, reduced_axes, keepdims)
    return reduced_axes, target_shape


def get_reduction_target_index(
    source_index: tuple[int, ...],
    reduced_axes: tuple[int, ...],
    keepdims: bool = False,
) -> tuple[int, ...]:
    """
    Return the result index which the value at a given source index
    contributes to.

    A reduced axis is removed from the result index when keepdims is false.
    When keepdims is true, the axis is retained but has length 1, so the only
    valid index along that axis is 0.
    """
    result = []
    for axis, index in enumerate(source_index):
        if axis in reduced_axes:
            if keepdims:
                result.append(0)
        else:
            result.append(index)
    return tuple(result)


def get_reduction_count(shape: tuple[int, ...], reduced_axes: tuple[int, ...]) -> int:
    """
    Return how many source values are combined for each result value.

    This is the product of the lengths of the axes being reduced.
    """
    return math.prod(shape[axis] for axis in reduced_axes)
