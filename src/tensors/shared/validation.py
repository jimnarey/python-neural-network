"""Shared guards for tensor creation and operations"""

import math


def shape_size(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def validate_shape_not_rank_0(shape: tuple[int, ...]) -> None:
    if not shape:
        raise ValueError("Tensor creation methods require a non-empty shape.")


def validate_shape_has_no_negative_dimensions(
    shape: tuple[int, ...], method_name: str
) -> None:
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"{method_name} does not support negative values in the target shape"
        )


def validate_scalar_is_not_bool(value: object) -> None:
    if isinstance(value, bool):
        raise ValueError("scalar value must not be a bool")


def validate_reduction_has_values(
    shape: tuple[int, ...], reduced_axes: tuple[int, ...]
) -> None:
    if shape_size(tuple(shape[axis] for axis in reduced_axes)) == 0:
        raise ValueError("reduction operation has no values")


def validate_tensor_has_values(shape: tuple[int, ...]) -> None:
    if shape_size(shape) == 0:
        raise ValueError(f"tensor with shape {shape} has no values")


def validate_shapes_match_except_axis(
    shapes: tuple[tuple[int, ...], ...], axis: int
) -> None:
    if not shapes:
        raise ValueError("at least one shape is required")
    base_shape = shapes[0]
    if axis < 0 or axis >= len(base_shape):
        raise ValueError("axis is outside the valid range for the shape")
    for shape in shapes[1:]:
        if len(shape) != len(base_shape):
            raise ValueError("shapes must have the same rank")
        for dim, (base_dim, other_dim) in enumerate(zip(base_shape, shape)):
            if dim != axis and base_dim != other_dim:
                raise ValueError("shapes must match except along the chosen axis")


def validate_axes_are_unique(axes: tuple[int, ...]) -> None:
    if len(set(axes)) != len(axes):
        raise ValueError("axes must not contain duplicates")


def validate_axes_are_permutation(axes: tuple[int, ...], ndim: int) -> None:
    validate_axes_are_unique(axes)
    if set(axes) != set(range(ndim)):
        raise ValueError("axes must include every tensor axis exactly once")


def validate_tensor_conversion_root_is_sequence(data: object) -> None:
    if not isinstance(data, (list, tuple)):
        raise ValueError("Tensor conversion requires a list or tuple input.")


def parse_tensor_data(data: object) -> tuple[tuple[int, ...], list[float]]:
    """
    Validate nested tensor input and return its shape with flat float values.

    The input must be a rectangular nested list/tuple structure whose leaf
    values are plain Python ints or floats. The returned values are ordered
    by walking the nested structure from left to right.

    The Python backend requires the returned values in order to instantiate
    PythonTensor. The NumPy backend just needs this function to not raise.
    There is some duplication of work in the latter case because np.array
    must also walk the input list(s)/tuple(s) when instantiating its tensor
    representation. That was deemed preferable to having two mostly-duplicative
    input guards. This is not on a hot path.
    """
    if isinstance(data, (list, tuple)):
        if not data:
            return (0,), []
        first_shape, first_values = parse_tensor_data(data[0])
        values = list(first_values)
        for item in data[1:]:
            item_shape, item_values = parse_tensor_data(item)
            if item_shape != first_shape:
                raise ValueError("Tensor conversion requires rectangular input.")
            values.extend(item_values)
        return (len(data), *first_shape), values
    if type(data) is int:
        return (), [float(data)]
    if type(data) is float:
        return (), [data]
    raise ValueError("Tensor conversion requires numeric values.")
