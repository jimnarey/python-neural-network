"""Shared guards for tensor creation and operations"""


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


def validate_transpose_axes_are_permutation(axes: tuple[int, ...], ndim: int) -> None:
    if len(axes) != ndim:
        raise ValueError("transpose axes must include every tensor axis exactly once")
    if set(axes) != set(range(ndim)):
        raise ValueError("transpose axes must include every tensor axis exactly once")


def validate_tensor_conversion_input(data: object) -> None:
    if not isinstance(data, (list, tuple)):
        raise ValueError("Tensor conversion requires a list or tuple input.")


# TODO - this needs unit tests but wait until we know we're not going to
# refactor any further
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
