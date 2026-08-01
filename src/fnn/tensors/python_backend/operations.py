import math
from array import array
from typing import Callable, Iterable

from fnn.tensors.python_backend.python_tensor import PythonTensor
from fnn.tensors.shared.broadcasting import get_target_shape, get_target_strides
from fnn.tensors.shared.matmul import get_matmul_result_index_parts
from fnn.tensors.shared.reductions import (
    get_reduction_axes_and_target_shape,
    get_reduction_target_index,
)
from fnn.tensors.shared.types import Scalar
from fnn.tensors.shared.validation import validate_scalar_is_not_bool


def first_max_index(values: Iterable[Scalar]) -> int:
    iterator = iter(values)
    best_index = 0
    best_value = next(iterator)
    for index, value in enumerate(iterator, start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def argmax_to_scalar(x: PythonTensor) -> int:
    return first_max_index(value for _, value in x.items())


def argmax_to_tensor(
    x: PythonTensor, normalised_axis: int, target_shape: tuple[int, ...]
) -> PythonTensor:
    result = PythonTensor(target_shape, typecode=PythonTensor.INT)
    for target_index in result.indices():
        best_axis_index = first_max_index(
            x.get_scalar(
                target_index[:normalised_axis]
                + (axis_index,)
                + target_index[normalised_axis:]
            )
            for axis_index in range(x.shape[normalised_axis])
        )
        result.set_scalar(target_index, best_axis_index)
    return result


def get_concatenate_shape(
    xs: tuple[PythonTensor, ...], normalised_axis: int
) -> tuple[int, ...]:
    base_shape = xs[0].shape
    axis_size = sum(x.shape[normalised_axis] for x in xs)
    return (
        base_shape[:normalised_axis] + (axis_size,) + base_shape[normalised_axis + 1 :]
    )


# TODO - this is probably slow. In many cases where it is used we can
# probably use shape data to construct a flat buffer with the output
# values properly located and pass to a new instance of PythonTensor.
# We might even get away without pre-calculating the shape. On the plus
# side, the current implementation is generic and simple.
def copy_sequence_values(
    xs: tuple[PythonTensor, ...],
    result: PythonTensor,
    target_index_fn: Callable[[int, tuple[int, ...]], tuple[int, ...]],
) -> PythonTensor:
    for tensor_index, tensor in enumerate(xs):
        for source_index, value in tensor.items():
            result.set_scalar(target_index_fn(tensor_index, source_index), value)
    return result


def concatenate_tensors(
    xs: tuple[PythonTensor, ...], normalised_axis: int, shape: tuple[int, ...]
) -> PythonTensor:
    result = PythonTensor(shape)
    axis_offsets = []
    axis_offset = 0
    for tensor in xs:
        axis_offsets.append(axis_offset)
        axis_offset += tensor.shape[normalised_axis]
    return copy_sequence_values(
        xs,
        result,
        lambda tensor_index, source_index: source_index[:normalised_axis]
        + (source_index[normalised_axis] + axis_offsets[tensor_index],)
        + source_index[normalised_axis + 1 :],
    )


def get_stack_shape(
    xs: tuple[PythonTensor, ...], normalised_axis: int
) -> tuple[int, ...]:
    base_shape = xs[0].shape
    return base_shape[:normalised_axis] + (len(xs),) + base_shape[normalised_axis:]


def stack_tensors(
    xs: tuple[PythonTensor, ...], normalised_axis: int, shape: tuple[int, ...]
) -> PythonTensor:
    result = PythonTensor(shape)
    return copy_sequence_values(
        xs,
        result,
        lambda tensor_index, source_index: source_index[:normalised_axis]
        + (tensor_index,)
        + source_index[normalised_axis:],
    )


def map_unary(x: PythonTensor, op: Callable[[float], float]) -> PythonTensor:
    return PythonTensor(x.shape, array("d", (op(value) for _, value in x.items())))


def map_binary(
    a: PythonTensor,
    b: PythonTensor | Scalar,
    op: Callable[[float, float], float],
) -> PythonTensor:
    if not isinstance(b, PythonTensor):
        validate_scalar_is_not_bool(b)
        scalar = float(b)
        return PythonTensor(
            a.shape,
            array("d", (op(value, scalar) for _, value in a.items())),
        )
    target_shape = get_target_shape(a.shape, b.shape)
    a_view = a.view(
        target_shape,
        strides=get_target_strides(a.shape, a.strides, target_shape),
    )
    b_view = b.view(
        target_shape,
        strides=get_target_strides(b.shape, b.strides, target_shape),
    )
    return PythonTensor(
        target_shape,
        array(
            "d",
            (
                op(a_view.get_scalar(index), b_view.get_scalar(index))
                for index in a_view.indices()
            ),
        ),
    )


def get_matmul_value(
    a: PythonTensor,
    b: PythonTensor,
    result_index: tuple[int, ...],
) -> float:
    """
    Return the single value produced at result_index in a matmul result.

    The result index is split into the operand positions needed for that
    output value. The function then walks along the shared inner dimension,
    multiplies each left-hand value by the corresponding right-hand value,
    and returns the sum of those products.
    """
    a_leading_index, b_leading_index, row_index, column_index = (
        get_matmul_result_index_parts(result_index, a.shape, b.shape)
    )
    total = 0.0
    for inner_index in range(a.shape[-1]):
        total += a.get_scalar(
            _get_matmul_left_index(a_leading_index, row_index, inner_index, a.shape)
        ) * b.get_scalar(
            _get_matmul_right_index(b_leading_index, column_index, inner_index, b.shape)
        )
    return total


def matmul_to_scalar(a: PythonTensor, b: PythonTensor) -> float:
    return get_matmul_value(a, b, ())


def matmul_to_tensor(
    a: PythonTensor, b: PythonTensor, result_shape: tuple[int, ...]
) -> PythonTensor:
    result = PythonTensor(result_shape)
    for result_index in result.indices():
        result.set_scalar(result_index, get_matmul_value(a, b, result_index))
    return result


def matmul_tensors(
    a: PythonTensor, b: PythonTensor, result_shape: tuple[int, ...]
) -> PythonTensor | float:
    if result_shape == ():
        return matmul_to_scalar(a, b)
    return matmul_to_tensor(a, b, result_shape)


def _get_matmul_left_index(
    leading_index: tuple[int, ...],
    row_index: int | None,
    inner_index: int,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    if len(shape) == 1:
        return (inner_index,)
    if row_index is None:
        raise ValueError("row index is required for rank-2-or-higher left operand")
    return leading_index + (row_index, inner_index)


def _get_matmul_right_index(
    leading_index: tuple[int, ...],
    column_index: int | None,
    inner_index: int,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    if len(shape) == 1:
        return (inner_index,)
    if column_index is None:
        raise ValueError("column index is required for rank-2-or-higher right operand")
    return leading_index + (inner_index, column_index)


def reduce_to_scalar(
    x: PythonTensor,
    initial_value: float,
    accumulate_fn: Callable[[float, float], float],
) -> float:
    accumulator = initial_value
    for _, value in x.items():
        accumulator = accumulate_fn(accumulator, value)
    return accumulator


def reduce_to_tensor(
    x: PythonTensor,
    reduced_axes: tuple[int, ...],
    target_shape: tuple[int, ...],
    keepdims: bool,
    initial_value: float,
    accumulate_fn: Callable[[float, float], float],
) -> PythonTensor:
    if target_shape == ():
        raise ValueError("target shape must not be empty")
    result = PythonTensor(
        target_shape, array("d", [initial_value]) * math.prod(target_shape)
    )
    for source_index, value in x.items():
        target_index = get_reduction_target_index(source_index, reduced_axes, keepdims)
        result.set_scalar(
            target_index, accumulate_fn(result.get_scalar(target_index), value)
        )
    return result


def reduce(
    x: PythonTensor,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    initial_value: float,
    accumulate_fn: Callable[[float, float], float],
) -> PythonTensor | float:
    reduced_axes, target_shape = get_reduction_axes_and_target_shape(
        x.shape, axis, keepdims
    )
    if target_shape == ():
        return reduce_to_scalar(x, initial_value, accumulate_fn)
    return reduce_to_tensor(
        x, reduced_axes, target_shape, keepdims, initial_value, accumulate_fn
    )


def divide_reduction_result(
    result: PythonTensor | float, divisor: float
) -> PythonTensor | float:
    if isinstance(result, float):
        return result / divisor
    for index, value in result.items():
        result.set_scalar(index, value / divisor)
    return result
