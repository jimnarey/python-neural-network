import math
from array import array
from typing import Callable

from src.tensors.python_backend.python_tensor import PythonTensor
from src.tensors.shared.broadcasting import get_target_shape, get_target_strides
from src.tensors.shared.reductions import (
    get_reduction_axes_and_target_shape,
    get_reduction_target_index,
)
from src.tensors.shared.types import Scalar
from src.tensors.shared.validation import validate_scalar_is_not_bool


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
