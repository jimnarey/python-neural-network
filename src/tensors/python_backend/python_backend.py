"""Pure Python implementation of the tensor backend

This is very slow in comparison to the reference (numpy) implementaion.
It's purpose is to ensure the underlying tensor calculations are genuinely
understood. It also serves as a basis for more performant custom backends.
"""

from src.tensors.python_backend.python_tensor import PythonTensor
from src.tensors.python_backend.operations import (
    argmax_to_scalar,
    argmax_to_tensor,
    concatenate_tensors,
    divide_reduction_result,
    get_concatenate_shape,
    get_stack_shape,
    map_binary,
    matmul_tensors,
    map_unary,
    reduce,
    stack_tensors,
)
from src.tensors.python_backend.validation import (
    require_non_empty_tensor_sequence,
    validate_stack_shapes,
)
from src.tensors.shared.axes import normalise_axis, normalise_axes
from src.tensors.shared.matmul import get_matmul_result_shape
from src.tensors.shared.reductions import get_reduction_axes_and_target_shape
from src.tensors.shared.scalar_ops import (
    divide_scalar,
    log_scalar,
    sign_scalar,
    sqrt_scalar,
)
from src.tensors.shared.types import Scalar
from src.tensors.shared.validation import (
    parse_tensor_data,
    validate_tensor_has_values,
    validate_reduction_has_values,
    validate_scalar_is_not_bool,
    validate_shape_has_no_negative_dimensions,
    validate_shape_not_rank_0,
    validate_shapes_match_except_axis,
    validate_tensor_conversion_root_is_sequence,
    validate_axes_are_permutation,
    validate_matmul_core_dimensions,
    validate_matmul_operand_ranks,
)
from typing import Sequence, Optional
from array import array
import math
import random


class PythonBackend:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._random = random.Random(seed)

    # PythonTensor supports a writable flag which is not currently
    # part of the Protocol class, so not used here.
    def to_tensor(self, data: list[object] | tuple[object, ...]) -> PythonTensor:
        validate_tensor_conversion_root_is_sequence(data)
        shape, values = parse_tensor_data(data)
        return PythonTensor(shape, array("d", values))

    def to_python(self, tensor: PythonTensor) -> list:
        return tensor.to_list()

    def randn(self, shape: tuple[int, ...]) -> PythonTensor:
        # Use .normalvariate here in place of .gauss because we know it's thread safe.
        # It's also a little slower so possibly revisit, though the difference is
        # likely to be marginal.
        return PythonTensor(
            shape,
            array(
                "d",
                (self._random.normalvariate(0.0, 1.0) for _ in range(math.prod(shape))),
            ),
        )

    def zeros(self, shape: tuple[int, ...]) -> PythonTensor:
        return PythonTensor(shape)

    def zeros_like(self, x: PythonTensor) -> PythonTensor:
        return self.zeros(x.shape)

    def ones(self, shape: tuple[int, ...]) -> PythonTensor:
        return PythonTensor(shape, array("d", [1.0]) * math.prod(shape))

    def ones_like(self, x: PythonTensor) -> PythonTensor:
        return self.ones(x.shape)

    def full(self, shape: tuple[int, ...], fill_value: Scalar) -> PythonTensor:
        validate_scalar_is_not_bool(fill_value)
        return PythonTensor(shape, array("d", [float(fill_value)]) * math.prod(shape))

    def full_like(self, x: PythonTensor, fill_value: Scalar) -> PythonTensor:
        return self.full(x.shape, fill_value)

    def empty(self, shape: tuple[int, ...]) -> PythonTensor:
        return PythonTensor(shape)

    def empty_like(self, x: PythonTensor) -> PythonTensor:
        return PythonTensor(x.shape)

    # PythonTensor.copy supports a writable flag which is not currently
    # part of the Protocol class, so not used here.
    def copy(self, x: PythonTensor) -> PythonTensor:
        return x.copy()

    def shape(self, x: PythonTensor) -> tuple[int, ...]:
        return x.shape

    # TODO - reshape here makes a copy. NumPy reshape will attempt to create a cheap
    # view if that is possible and fall back to a copy if not. Implement this later
    # and before attempting to implement any of the tougher backends, so we can
    # borrow the logic. This behaviour should be added to the contract and tested.
    def reshape(self, x: PythonTensor, shape: tuple[int, ...]) -> PythonTensor:
        validate_shape_not_rank_0(shape)
        validate_shape_has_no_negative_dimensions(shape, "reshape")
        if x.size() != math.prod(shape):
            raise ValueError("reshape cannot change the number of tensor elements")
        return PythonTensor(shape, array("d", (value for _, value in x.items())))

    def transpose(
        self, x: PythonTensor, axes: tuple[int, ...] | None = None
    ) -> PythonTensor:
        if axes is None:
            normalised_axes = tuple(reversed(range(x.ndim())))
        else:
            normalised_axes = normalise_axes(axes, x.ndim())
            validate_axes_are_permutation(normalised_axes, x.ndim())
        shape = tuple(x.shape[axis] for axis in normalised_axes)
        strides = tuple(x.strides[axis] for axis in normalised_axes)
        return x.view(shape, strides=strides)

    def add(self, a: PythonTensor, b: PythonTensor | Scalar) -> PythonTensor:
        return map_binary(a, b, lambda left, right: left + right)

    def subtract(self, a: PythonTensor, b: PythonTensor | Scalar) -> PythonTensor:
        return map_binary(a, b, lambda left, right: left - right)

    def multiply(self, a: PythonTensor, b: PythonTensor | Scalar) -> PythonTensor:
        return map_binary(a, b, lambda left, right: left * right)

    def divide(self, a: PythonTensor, b: PythonTensor | Scalar) -> PythonTensor:
        return map_binary(a, b, divide_scalar)

    def matmul(self, a: PythonTensor, b: PythonTensor) -> PythonTensor | float:
        validate_matmul_operand_ranks(a.shape, b.shape)
        validate_matmul_core_dimensions(a.shape, b.shape)
        shape = get_matmul_result_shape(a.shape, b.shape)
        return matmul_tensors(a, b, shape)

    def maximum(self, a: PythonTensor, b: PythonTensor | Scalar) -> PythonTensor:
        return map_binary(a, b, max)

    def exp(self, x: PythonTensor) -> PythonTensor:
        return map_unary(x, math.exp)

    def sum(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        return reduce(x, axis, keepdims, 0.0, lambda total, value: total + value)

    def max(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        reduced_axes, _ = get_reduction_axes_and_target_shape(x.shape, axis, keepdims)
        validate_reduction_has_values(x.shape, reduced_axes)
        return reduce(x, axis, keepdims, -math.inf, max)

    def minimum(self, a: PythonTensor, b: PythonTensor | Scalar) -> PythonTensor:
        return map_binary(a, b, min)

    def argmax(self, x: PythonTensor, axis: int | None = None) -> PythonTensor | int:
        validate_tensor_has_values(x.shape)
        if axis is None:
            return argmax_to_scalar(x)
        if type(axis) is not int:
            raise TypeError("axis must be an int or None")
        normalised_axis = normalise_axes((axis,), x.ndim())[0]
        target_shape = x.shape[:normalised_axis] + x.shape[normalised_axis + 1 :]
        if target_shape == ():
            return argmax_to_scalar(x)
        return argmax_to_tensor(x, normalised_axis, target_shape)

    def log(self, x: PythonTensor) -> PythonTensor:
        return map_unary(x, log_scalar)

    def sqrt(self, x: PythonTensor) -> PythonTensor:
        return map_unary(x, sqrt_scalar)

    def absolute(self, x: PythonTensor) -> PythonTensor:
        return map_unary(x, abs)

    def sign(self, x: PythonTensor) -> PythonTensor:
        return map_unary(x, sign_scalar)

    def clip(
        self, x: PythonTensor, min_value: Scalar, max_value: Scalar
    ) -> PythonTensor:
        validate_scalar_is_not_bool(min_value)
        validate_scalar_is_not_bool(max_value)
        return map_unary(x, lambda value: min(max(value, min_value), max_value))

    def mean(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        reduced_axes, _ = get_reduction_axes_and_target_shape(x.shape, axis, keepdims)
        validate_reduction_has_values(x.shape, reduced_axes)
        total = reduce(
            x, axis, keepdims, 0.0, lambda accumulator, value: accumulator + value
        )
        return divide_reduction_result(
            total, float(math.prod(x.shape[axis] for axis in reduced_axes))
        )

    def min(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        reduced_axes, _ = get_reduction_axes_and_target_shape(x.shape, axis, keepdims)
        validate_reduction_has_values(x.shape, reduced_axes)
        return reduce(x, axis, keepdims, math.inf, min)

    def std(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        reduced_axes, _ = get_reduction_axes_and_target_shape(x.shape, axis, keepdims)
        validate_reduction_has_values(x.shape, reduced_axes)
        mean = self.mean(x, axis=reduced_axes, keepdims=True)
        squared_deviations = map_binary(
            x,
            mean,
            lambda value, mean_value: (value - mean_value) * (value - mean_value),
        )
        variance = self.mean(squared_deviations, axis=reduced_axes, keepdims=keepdims)
        if isinstance(variance, float):
            return math.sqrt(variance)
        return self.sqrt(variance)

    def stack(self, xs: Sequence[PythonTensor], axis: int = 0) -> PythonTensor:
        tensors = require_non_empty_tensor_sequence(xs)
        normalised_axis = normalise_axis(axis, tensors[0].ndim() + 1)
        validate_stack_shapes(tensors)
        shape = get_stack_shape(tensors, normalised_axis)
        return stack_tensors(tensors, normalised_axis, shape)

    def concatenate(self, xs: Sequence[PythonTensor], axis: int = 0) -> PythonTensor:
        tensors = require_non_empty_tensor_sequence(xs)
        normalised_axis = normalise_axis(axis, tensors[0].ndim())
        validate_shapes_match_except_axis(
            tuple(tensor.shape for tensor in tensors), normalised_axis
        )
        shape = get_concatenate_shape(tensors, normalised_axis)
        return concatenate_tensors(tensors, normalised_axis, shape)

    def eye(self, n: int, m: int | None = None) -> PythonTensor:
        if m is None:
            m = n
        tensor = PythonTensor((n, m))
        for i in range(min(n, m)):
            tensor.set_scalar((i, i), 1.0)
        return tensor
