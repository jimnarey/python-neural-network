"""Pure Python implementation of the tensor backend

This is very slow in comparison to the reference (numpy) implementaion.
It's purpose is to ensure the underlying tensor calculations are genuinely
understood. It also serves as a basis for more performant custom backends.
"""

from src.tensors.python_backend.tensor import PythonTensor
from src.tensors.axes import normalise_axes
from src.tensors.broadcasting import get_target_shape, get_target_strides
from src.tensors.reductions import (
    get_reduction_count,
    get_reduction_axes_and_target_shape,
    get_reduction_target_index,
)
from src.tensors.validation import (
    parse_tensor_data,
    validate_scalar_is_not_bool,
    validate_shape_has_no_negative_dimensions,
    validate_shape_not_rank_0,
    validate_tensor_conversion_root_is_sequence,
    validate_transpose_axes_are_permutation,
)
from typing import Callable, Sequence, Optional
from array import array
import math
import random


class PythonBackend:

    # Consider checking whether the tensor has default strides and offset zero
    # and, if it does, parsing the buffer direct. Profile first.
    @staticmethod
    def _map_unary(x: PythonTensor, op: Callable[[float], float]) -> PythonTensor:
        return PythonTensor(x.shape, array("d", (op(value) for _, value in x.items())))

    @staticmethod
    def _map_binary(
        a: PythonTensor,
        b: PythonTensor | float | int,
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

    @staticmethod
    def _divide_scalar(left: float, right: float) -> float:
        if right == 0.0:
            if left == 0.0:
                return math.nan
            sign = math.copysign(1.0, left) * math.copysign(1.0, right)
            return math.copysign(math.inf, sign)
        return left / right

    @staticmethod
    def _reduce_to_scalar(
        x: PythonTensor,
        initial_value: float,
        accumulate_fn: Callable[[float, float], float],
    ) -> float:
        accumulator = initial_value
        for _, value in x.items():
            accumulator = accumulate_fn(accumulator, value)
        return accumulator

    @staticmethod
    def _reduce_to_tensor(
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
            target_index = get_reduction_target_index(
                source_index, reduced_axes, keepdims
            )
            result.set_scalar(
                target_index, accumulate_fn(result.get_scalar(target_index), value)
            )
        return result

    @staticmethod
    def _reduce(
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
            return PythonBackend._reduce_to_scalar(x, initial_value, accumulate_fn)
        return PythonBackend._reduce_to_tensor(
            x, reduced_axes, target_shape, keepdims, initial_value, accumulate_fn
        )

    @staticmethod
    def _raise_if_reduction_has_no_values(
        shape: tuple[int, ...], reduced_axes: tuple[int, ...]
    ) -> None:
        if get_reduction_count(shape, reduced_axes) == 0:
            raise ValueError("reduction operation has no values")

    @staticmethod
    def _divide_reduction_result(
        result: PythonTensor | float, divisor: float
    ) -> PythonTensor | float:
        if isinstance(result, float):
            return result / divisor
        for index, value in result.items():
            result.set_scalar(index, value / divisor)
        return result

    @staticmethod
    def _log_scalar(value: float) -> float:
        """
        Treat log(0.0) as -inf because log values become more negative without
        limit as positive inputs get closer to zero.
        """
        if value == 0.0:
            return -math.inf
        if value < 0.0:
            return math.nan
        return math.log(value)

    @staticmethod
    def _sqrt_scalar(value: float) -> float:
        if value < 0.0:
            return math.nan
        return math.sqrt(value)

    @staticmethod
    def _sign_scalar(value: float) -> float:
        if value < 0.0:
            return -1.0
        if value > 0.0:
            return 1.0
        return 0.0

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

    def full(self, shape: tuple[int, ...], fill_value: float | int) -> PythonTensor:
        validate_scalar_is_not_bool(fill_value)
        return PythonTensor(shape, array("d", [float(fill_value)]) * math.prod(shape))

    def full_like(self, x: PythonTensor, fill_value: float | int) -> PythonTensor:
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
            validate_transpose_axes_are_permutation(normalised_axes, x.ndim())
        shape = tuple(x.shape[axis] for axis in normalised_axes)
        strides = tuple(x.strides[axis] for axis in normalised_axes)
        return x.view(shape, strides=strides)

    def add(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        return PythonBackend._map_binary(a, b, lambda left, right: left + right)

    def subtract(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        return PythonBackend._map_binary(a, b, lambda left, right: left - right)

    def multiply(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        return PythonBackend._map_binary(a, b, lambda left, right: left * right)

    def divide(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        return PythonBackend._map_binary(a, b, PythonBackend._divide_scalar)

    def matmul(self, a: PythonTensor, b: PythonTensor) -> PythonTensor:
        raise NotImplementedError

    def maximum(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        return PythonBackend._map_binary(a, b, max)

    def exp(self, x: PythonTensor) -> PythonTensor:
        return PythonBackend._map_unary(x, math.exp)

    def sum(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        return PythonBackend._reduce(
            x, axis, keepdims, 0.0, lambda total, value: total + value
        )

    def max(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        reduced_axes, _ = get_reduction_axes_and_target_shape(x.shape, axis, keepdims)
        PythonBackend._raise_if_reduction_has_no_values(x.shape, reduced_axes)
        return PythonBackend._reduce(x, axis, keepdims, -math.inf, max)

    def minimum(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        return PythonBackend._map_binary(a, b, min)

    def argmax(self, x: PythonTensor, axis: int | None = None) -> PythonTensor | int:
        raise NotImplementedError

    def log(self, x: PythonTensor) -> PythonTensor:
        return PythonBackend._map_unary(x, PythonBackend._log_scalar)

    def sqrt(self, x: PythonTensor) -> PythonTensor:
        return PythonBackend._map_unary(x, PythonBackend._sqrt_scalar)

    def absolute(self, x: PythonTensor) -> PythonTensor:
        return PythonBackend._map_unary(x, abs)

    def sign(self, x: PythonTensor) -> PythonTensor:
        return PythonBackend._map_unary(x, PythonBackend._sign_scalar)

    def clip(
        self, x: PythonTensor, min_value: float | int, max_value: float | int
    ) -> PythonTensor:
        validate_scalar_is_not_bool(min_value)
        validate_scalar_is_not_bool(max_value)
        return PythonBackend._map_unary(
            x, lambda value: min(max(value, min_value), max_value)
        )

    def mean(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        reduced_axes, _ = get_reduction_axes_and_target_shape(x.shape, axis, keepdims)
        PythonBackend._raise_if_reduction_has_no_values(x.shape, reduced_axes)
        total = PythonBackend._reduce(
            x, axis, keepdims, 0.0, lambda accumulator, value: accumulator + value
        )
        return PythonBackend._divide_reduction_result(
            total, float(get_reduction_count(x.shape, reduced_axes))
        )

    def min(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        reduced_axes, _ = get_reduction_axes_and_target_shape(x.shape, axis, keepdims)
        PythonBackend._raise_if_reduction_has_no_values(x.shape, reduced_axes)
        return PythonBackend._reduce(x, axis, keepdims, math.inf, min)

    def std(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        reduced_axes, _ = get_reduction_axes_and_target_shape(x.shape, axis, keepdims)
        PythonBackend._raise_if_reduction_has_no_values(x.shape, reduced_axes)
        mean = self.mean(x, axis=reduced_axes, keepdims=True)
        squared_deviations = PythonBackend._map_binary(
            x,
            mean,
            lambda value, mean_value: (value - mean_value) * (value - mean_value),
        )
        variance = self.mean(squared_deviations, axis=reduced_axes, keepdims=keepdims)
        if isinstance(variance, float):
            return math.sqrt(variance)
        return self.sqrt(variance)

    def stack(self, xs: Sequence[PythonTensor], axis: int = 0) -> PythonTensor:
        raise NotImplementedError

    def concatenate(self, xs: Sequence[PythonTensor], axis: int = 0) -> PythonTensor:
        raise NotImplementedError

    def eye(self, n: int, m: int | None = None) -> PythonTensor:
        if m is None:
            m = n
        tensor = PythonTensor((n, m))
        for i in range(min(n, m)):
            tensor.set_scalar((i, i), 1.0)
        return tensor
