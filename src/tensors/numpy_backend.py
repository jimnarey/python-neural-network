"""
The reference implementation for the tensor backend

numpy features heavily in teaching resources about neural networks,
including the NNfSiP book making it easy to write effective tests
for this implementation.
"""

import numpy as np
from typing import Sequence

from src.tensors.tensor_backend import Scalar
from src.tensors.validation import (
    parse_tensor_data,
    validate_shape_has_no_negative_dimensions,
    validate_shape_not_rank_0,
    validate_tensor_conversion_input,
)

type NumpyTensor = np.ndarray


class NumpyBackend:
    def __init__(self, seed: int | None = None):
        # - Set self._random to a generator which is then used to generate
        # random values (or not-random, if we provide a seed value). This
        # is preferable to setting the global NumPy seed, which will affect
        # anything else in the program which uses NumPy's randn method.
        # - standard_normal defaults to float64 which is the same precision
        # as a Python float.
        self._random = np.random.default_rng(seed)

    def _normalise_scalar_result(self, x: NumpyTensor | Scalar) -> NumpyTensor | Scalar:
        if isinstance(x, np.ndarray) and x.shape == ():
            # Convert zero rank arrays containing a scalar to a simple scalar
            return x.item()
        if isinstance(x, np.generic):
            # There is a small performance cost here as this forces conversion to a
            # Python type from a C/NumPy type (e.g. np.int64) which could be avoided
            # if subsequent operations only use NumPy. But it can't be avoided if we
            # want a consistent backend contract
            return x.item()
        return x

    def _normalise_float_scalar_result(
        self, x: NumpyTensor | Scalar
    ) -> NumpyTensor | float:
        result = self._normalise_scalar_result(x)
        if isinstance(result, np.ndarray):
            return result
        return float(result)

    def _validate_tensor_not_rank_0(self, x: object) -> None:
        if isinstance(x, np.ndarray) and x.shape == ():
            raise ValueError("Backend methods do not accept rank 0 arrays.")
        if isinstance(x, np.generic):
            raise ValueError("Backend methods do not accept NumPy scalar values.")

    def _validate_not_empty(self, x: NumpyTensor) -> None:
        if np.size(x) == 0:
            raise ValueError("This reduction method does not accept empty tensors.")

    def _validate_tensors_in_sequence_not_rank_0(
        self, xs: Sequence[NumpyTensor]
    ) -> None:
        for x in xs:
            self._validate_tensor_not_rank_0(x)

    def to_tensor(self, data: list[object] | tuple[object, ...]) -> NumpyTensor:
        validate_tensor_conversion_input(data)
        parse_tensor_data(data)
        tensor = np.array(data, dtype=float)
        self._validate_tensor_not_rank_0(tensor)
        return tensor

    def to_python(self, tensor: NumpyTensor) -> list:
        self._validate_tensor_not_rank_0(tensor)
        return tensor.tolist()

    def randn(self, shape: tuple[int, ...]) -> NumpyTensor:
        validate_shape_not_rank_0(shape)
        return self._random.standard_normal(size=shape)

    def zeros(self, shape: tuple[int, ...]) -> NumpyTensor:
        validate_shape_not_rank_0(shape)
        return np.zeros(shape, dtype=float)

    def ones(self, shape: tuple[int, ...]) -> NumpyTensor:
        validate_shape_not_rank_0(shape)
        return np.ones(shape, dtype=float)

    def ones_like(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.ones_like(x, dtype=float)

    def zeros_like(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.zeros_like(x, dtype=float)

    def full(self, shape: tuple[int, ...], fill_value: float | int) -> NumpyTensor:
        validate_shape_not_rank_0(shape)
        return np.full(shape, fill_value, dtype=float)

    def full_like(self, x: NumpyTensor, fill_value: float | int) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.full_like(x, fill_value, dtype=float)

    def empty(self, shape: tuple[int, ...]) -> NumpyTensor:
        validate_shape_not_rank_0(shape)
        return np.empty(shape, dtype=float)

    def empty_like(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.empty_like(x, dtype=float)

    def copy(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.array(x, dtype=float, copy=True)

    def shape(self, x: NumpyTensor) -> tuple[int, ...]:
        self._validate_tensor_not_rank_0(x)
        return x.shape

    def reshape(self, x: NumpyTensor, shape: tuple[int, ...]) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        validate_shape_not_rank_0(shape)
        validate_shape_has_no_negative_dimensions(shape, "reshape")
        return np.reshape(x, shape)

    def transpose(
        self, x: NumpyTensor, axes: tuple[int, ...] | None = None
    ) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.transpose(x, axes=axes)

    def add(self, a: NumpyTensor, b: NumpyTensor | float | int) -> NumpyTensor:
        self._validate_tensor_not_rank_0(a)
        self._validate_tensor_not_rank_0(b)
        return np.add(a, b)

    def subtract(self, a: NumpyTensor, b: NumpyTensor | float | int) -> NumpyTensor:
        self._validate_tensor_not_rank_0(a)
        self._validate_tensor_not_rank_0(b)
        return np.subtract(a, b)

    def multiply(self, a: NumpyTensor, b: NumpyTensor | float | int) -> NumpyTensor:
        self._validate_tensor_not_rank_0(a)
        self._validate_tensor_not_rank_0(b)
        return np.multiply(a, b)

    def divide(self, a: NumpyTensor, b: NumpyTensor | float | int) -> NumpyTensor:
        self._validate_tensor_not_rank_0(a)
        self._validate_tensor_not_rank_0(b)
        return np.divide(a, b)

    def matmul(self, a: NumpyTensor, b: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(a)
        self._validate_tensor_not_rank_0(b)
        return np.matmul(a, b)

    def maximum(self, a: NumpyTensor, b: NumpyTensor | float | int) -> NumpyTensor:
        self._validate_tensor_not_rank_0(a)
        self._validate_tensor_not_rank_0(b)
        return np.maximum(a, b)

    def minimum(self, a: NumpyTensor, b: NumpyTensor | float | int) -> NumpyTensor:
        self._validate_tensor_not_rank_0(a)
        self._validate_tensor_not_rank_0(b)
        return np.minimum(a, b)

    def argmax(self, x: NumpyTensor, axis: int | None = None) -> NumpyTensor | int:
        self._validate_tensor_not_rank_0(x)
        return self._normalise_scalar_result(np.argmax(x, axis=axis))

    def exp(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.exp(x)

    def log(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.log(x)

    def sqrt(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.sqrt(x)

    def absolute(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.absolute(x)

    def sign(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.sign(x)

    def clip(
        self, x: NumpyTensor, min_value: float | int, max_value: float | int
    ) -> NumpyTensor:
        self._validate_tensor_not_rank_0(x)
        return np.clip(x, min_value, max_value)

    def sum(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor | float:
        self._validate_tensor_not_rank_0(x)
        return self._normalise_float_scalar_result(
            np.sum(x, axis=axis, keepdims=keepdims)
        )

    def mean(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor | float:
        self._validate_tensor_not_rank_0(x)
        self._validate_not_empty(x)
        return self._normalise_float_scalar_result(
            np.mean(x, axis=axis, keepdims=keepdims)
        )

    def max(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor | float:
        self._validate_tensor_not_rank_0(x)
        self._validate_not_empty(x)
        return self._normalise_float_scalar_result(
            np.max(x, axis=axis, keepdims=keepdims)
        )

    def min(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor | float:
        self._validate_tensor_not_rank_0(x)
        self._validate_not_empty(x)
        return self._normalise_float_scalar_result(
            np.min(x, axis=axis, keepdims=keepdims)
        )

    def std(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor | float:
        self._validate_tensor_not_rank_0(x)
        self._validate_not_empty(x)
        return self._normalise_float_scalar_result(
            np.std(x, axis=axis, keepdims=keepdims)
        )

    def stack(self, xs: Sequence[NumpyTensor], axis: int = 0) -> NumpyTensor:
        self._validate_tensors_in_sequence_not_rank_0(xs)
        return np.stack(xs, axis=axis)

    def concatenate(self, xs: Sequence[NumpyTensor], axis: int = 0) -> NumpyTensor:
        self._validate_tensors_in_sequence_not_rank_0(xs)
        return np.concatenate(xs, axis=axis)

    def eye(self, n: int, m: int | None = None) -> NumpyTensor:
        return np.eye(n, m, dtype=float)
