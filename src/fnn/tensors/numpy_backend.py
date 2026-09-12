"""
The reference implementation for the tensor backend

numpy features heavily in teaching resources about neural networks,
including the NNfSiP book making it easy to write effective tests
for this implementation.
"""

import numpy as np
from typing import Sequence

from fnn.tensors.shared.axes import normalise_axis
from fnn.tensors.shared.reductions import get_reduction_axes_and_target_shape
from fnn.tensors.shared.types import Scalar
from fnn.tensors.shared.validation import (
    parse_tensor_data,
    validate_tensor_has_values,
    validate_scalar_is_not_bool,
    validate_shape_has_no_negative_dimensions,
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

    def _normalise_tensor_result(
        self, x: NumpyTensor | np.generic, dtype: type[float] | type[int]
    ) -> NumpyTensor:
        return np.asarray(x, dtype=dtype)

    def _validate_not_empty(self, x: NumpyTensor) -> None:
        if np.size(x) == 0:
            raise ValueError("This reduction method does not accept empty tensors.")

    def _validate_reduction_axes(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None,
        keepdims: bool,
    ) -> None:
        get_reduction_axes_and_target_shape(x.shape, axis, keepdims)

    def _validate_tensor_not_numpy_scalar(self, x: object) -> None:
        if isinstance(x, np.generic):
            raise ValueError("Backend methods do not accept NumPy scalar values.")

    def _validate_tensors_in_sequence_not_numpy_scalar(
        self, xs: Sequence[NumpyTensor]
    ) -> None:
        for x in xs:
            self._validate_tensor_not_numpy_scalar(x)

    def to_tensor(
        self, data: Scalar | list[object] | tuple[object, ...]
    ) -> NumpyTensor:
        parse_tensor_data(data)
        tensor = np.array(data, dtype=float)
        return tensor

    def to_python(self, tensor: NumpyTensor) -> Scalar | list:
        self._validate_tensor_not_numpy_scalar(tensor)
        result = self._normalise_scalar_result(tensor)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    def randn(self, shape: tuple[int, ...]) -> NumpyTensor:
        return self._random.standard_normal(size=shape)

    def zeros(self, shape: tuple[int, ...]) -> NumpyTensor:
        return np.zeros(shape, dtype=float)

    def ones(self, shape: tuple[int, ...]) -> NumpyTensor:
        return np.ones(shape, dtype=float)

    def ones_like(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return np.ones_like(x, dtype=float)

    def zeros_like(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return np.zeros_like(x, dtype=float)

    def full(self, shape: tuple[int, ...], fill_value: Scalar) -> NumpyTensor:
        validate_scalar_is_not_bool(fill_value)
        return np.full(shape, fill_value, dtype=float)

    def full_like(self, x: NumpyTensor, fill_value: Scalar) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        validate_scalar_is_not_bool(fill_value)
        return np.full_like(x, fill_value, dtype=float)

    def empty(self, shape: tuple[int, ...]) -> NumpyTensor:
        return np.empty(shape, dtype=float)

    def empty_like(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return np.empty_like(x, dtype=float)

    def copy(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return np.array(x, dtype=float, copy=True)

    def shape(self, x: NumpyTensor) -> tuple[int, ...]:
        self._validate_tensor_not_numpy_scalar(x)
        return x.shape

    def reshape(self, x: NumpyTensor, shape: tuple[int, ...]) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        validate_shape_has_no_negative_dimensions(shape, "reshape")
        return np.reshape(x, shape)

    def transpose(
        self, x: NumpyTensor, axes: tuple[int, ...] | None = None
    ) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return np.transpose(x, axes=axes)

    def add(self, a: NumpyTensor, b: NumpyTensor | Scalar) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(a)
        self._validate_tensor_not_numpy_scalar(b)
        validate_scalar_is_not_bool(b)
        return self._normalise_tensor_result(np.add(a, b), float)

    def subtract(self, a: NumpyTensor, b: NumpyTensor | Scalar) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(a)
        self._validate_tensor_not_numpy_scalar(b)
        validate_scalar_is_not_bool(b)
        return self._normalise_tensor_result(np.subtract(a, b), float)

    def multiply(self, a: NumpyTensor, b: NumpyTensor | Scalar) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(a)
        self._validate_tensor_not_numpy_scalar(b)
        validate_scalar_is_not_bool(b)
        return self._normalise_tensor_result(np.multiply(a, b), float)

    def divide(self, a: NumpyTensor, b: NumpyTensor | Scalar) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(a)
        self._validate_tensor_not_numpy_scalar(b)
        validate_scalar_is_not_bool(b)
        return self._normalise_tensor_result(np.divide(a, b), float)

    def matmul(self, a: NumpyTensor, b: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(a)
        self._validate_tensor_not_numpy_scalar(b)
        return self._normalise_tensor_result(np.matmul(a, b), float)

    def maximum(self, a: NumpyTensor, b: NumpyTensor | Scalar) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(a)
        self._validate_tensor_not_numpy_scalar(b)
        validate_scalar_is_not_bool(b)
        return self._normalise_tensor_result(np.maximum(a, b), float)

    def minimum(self, a: NumpyTensor, b: NumpyTensor | Scalar) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(a)
        self._validate_tensor_not_numpy_scalar(b)
        validate_scalar_is_not_bool(b)
        return self._normalise_tensor_result(np.minimum(a, b), float)

    def argmax(self, x: NumpyTensor, axis: int | None = None) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        validate_tensor_has_values(x.shape)
        if axis is not None:
            if type(axis) is not int:
                raise TypeError("axis must be an int or None")
            axis = normalise_axis(axis, len(x.shape))
        return self._normalise_tensor_result(np.argmax(x, axis=axis), int)

    def exp(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return self._normalise_tensor_result(np.exp(x), float)

    def log(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return self._normalise_tensor_result(np.log(x), float)

    def sqrt(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return self._normalise_tensor_result(np.sqrt(x), float)

    def absolute(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return self._normalise_tensor_result(np.absolute(x), float)

    def sign(self, x: NumpyTensor) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        return self._normalise_tensor_result(np.sign(x), float)

    def clip(self, x: NumpyTensor, min_value: Scalar, max_value: Scalar) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        validate_scalar_is_not_bool(min_value)
        validate_scalar_is_not_bool(max_value)
        return self._normalise_tensor_result(np.clip(x, min_value, max_value), float)

    def sum(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        self._validate_reduction_axes(x, axis, keepdims)
        return self._normalise_tensor_result(
            np.sum(x, axis=axis, keepdims=keepdims), float
        )

    def mean(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        self._validate_reduction_axes(x, axis, keepdims)
        self._validate_not_empty(x)
        return self._normalise_tensor_result(
            np.mean(x, axis=axis, keepdims=keepdims), float
        )

    def max(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        self._validate_reduction_axes(x, axis, keepdims)
        self._validate_not_empty(x)
        return self._normalise_tensor_result(
            np.max(x, axis=axis, keepdims=keepdims), float
        )

    def min(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        self._validate_reduction_axes(x, axis, keepdims)
        self._validate_not_empty(x)
        return self._normalise_tensor_result(
            np.min(x, axis=axis, keepdims=keepdims), float
        )

    def std(
        self,
        x: NumpyTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyTensor:
        self._validate_tensor_not_numpy_scalar(x)
        self._validate_reduction_axes(x, axis, keepdims)
        self._validate_not_empty(x)
        return self._normalise_tensor_result(
            np.std(x, axis=axis, keepdims=keepdims), float
        )

    def stack(self, xs: Sequence[NumpyTensor], axis: int = 0) -> NumpyTensor:
        self._validate_tensors_in_sequence_not_numpy_scalar(xs)
        return np.stack(xs, axis=axis)

    def concatenate(self, xs: Sequence[NumpyTensor], axis: int = 0) -> NumpyTensor:
        self._validate_tensors_in_sequence_not_numpy_scalar(xs)
        return np.concatenate(xs, axis=axis)

    def eye(self, n: int, m: int | None = None) -> NumpyTensor:
        return np.eye(n, m, dtype=float)
