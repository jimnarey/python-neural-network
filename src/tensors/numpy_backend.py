"""
The reference implementation for the tensor backend

numpy features heavily in teaching resources about neural networks,
including the NNfSiP book making it easy to write effective tests
for this implementation.
"""

import numpy as np
from typing import Sequence

from src.tensors.tensor_backend import NonEmptyShape, Scalar, Tensor


class NumpyBackend:
    def __init__(self, seed: int | None = None):
        # - Set self._random to a generator which is then used to generate
        # random values (or not-random, if we provide a seed value). This
        # is preferable to setting the global NumPy seed, which will affect
        # anything else in the program which uses NumPy's randn method.
        # - standard_normal defaults to float64 which is the same precision
        # as a Python float.
        self._random = np.random.default_rng(seed)

    def _validate_non_empty_shape(self, shape: tuple[int, ...]) -> None:
        if not shape:
            raise ValueError("Tensor creation methods require a non-empty shape.")

    def _normalise_scalar_result(self, x: Tensor | Scalar) -> Tensor | Scalar:
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

    def _normalise_float_scalar_result(self, x: Tensor | Scalar) -> Tensor | float:
        result = self._normalise_scalar_result(x)
        if isinstance(result, np.ndarray):
            return result
        # The logic is sound here but that's not enough for mypy. Revisit
        # once we have pinned down the Tensor type
        return float(result)  # type: ignore

    def _validate_not_rank_0(self, x: object) -> None:
        if isinstance(x, np.ndarray) and x.shape == ():
            raise ValueError("Backend methods do not accept rank 0 arrays.")
        if isinstance(x, np.generic):
            raise ValueError("Backend methods do not accept NumPy scalar values.")

    def _validate_not_rank_0_sequence(self, xs: Sequence[Tensor]) -> None:
        for x in xs:
            self._validate_not_rank_0(x)

    def _validate_tensor_input_values(self, data: object) -> None:
        """
        Check that values are a valid type. It is limited and relies on
        NumPy doing some of the work for us. It will not catch cases like:

        [[(), 0], 1, 2]
        [[(1,), 0], 1, 2]

        which are both invalid. Nor is it safe to use unless conversion of
        ints to floats happens elsewhere. Again, in the case of NumPy that
        happens if we pass the right dtype when calling np.array.

        We can't just check that the type of incoming values is float or
        int because bool is a subclass of int. So it gets special treatment.
        """
        if isinstance(data, (list, tuple)):
            for item in data:
                self._validate_tensor_input_values(item)
            return
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ValueError("Tensor conversion requires numeric values.")

    def to_tensor(self, data: list[object] | tuple[object, ...]) -> Tensor:
        if not isinstance(data, (list, tuple)):
            raise ValueError("Tensor conversion requires a list or tuple input.")
        self._validate_tensor_input_values(data)
        tensor = np.array(data, dtype=float)
        self._validate_not_rank_0(tensor)
        return tensor

    def to_python(self, tensor: Tensor) -> object:
        self._validate_not_rank_0(tensor)
        assert isinstance(
            tensor, np.ndarray
        )  # Remove this once we've properly tackled typing of tensors
        return tensor.tolist()

    def randn(self, shape: NonEmptyShape) -> Tensor:
        self._validate_non_empty_shape(shape)
        return self._random.standard_normal(size=shape)

    def zeros(self, shape: NonEmptyShape) -> Tensor:
        self._validate_non_empty_shape(shape)
        return np.zeros(shape, dtype=float)

    def ones(self, shape: NonEmptyShape) -> Tensor:
        self._validate_non_empty_shape(shape)
        return np.ones(shape, dtype=float)

    def ones_like(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.ones_like(x, dtype=float)

    def zeros_like(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.zeros_like(x, dtype=float)

    def full(self, shape: NonEmptyShape, fill_value: float | int) -> Tensor:
        self._validate_non_empty_shape(shape)
        return np.full(shape, fill_value, dtype=float)

    def full_like(self, x: Tensor, fill_value: float | int) -> Tensor:
        self._validate_not_rank_0(x)
        return np.full_like(x, fill_value, dtype=float)

    def empty(self, shape: NonEmptyShape) -> Tensor:
        self._validate_non_empty_shape(shape)
        return np.empty(shape, dtype=float)

    def empty_like(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.empty_like(x, dtype=float)

    def copy(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.array(x, dtype=float, copy=True)

    def shape(self, x: Tensor) -> tuple[int, ...]:
        self._validate_not_rank_0(x)
        # Fix this once we pin down type checking for Tensor
        return x.shape  # type: ignore

    def reshape(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        self._validate_not_rank_0(x)
        self._validate_non_empty_shape(shape)
        if any(dimension < 0 for dimension in shape):
            raise ValueError(
                "reshape does not support negative values in the target shape"
            )
        return np.reshape(x, shape)

    def transpose(self, x: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
        self._validate_not_rank_0(x)
        return np.transpose(x, axes=axes)

    def add(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        self._validate_not_rank_0(a)
        self._validate_not_rank_0(b)
        return np.add(a, b)

    def subtract(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        self._validate_not_rank_0(a)
        self._validate_not_rank_0(b)
        return np.subtract(a, b)

    def multiply(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        self._validate_not_rank_0(a)
        self._validate_not_rank_0(b)
        return np.multiply(a, b)

    def divide(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        self._validate_not_rank_0(a)
        self._validate_not_rank_0(b)
        return np.divide(a, b)

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        self._validate_not_rank_0(a)
        self._validate_not_rank_0(b)
        return np.matmul(a, b)

    def maximum(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        self._validate_not_rank_0(a)
        self._validate_not_rank_0(b)
        return np.maximum(a, b)

    def minimum(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        self._validate_not_rank_0(a)
        self._validate_not_rank_0(b)
        return np.minimum(a, b)

    def argmax(self, x: Tensor, axis: int | None = None) -> Tensor | int:
        self._validate_not_rank_0(x)
        return self._normalise_scalar_result(np.argmax(x, axis=axis))

    def exp(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.exp(x)

    def log(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.log(x)

    def sqrt(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.sqrt(x)

    def absolute(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.absolute(x)

    def sign(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.sign(x)

    def clip(self, x: Tensor, min_value: float | int, max_value: float | int) -> Tensor:
        self._validate_not_rank_0(x)
        return np.clip(x, min_value, max_value)

    def sum(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | float:
        self._validate_not_rank_0(x)
        return self._normalise_float_scalar_result(
            np.sum(x, axis=axis, keepdims=keepdims)
        )

    def mean(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | float:
        self._validate_not_rank_0(x)
        return self._normalise_float_scalar_result(
            np.mean(x, axis=axis, keepdims=keepdims)
        )

    def max(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | float:
        self._validate_not_rank_0(x)
        return self._normalise_float_scalar_result(
            np.max(x, axis=axis, keepdims=keepdims)
        )

    def min(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | float:
        self._validate_not_rank_0(x)
        return self._normalise_float_scalar_result(
            np.min(x, axis=axis, keepdims=keepdims)
        )

    def std(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | float:
        self._validate_not_rank_0(x)
        return self._normalise_float_scalar_result(
            np.std(x, axis=axis, keepdims=keepdims)
        )

    def stack(self, xs: Sequence[Tensor], axis: int = 0) -> Tensor:
        self._validate_not_rank_0_sequence(xs)
        return np.stack(xs, axis=axis)

    def concatenate(self, xs: Sequence[Tensor], axis: int = 0) -> Tensor:
        self._validate_not_rank_0_sequence(xs)
        return np.concatenate(xs, axis=axis)

    def vstack(self, xs: Sequence[Tensor]) -> Tensor:
        self._validate_not_rank_0_sequence(xs)
        return np.vstack(xs)

    def hstack(self, xs: Sequence[Tensor]) -> Tensor:
        self._validate_not_rank_0_sequence(xs)
        return np.hstack(xs)

    def eye(self, n: int, m: int | None = None) -> Tensor:
        return np.eye(n, m, dtype=float)
