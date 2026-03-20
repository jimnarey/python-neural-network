"""
The reference implementation for the tensor backend

numpy features heavily in teaching resources about neural networks,
including the NNfSiP book making it easy to write effective tests
for this implementation.
"""

import numpy as np

from src.tensors.backend import NonEmptyShape, Scalar, Tensor


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

    def _validate_not_rank_0(self, x: object) -> None:
        if isinstance(x, np.ndarray) and x.shape == ():
            raise ValueError("Backend methods do not accept rank 0 arrays.")
        if isinstance(x, np.generic):
            raise ValueError("Backend methods do not accept NumPy scalar values.")

    def _validate_not_rank_0_sequence(
        self, xs: tuple[Tensor, ...] | list[Tensor]
    ) -> None:
        for x in xs:
            self._validate_not_rank_0(x)

    def randn(self, shape: NonEmptyShape) -> Tensor:
        self._validate_non_empty_shape(shape)
        return self._random.standard_normal(size=shape)

    def zeros(self, shape: NonEmptyShape) -> Tensor:
        self._validate_non_empty_shape(shape)
        return np.zeros(shape)

    def ones(self, shape: NonEmptyShape) -> Tensor:
        self._validate_non_empty_shape(shape)
        return np.ones(shape)

    def ones_like(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.ones_like(x)

    def zeros_like(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.zeros_like(x)

    def full(self, shape: NonEmptyShape, fill_value: float | int) -> Tensor:
        self._validate_non_empty_shape(shape)
        return np.full(shape, fill_value)

    def full_like(self, x: Tensor, fill_value: float | int) -> Tensor:
        self._validate_not_rank_0(x)
        return np.full_like(x, fill_value)

    def empty(self, shape: NonEmptyShape) -> Tensor:
        self._validate_non_empty_shape(shape)
        return np.empty(shape)

    def empty_like(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.empty_like(x)

    def copy(self, x: Tensor) -> Tensor:
        self._validate_not_rank_0(x)
        return np.copy(x)

    def shape(self, x: Tensor) -> tuple[int, ...]:
        self._validate_not_rank_0(x)
        # Fix this once we pin down type checking for Tensor
        return x.shape  # type: ignore

    def reshape(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        self._validate_not_rank_0(x)
        self._validate_non_empty_shape(shape)
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
    ) -> Tensor | Scalar:
        self._validate_not_rank_0(x)
        return self._normalise_scalar_result(np.sum(x, axis=axis, keepdims=keepdims))

    def mean(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | Scalar:
        self._validate_not_rank_0(x)
        return self._normalise_scalar_result(np.mean(x, axis=axis, keepdims=keepdims))

    def max(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | Scalar:
        self._validate_not_rank_0(x)
        return self._normalise_scalar_result(np.max(x, axis=axis, keepdims=keepdims))

    def min(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | Scalar:
        self._validate_not_rank_0(x)
        return self._normalise_scalar_result(np.min(x, axis=axis, keepdims=keepdims))

    def std(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | Scalar:
        self._validate_not_rank_0(x)
        return self._normalise_scalar_result(np.std(x, axis=axis, keepdims=keepdims))

    def stack(self, xs: tuple[Tensor, ...] | list[Tensor], axis: int = 0) -> Tensor:
        self._validate_not_rank_0_sequence(xs)
        return np.stack(xs, axis=axis)

    def concatenate(
        self, xs: tuple[Tensor, ...] | list[Tensor], axis: int = 0
    ) -> Tensor:
        self._validate_not_rank_0_sequence(xs)
        return np.concatenate(xs, axis=axis)

    def vstack(self, xs: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        self._validate_not_rank_0_sequence(xs)
        return np.vstack(xs)

    def hstack(self, xs: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        self._validate_not_rank_0_sequence(xs)
        return np.hstack(xs)

    def eye(self, n: int, m: int | None = None) -> Tensor:
        return np.eye(n, m)
