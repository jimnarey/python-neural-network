"""
The reference implementation for the tensor backend

numpy features heavily in teaching resources about neural networks,
including the NNfSiP book making it easy to write effective tests
for this implementation.
"""

import numpy as np

from src.tensors.backend import Tensor


class NumpyBackend:
    def __init__(self, seed: int | None = None):
        self._random = np.random.default_rng(seed)

    def randn(self, shape: tuple[int, ...]) -> Tensor:
        return self._random.standard_normal(size=shape)

    def zeros(self, shape: tuple[int, ...]) -> Tensor:
        return np.zeros(shape)

    def ones(self, shape: tuple[int, ...]) -> Tensor:
        return np.ones(shape)

    def ones_like(self, x: Tensor) -> Tensor:
        return np.ones_like(x)

    def zeros_like(self, x: Tensor) -> Tensor:
        return np.zeros_like(x)

    def full(self, shape: tuple[int, ...], fill_value: float | int) -> Tensor:
        return np.full(shape, fill_value)

    def full_like(self, x: Tensor, fill_value: float | int) -> Tensor:
        return np.full_like(x, fill_value)

    def empty(self, shape: tuple[int, ...]) -> Tensor:
        return np.empty(shape)

    def empty_like(self, x: Tensor) -> Tensor:
        return np.empty_like(x)

    def copy(self, x: Tensor) -> Tensor:
        return np.copy(x)

    def shape(self, x: Tensor) -> tuple[int, ...]:
        return x.shape

    def reshape(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        return np.reshape(x, shape)

    def transpose(self, x: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
        return np.transpose(x, axes=axes)

    def add(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return np.add(a, b)

    def subtract(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return np.subtract(a, b)

    def multiply(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return np.multiply(a, b)

    def divide(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return np.divide(a, b)

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        return np.matmul(a, b)

    def maximum(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return np.maximum(a, b)

    def minimum(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return np.minimum(a, b)

    def argmax(self, x: Tensor, axis: int | None = None) -> Tensor:
        return np.argmax(x, axis=axis)

    def exp(self, x: Tensor) -> Tensor:
        return np.exp(x)

    def log(self, x: Tensor) -> Tensor:
        return np.log(x)

    def sqrt(self, x: Tensor) -> Tensor:
        return np.sqrt(x)

    def absolute(self, x: Tensor) -> Tensor:
        return np.absolute(x)

    def sign(self, x: Tensor) -> Tensor:
        return np.sign(x)

    def clip(self, x: Tensor, min_value: float | int, max_value: float | int) -> Tensor:
        return np.clip(x, min_value, max_value)

    def sum(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return np.max(x, axis=axis, keepdims=keepdims)

    def min(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return np.min(x, axis=axis, keepdims=keepdims)

    def std(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return np.std(x, axis=axis, keepdims=keepdims)

    def stack(self, xs: tuple[Tensor, ...] | list[Tensor], axis: int = 0) -> Tensor:
        return np.stack(xs, axis=axis)

    def concatenate(
        self, xs: tuple[Tensor, ...] | list[Tensor], axis: int = 0
    ) -> Tensor:
        return np.concatenate(xs, axis=axis)

    def vstack(self, xs: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        return np.vstack(xs)

    def hstack(self, xs: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        return np.hstack(xs)

    def eye(self, n: int, m: int | None = None) -> Tensor:
        return np.eye(n, m)
