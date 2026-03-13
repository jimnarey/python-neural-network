import numpy as np

from src.tensors.backend import Tensor


class NumpyBackend:
    def __init__(self, seed: int | None = None):
        self._random = np.random.default_rng(seed)

    def randn(self, shape: tuple[int, ...]) -> Tensor:
        return self._random.standard_normal(size=shape)

    def zeros(self, shape: tuple[int, ...]) -> Tensor:
        return np.zeros(shape)

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

    def exp(self, x: Tensor) -> Tensor:
        return np.exp(x)

    def sum(self, x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def max(self, x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return np.max(x, axis=axis, keepdims=keepdims)
