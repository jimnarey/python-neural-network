"""Pure Python implementation of the tensor backend

This is very slow in comparison to the reference (numpy) implementaion.
It's purpose is to ensure the underlying tensor calculations are genuinely
understood. It also serves as a basis for more performant custom backends.
"""

from src.tensors.backend import Tensor, TensorBackend
from typing import Sequence, Optional
import random


class PythonBackend(TensorBackend):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            # This is global. Needs to be changed, maybe self._random = random.Random(seed)
            # ...though it would be nice to avoid runtime imports altogether...
            random.seed(seed)

    def randn(self, shape: tuple[int, ...]) -> Tensor:
        return None

    def zeros(self, shape: tuple[int, ...]) -> Tensor:
        return None

    def zeros_like(self, x: Tensor) -> Tensor:
        return None

    def ones(self, shape: tuple[int, ...]) -> Tensor:
        return None

    def ones_like(self, x: Tensor) -> Tensor:
        return None

    def full(self, shape: tuple[int, ...], fill_value: float | int) -> Tensor:
        return None

    def full_like(self, x: Tensor, fill_value: float | int) -> Tensor:
        return None

    def empty(self, shape: tuple[int, ...]) -> Tensor:
        return None

    def empty_like(self, x: Tensor) -> Tensor:
        return None

    def copy(self, x: Tensor) -> Tensor:
        return None

    def shape(self, x: Tensor) -> tuple[int, ...]:
        return ()

    def reshape(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        return None

    def transpose(self, x: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
        return None

    def add(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return None

    def subtract(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return None

    def multiply(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return None

    def divide(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return None

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        return None

    def maximum(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return None

    def exp(self, x: Tensor) -> Tensor:
        return None

    def sum(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return None

    def max(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return None

    def minimum(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        return None

    def argmax(self, x: Tensor, axis: int | None = None) -> Tensor | int:
        return None

    def log(self, x: Tensor) -> Tensor:
        return None

    def sqrt(self, x: Tensor) -> Tensor:
        return None

    def absolute(self, x: Tensor) -> Tensor:
        return None

    def sign(self, x: Tensor) -> Tensor:
        return None

    def clip(self, x: Tensor, min_value: float | int, max_value: float | int) -> Tensor:
        return None

    def mean(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | float:
        return None

    def min(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | float:
        return None

    def std(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor | float:
        return None

    def stack(self, xs: Sequence[Tensor], axis: int = 0) -> Tensor:
        return None

    def concatenate(self, xs: Sequence[Tensor], axis: int = 0) -> Tensor:
        return None

    def vstack(self, xs: Sequence[Tensor]) -> Tensor:
        return None

    def hstack(self, xs: Sequence[Tensor]) -> Tensor:
        return None

    def eye(self, n: int, m: int | None = None) -> Tensor:
        return None
