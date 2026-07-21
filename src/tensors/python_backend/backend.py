"""Pure Python implementation of the tensor backend

This is very slow in comparison to the reference (numpy) implementaion.
It's purpose is to ensure the underlying tensor calculations are genuinely
understood. It also serves as a basis for more performant custom backends.
"""

from src.tensors.tensor_backend import Tensor, TensorBackend
from src.tensors.python_backend.tensor import PythonTensor
from src.tensors.validation import (
    parse_tensor_data,
    validate_tensor_conversion_input,
)
from typing import Sequence, Optional
from array import array
import math
import random


class PythonBackend(TensorBackend):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            # This is global. Needs to be changed, maybe self._random = random.Random(seed)
            # ...though it would be nice to avoid runtime imports altogether...
            random.seed(seed)

    # A temporary measure until we tighten the type annotations in the Protocol
    # and the classes conforming to it.
    def _require_python_tensor(self, x: Tensor, method_name: str) -> PythonTensor:
        if not isinstance(x, PythonTensor):
            raise TypeError(f"{method_name} requires a PythonTensor input.")
        return x

    def to_tensor(self, data: list[object] | tuple[object, ...]) -> Tensor:
        validate_tensor_conversion_input(data)
        shape, values = parse_tensor_data(data)
        return PythonTensor(shape, array("d", values))

    def to_python(self, tensor: Tensor) -> list:
        tensor = self._require_python_tensor(tensor, "to_python")
        return tensor.to_list()

    def randn(self, shape: tuple[int, ...]) -> Tensor:
        return None

    def zeros(self, shape: tuple[int, ...]) -> Tensor:
        return PythonTensor(shape)

    def zeros_like(self, x: Tensor) -> Tensor:
        x = self._require_python_tensor(x, "zeros_like")
        return self.zeros(x.shape)

    def ones(self, shape: tuple[int, ...]) -> Tensor:
        return PythonTensor(shape, array("d", [1.0]) * math.prod(shape))

    def ones_like(self, x: Tensor) -> Tensor:
        x = self._require_python_tensor(x, "ones_like")
        return self.ones(x.shape)

    def full(self, shape: tuple[int, ...], fill_value: float | int) -> Tensor:
        return PythonTensor(shape, array("d", [float(fill_value)]) * math.prod(shape))

    def full_like(self, x: Tensor, fill_value: float | int) -> Tensor:
        x = self._require_python_tensor(x, "full_like")
        return self.full(x.shape, fill_value)

    def empty(self, shape: tuple[int, ...]) -> Tensor:
        return PythonTensor(shape)

    def empty_like(self, x: Tensor) -> Tensor:
        x = self._require_python_tensor(x, "empty_like")
        return PythonTensor(x.shape)

    def copy(self, x: Tensor) -> Tensor:
        return None

    def shape(self, x: Tensor) -> tuple[int, ...]:
        x = self._require_python_tensor(x, "shape")
        return x.shape

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

    def eye(self, n: int, m: int | None = None) -> Tensor:
        if m is None:
            m = n
        tensor = PythonTensor((n, m))
        for i in range(min(n, m)):
            tensor.set_scalar((i, i), 1.0)
        return tensor
