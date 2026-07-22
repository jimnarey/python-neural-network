"""Pure Python implementation of the tensor backend

This is very slow in comparison to the reference (numpy) implementaion.
It's purpose is to ensure the underlying tensor calculations are genuinely
understood. It also serves as a basis for more performant custom backends.
"""

from src.tensors.python_backend.tensor import PythonTensor
from src.tensors.validation import (
    parse_tensor_data,
    validate_tensor_conversion_input,
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
        validate_tensor_conversion_input(data)
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

    def reshape(self, x: PythonTensor, shape: tuple[int, ...]) -> PythonTensor:
        raise NotImplementedError

    def transpose(
        self, x: PythonTensor, axes: tuple[int, ...] | None = None
    ) -> PythonTensor:
        raise NotImplementedError

    def add(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        raise NotImplementedError

    def subtract(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        raise NotImplementedError

    def multiply(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        raise NotImplementedError

    def divide(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        raise NotImplementedError

    def matmul(self, a: PythonTensor, b: PythonTensor) -> PythonTensor:
        raise NotImplementedError

    def maximum(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        raise NotImplementedError

    def exp(self, x: PythonTensor) -> PythonTensor:
        raise NotImplementedError

    def sum(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        raise NotImplementedError

    def max(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        raise NotImplementedError

    def minimum(self, a: PythonTensor, b: PythonTensor | float | int) -> PythonTensor:
        raise NotImplementedError

    def argmax(self, x: PythonTensor, axis: int | None = None) -> PythonTensor | int:
        raise NotImplementedError

    def log(self, x: PythonTensor) -> PythonTensor:
        raise NotImplementedError

    def sqrt(self, x: PythonTensor) -> PythonTensor:
        raise NotImplementedError

    def absolute(self, x: PythonTensor) -> PythonTensor:
        raise NotImplementedError

    def sign(self, x: PythonTensor) -> PythonTensor:
        raise NotImplementedError

    def clip(
        self, x: PythonTensor, min_value: float | int, max_value: float | int
    ) -> PythonTensor:
        raise NotImplementedError

    def mean(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        raise NotImplementedError

    def min(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        raise NotImplementedError

    def std(
        self,
        x: PythonTensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> PythonTensor | float:
        raise NotImplementedError

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
