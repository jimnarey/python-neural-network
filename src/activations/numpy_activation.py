"""The reference implementation for activation functions

Numpy was chosen as the reference for the same reason it is used
as such for the tensor backends.
"""

from src.activations.activation import Activation
from src.tensors import TensorBackend


class ReLU[T](Activation[T]):
    def forward(self, backend: TensorBackend[T], x: T) -> T:
        return backend.maximum(x, 0)


class Softmax[T](Activation[T]):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, backend: TensorBackend[T], x: T) -> T:
        safe_x = backend.subtract(x, backend.max(x, axis=self.axis, keepdims=True))
        exp_values = backend.exp(safe_x)
        norm_base = backend.sum(exp_values, axis=self.axis, keepdims=True)
        return backend.divide(exp_values, norm_base)
