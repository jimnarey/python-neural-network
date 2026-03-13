"""The reference implementation for activation functions

Numpy was chosen as the reference for the same reason it is used
as such for the tensor backends.
"""

from src.activations.base import Activation
from src.tensors import Tensor, TensorBackend


class ReLU(Activation):
    def forward(self, backend: TensorBackend, x: Tensor) -> Tensor:
        return backend.maximum(x, 0)


class Softmax(Activation):
    def forward(self, backend: TensorBackend, x: Tensor) -> Tensor:
        safe_x = backend.subtract(x, backend.max(x, axis=1, keepdims=True))
        exp_values = backend.exp(safe_x)
        norm_base = backend.sum(exp_values, axis=1, keepdims=True)
        return backend.divide(exp_values, norm_base)
