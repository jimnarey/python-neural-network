"""Interface for neuron activation functions

We're using classes instead of functions because some state
is properly the activation function rather than the layer/neuron
using it
"""

from typing import Protocol, runtime_checkable

from src.tensors import Tensor, TensorBackend


@runtime_checkable
class Activation(Protocol):
    def forward(self, backend: TensorBackend, x: Tensor) -> Tensor:
        """
        Apply the activation to a backend-native tensor and return a tensor
        belonging to the same backend.
        """
