from typing import Protocol, runtime_checkable

from src.tensors import Tensor, TensorBackend


@runtime_checkable
class Activation(Protocol):
    def forward(self, backend: TensorBackend, x: Tensor) -> Tensor:
        """
        Apply the activation to a backend-native tensor and return a tensor
        belonging to the same backend.
        """
