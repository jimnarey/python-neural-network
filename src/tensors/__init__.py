"""Manage imports of the backend classes

The idea is that eventually it will be possible to run a version
of the network using a pure Python backend. It should therefore be
able to run without NumPy being installed. This module prevents an
exception being thrown due to NumPy's absence as long as nothing
being executed actually imports the NumpyBackend.
"""

from typing import TYPE_CHECKING

from src.tensors.tensor_backend import Tensor, TensorBackend

# This is needed to deal with a subtle problem whereby importing
# NumpyBackend using the __getattr__ approach below prevented mypy
# from fully checking that NumpyBackend conformed to TensorBackend.
# This can be fixed where needed by importing it from its module
# rather than as a package attribute. However, the if TYPE_CHECKING
# block handles this globally
if TYPE_CHECKING:
    from src.tensors.numpy_backend import NumpyBackend

__all__ = ["Tensor", "TensorBackend", "NumpyBackend"]


def __getattr__(name: str):
    if name == "NumpyBackend":
        try:
            from src.tensors.numpy_backend import NumpyBackend
        except ImportError as exc:
            raise RuntimeError("NumpyBackend requires numpy to be installed.") from exc
        return NumpyBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
