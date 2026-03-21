from src.tensors.tensor_backend import Tensor, TensorBackend

__all__ = ["Tensor", "TensorBackend", "NumpyBackend"]


def __getattr__(name: str):
    if name == "NumpyBackend":
        try:
            from src.tensors.numpy_backend import NumpyBackend
        except ImportError as exc:
            raise RuntimeError("NumpyBackend requires numpy to be installed.") from exc
        return NumpyBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
