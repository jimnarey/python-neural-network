from typing import Sequence

from src.tensors.python_backend.python_tensor import PythonTensor


def require_non_empty_tensor_sequence(
    xs: Sequence[PythonTensor],
) -> tuple[PythonTensor, ...]:
    tensors = tuple(xs)
    if not tensors:
        raise ValueError("tensor sequence must not be empty")
    return tensors


def validate_stack_shapes(xs: tuple[PythonTensor, ...]) -> None:
    base_shape = xs[0].shape
    for tensor in xs[1:]:
        if tensor.shape != base_shape:
            raise ValueError("all tensors must have the same shape")
