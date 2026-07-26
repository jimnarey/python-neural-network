"""Shared broadcasting shape and stride arithmetic

Provides the shape-resolution and stride-resolution logic needed to combine
two differently-shaped tensors wherever broadcasting applies: computing the
broadcast target shape for a pair of operand shapes, and computing the
strides that let a single operand be viewed as if it had that target shape.

These functions operate on whatever shape (or portion of a shape) they are
given, so callers that only broadcast some of a tensor's axes — such as the
leading axes in a matmul, where the trailing two axes follow a different,
non-broadcasting compatibility rule — can slice out just the axes that
broadcasting applies to before calling them.

Padding is carried out explicitly (see left_pad) rather than via
itertools.zip_longest, so backends written in other languages (e.g. C or
Cython) can reproduce the same logic without relying on a Python-specific
convenience.

Shapes are assumed to already be validated by the time they reach this
module. In particular, rank-0 shapes are not guarded against here; elsewhere
in this project scalars are represented as plain floats/ints rather than
rank-0 tensors, so a shape of () is not expected to reach these functions.
"""


def left_pad(
    values: tuple[int, ...], target_rank: int, fill_value: int
) -> tuple[int, ...]:
    """
    Return a tuple with fill_value prepended to the contents of the values
    tuple until it reaches target_rank.

    Used to align a shape or strides tuple to a longer rank before comparing
    it axis by axis against another tensor's shape or strides.
    """
    if target_rank < len(values):
        raise ValueError("target rank must be at least the rank of values")
    return (fill_value,) * (target_rank - len(values)) + values


def get_target_dimension(a_dim: int, b_dim: int) -> int:
    """
    Return the axis length produced when two axis lengths are broadcast
    together.

    The two lengths are compatible if they are equal, or if one of them is
    1, in which case the other length is returned. Any other combination is
    not broadcast compatible and raises.
    """
    if a_dim == b_dim or b_dim == 1:
        return a_dim
    if a_dim == 1:
        return b_dim
    raise ValueError("shapes are not broadcast compatible")


def get_target_stride(source_dim: int, stride: int, target_dim: int) -> int:
    """
    Return the stride an axis needs to broadcast from source_dim to target_dim.

    If the axis is already the required length, its existing stride is
    reused unchanged. If it needs to be stretched from length 1, the
    returned stride is 0, so the same underlying value is reused at every
    position along that axis instead of being copied. Any other combination
    is not broadcast compatible and raises.
    """
    if source_dim == target_dim:
        return stride
    if source_dim == 1:
        return 0
    raise ValueError("target shape is not broadcast compatible")


def get_target_strides(
    source_shape: tuple[int, ...],
    source_strides: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Return strides which let a tensor with source_shape act as target_shape.

    The result is a tuple that can be passed when instantiating a new view or
    tensor. In the Python backend, PythonTensor.view can use the returned
    strides to create a cheap broadcast view: axes stretched from length 1 use
    stride 0 so the same source value is reused without copying.
    """
    if len(source_shape) != len(source_strides):
        raise ValueError("source shape and strides must have the same rank")

    try:
        padded_shape = left_pad(source_shape, len(target_shape), 1)
        padded_strides = left_pad(source_strides, len(target_shape), 0)
    except ValueError as e:
        raise ValueError("target shape is not broadcast compatible") from e

    return tuple(
        get_target_stride(source_dim, stride, target_dim)
        for source_dim, stride, target_dim in zip(
            padded_shape, padded_strides, target_shape
        )
    )


def get_target_shape(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Return the shape produced when two operand shapes are broadcast together.

    Elementwise operations use the result as the output tensor shape and as the
    target shape for creating broadcast views of the input operands.
    """
    max_rank = max(len(a_shape), len(b_shape))
    a_padded = left_pad(a_shape, max_rank, 1)
    b_padded = left_pad(b_shape, max_rank, 1)

    return tuple(
        get_target_dimension(a_dim, b_dim) for a_dim, b_dim in zip(a_padded, b_padded)
    )
