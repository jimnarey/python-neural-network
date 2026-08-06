"""Shared shape arithmetic for combining several tensors into one.

Covers the two ways this project combines a sequence of same-rank tensors:
concatenating them along an existing axis, and stacking them along a new
one. Both only need the operands' shapes, not their values.
"""


def get_concatenate_shape(
    shapes: tuple[tuple[int, ...], ...], normalised_axis: int
) -> tuple[int, ...]:
    base_shape = shapes[0]
    axis_size = sum(shape[normalised_axis] for shape in shapes)
    return (
        base_shape[:normalised_axis] + (axis_size,) + base_shape[normalised_axis + 1 :]
    )


def get_stack_shape(
    shapes: tuple[tuple[int, ...], ...], normalised_axis: int
) -> tuple[int, ...]:
    base_shape = shapes[0]
    return base_shape[:normalised_axis] + (len(shapes),) + base_shape[normalised_axis:]
