from src.tensors.shared.broadcasting import get_target_shape


def get_matmul_result_inner_shape(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Return the part of the matmul result shape produced by the final axes.
    """
    if len(a_shape) == 1 and len(b_shape) == 1:
        return ()
    elif len(a_shape) == 1:
        return (b_shape[-1],)
    elif len(b_shape) == 1:
        return (a_shape[-2],)
    return (a_shape[-2], b_shape[-1])


def get_matmul_leading_shape(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Return the broadcast shape for the leading axes of two matmul operands.

    For rank-2 operands, there are no leading axes: the two axes describe the
    single matrix being multiplied. For higher-rank operands, any axes before
    the final two are leading axes. They identify separate matrix
    multiplications and are broadcast together.
    """
    a_leading_shape = () if len(a_shape) == 1 else a_shape[:-2]
    b_leading_shape = () if len(b_shape) == 1 else b_shape[:-2]
    return get_target_shape(a_leading_shape, b_leading_shape)


def get_matmul_result_shape(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Return the full output shape for a matmul operation.

    The result shape combines the broadcast leading shape with the result
    produced by multiplying the operands' inner matrix shapes.
    """
    result_inner_shape = get_matmul_result_inner_shape(a_shape, b_shape)
    return get_matmul_leading_shape(a_shape, b_shape) + result_inner_shape


def get_matmul_result_index_parts(
    result_index: tuple[int, ...],
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int | None, int | None]:
    """
    This returns four things:
    - a tuple of indices which specify which matrix, within a tensor
      of three dimensions or more, is the left hand operand
    - a tuple of indices doing the same for the right hand operand
    - a single value specifying which row in the left hand matrix is
      being operated on
    - a single value specifying which column in the the right hand matrix
      is being operated on

    This allows us to populate the output tensor by iterating through
    its positions.

    Any broadcasting required to identify the correct positions in the
    operands is reflected in the return values.

    Work out which row of a, which column of b, and which position in each
    operand's own stack, are needed to compute one value of a matmul
    result.

    For example, take a_shape (5, 1, 2, 3) and b_shape (1, 4, 3, 2). Their
    leading shapes, (5, 1) and (1, 4), broadcast to (5, 4), so the result
    has shape (5, 4, 2, 2). For result_index (2, 3, 0, 1):

    - the leading part of the result index is (2, 3)
    - a's own leading shape is (5, 1), so its second leading axis
      broadcasts: a's leading index is (2, 0), not (2, 3)
    - b's own leading shape is (1, 4), so its first leading axis
      broadcasts: b's leading index is (0, 3), not (2, 3)
    - the row position, 0, selects a row from a's matrix
    - the column position, 1, selects a column from b's matrix

    If a is a 1D vector, there is no row axis in the result — the vector is
    treated as a single row that gets consumed entirely by the
    multiplication, not read position by position — so row_index is None.
    The same applies to column_index when b is a 1D vector. When both are
    1D vectors, neither a row nor a column axis exists in the result.
    """
    leading_shape = get_matmul_leading_shape(a_shape, b_shape)
    leading_index = result_index[: len(leading_shape)]
    row_index = None
    column_index = None
    if len(a_shape) == 1 and len(b_shape) == 1:
        pass
    elif len(a_shape) == 1:
        column_index = result_index[len(leading_shape)]
    elif len(b_shape) == 1:
        row_index = result_index[len(leading_shape)]
    else:
        row_index = result_index[len(leading_shape)]
        column_index = result_index[len(leading_shape) + 1]
    return (
        _get_source_leading_index(leading_index, _get_matmul_leading_axes(a_shape)),
        _get_source_leading_index(leading_index, _get_matmul_leading_axes(b_shape)),
        row_index,
        column_index,
    )


def _get_matmul_leading_axes(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Return the axes that broadcast across separate matrix multiplications.

    A 1D vector and a 2D matrix both have no leading axes, so shapes such as
    (3,) and (2, 3) return (). A higher-rank shape such as (4, 2, 3)
    represents four separate (2, 3) matrices, so the leading axes are (4,).
    """
    if len(shape) == 1:
        return ()
    return shape[:-2]


def _get_source_leading_index(
    target_index: tuple[int, ...], source_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Return the leading index one operand needs, given a leading index into
    the broadcast shape shared by both operands.

    get_matmul_result_index_parts calls this once for each operand to turn
    a single shared leading index into the index that operand's own stack
    actually needs — the two can differ because of broadcasting, in either
    of two ways.

    First, target_index may have more positions than source_shape has
    axes, if this operand's stack has fewer leading axes than the other
    operand's. Those extra positions belong only to the other operand, so
    they are skipped over: offset counts how many, and every axis in
    source_shape is read from target_index starting offset positions in,
    not from the very start.

    Second, once aligned, an axis in source_shape may itself have length
    1, meaning this operand's stack does not vary along that axis at all
    — the same single position is reused regardless of where the other
    operand is on the corresponding axis. Reading that axis from
    target_index would be wrong, so it is fixed at 0 instead.

    For example, take source_shape (1, 4) and a target_index with four
    values, (5, 2, 7, 3):

    - offset is 4 - 2 = 2, so source_shape lines up with the last two
      values of target_index, (7, 3); the first two values, (5, 2), belong
      only to the other operand and are never read
    - the first axis in source_shape has length 1, so it reads as 0, not
      7, even though target_index has a 7 in that position
    - the second axis has length 4, so it reads straight through as 3

    The result is (0, 3).
    """
    offset = len(target_index) - len(source_shape)
    return tuple(
        0 if dimension == 1 else target_index[offset + axis]
        for axis, dimension in enumerate(source_shape)
    )
