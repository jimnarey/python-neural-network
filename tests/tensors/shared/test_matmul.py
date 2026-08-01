import unittest

from fnn.tensors.shared.matmul import (
    get_matmul_leading_shape,
    get_matmul_result_index_parts,
    get_matmul_result_shape,
    get_matmul_result_inner_shape,
)


class TestGetMatmulResultInnerShape(unittest.TestCase):

    def test_returns_empty_shape_for_two_1D_operands(self):
        result = get_matmul_result_inner_shape((3,), (3,))
        self.assertEqual(result, ())

    def test_returns_vector_shape_for_1D_left_operand(self):
        result = get_matmul_result_inner_shape((3,), (3, 4))
        self.assertEqual(result, (4,))

    def test_returns_vector_shape_for_1D_right_operand(self):
        result = get_matmul_result_inner_shape((2, 3), (3,))
        self.assertEqual(result, (2,))

    def test_returns_matrix_shape_for_rank_2_or_higher_operands(self):
        cases = (
            ((2, 3), (3, 4), (2, 4)),
            ((5, 2, 3), (5, 3, 4), (2, 4)),
            ((6, 5, 2, 3), (6, 5, 3, 4), (2, 4)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_result_inner_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_preserves_zero_length_dimensions(self):
        cases = (
            ((0,), (0,), ()),
            ((0,), (0, 4), (4,)),
            ((2, 0), (0,), (2,)),
            ((2, 0), (0, 4), (2, 4)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_result_inner_shape(a_shape, b_shape)
                self.assertEqual(result, expected)


class TestGetMatmulLeadingShape(unittest.TestCase):

    def test_returns_empty_shape_when_neither_operand_has_leading_axes(self):
        cases = (
            ((3,), (3,)),
            ((3,), (3, 4)),
            ((2, 3), (3,)),
            ((2, 3), (3, 4)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_leading_shape(a_shape, b_shape)
                self.assertEqual(result, ())

    def test_returns_left_leading_shape_when_right_operand_has_no_leading_axes(self):
        cases = (
            ((5, 2, 3), (3, 4), (5,)),
            ((6, 5, 2, 3), (3,), (6, 5)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_leading_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_returns_right_leading_shape_when_left_operand_has_no_leading_axes(self):
        cases = (
            ((2, 3), (5, 3, 4), (5,)),
            ((3,), (6, 5, 3, 4), (6, 5)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_leading_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_broadcasts_leading_shapes(self):
        cases = (
            ((1, 2, 3), (5, 3, 4), (5,)),
            ((5, 1, 2, 3), (1, 4, 3, 2), (5, 4)),
            ((1, 6, 1, 2, 3), (7, 1, 5, 3, 4), (7, 6, 5)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_leading_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_raises_when_leading_shapes_are_not_broadcast_compatible(self):
        cases = (
            ((2, 2, 3), (3, 3, 4)),
            ((5, 2, 2, 3), (5, 3, 3, 4)),
            ((1, 2, 4, 2, 3), (3, 1, 5, 3, 4)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                with self.assertRaisesRegex(
                    ValueError, "shapes are not broadcast compatible"
                ):
                    get_matmul_leading_shape(a_shape, b_shape)


class TestGetMatmulResultShape(unittest.TestCase):
    """
    Tests for shape incompatibility are limited to a single
    broadcasting test because other types of compatibility
    resolution are delegated very straighforwardly.
    """

    def test_returns_empty_shape_for_two_1D_operands(self):
        """
        Tells the matmul operation it needs to return a scalar
        """
        result = get_matmul_result_shape((3,), (3,))
        self.assertEqual(result, ())

    def test_returns_vector_shape_for_1D_left_operand(self):
        """
        Note that the vector has to be the same length as
        the penultimate axis in the tensor.
        """
        cases = (
            ((3,), (3, 4), (4,)),
            ((3,), (5, 3, 4), (5, 4)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_result_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_returns_vector_shape_for_1D_right_operand(self):
        """
        Note that the vector has to be the same length as the
        final/innermost axis in the tensor.
        """
        cases = (
            ((2, 3), (3,), (2,)),
            ((5, 2, 3), (3,), (5, 2)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_result_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_returns_matrix_shape_for_2D_operands(self):
        cases = (
            ((2, 3), (3, 4), (2, 4)),
            ((4, 2), (2, 5), (4, 5)),
            ((1, 3), (3, 1), (1, 1)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_result_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_returns_higher_rank_shape_with_broadcast_leading_axes(self):
        """
        The first case proves that matching leading shapes are preserved.

        The second proves that a leading dimension of 1 can be broadcast
        to the other operand's leading dimension.

        The third proves that the same rule applies across multiple leading
        axes before the final matrix dimensions are added to the result.
        """
        cases = (
            ((5, 2, 3), (5, 3, 4), (5, 2, 4)),
            ((1, 2, 3), (5, 3, 4), (5, 2, 4)),
            ((5, 1, 2, 3), (1, 4, 3, 2), (5, 4, 2, 2)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_result_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_applies_matmul_shape_rules_when_shared_dimension_is_zero(self):
        cases = (
            ((0,), (0,), ()),
            ((0,), (0, 4), (4,)),
            ((2, 0), (0,), (2,)),
            ((2, 0), (0, 4), (2, 4)),
            ((5, 2, 0), (5, 0, 4), (5, 2, 4)),
        )
        for a_shape, b_shape, expected in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                result = get_matmul_result_shape(a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_raises_when_leading_shapes_are_not_broadcast_compatible(self):
        cases = (
            ((2, 2, 3), (3, 3, 4)),
            ((5, 2, 2, 3), (5, 3, 3, 4)),
            ((1, 2, 4, 2, 3), (3, 1, 5, 3, 4)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                with self.assertRaisesRegex(
                    ValueError, "shapes are not broadcast compatible"
                ):
                    get_matmul_result_shape(a_shape, b_shape)


class TestGetMatmulResultIndexParts(unittest.TestCase):

    def test_returns_parts_for_two_1D_operands(self):
        """
        The operands both have shape (3,), so this is a vector-vector
        multiplication. The result is a scalar, so result_index is ().

        Neither operand has leading axes, so both leading index tuples are
        (). Because both operands are 1D, there is no visible row or column
        position in the result, so both row_index and column_index are None.
        """
        result = get_matmul_result_index_parts((), (3,), (3,))
        self.assertEqual(result, ((), (), None, None))

    def test_returns_parts_for_1D_left_operand(self):
        """
        The left operand has shape (3,), so it is treated as a single row.
        The right operand has shape (3, 4), so the result has shape (4,).

        There are no leading axes. The result index (2,) selects column 2
        in the right-hand matrix. The left operand contributes no visible
        row position, so row_index is None.
        """
        result = get_matmul_result_index_parts((2,), (3,), (3, 4))
        self.assertEqual(result, ((), (), None, 2))

    def test_returns_parts_for_1D_right_operand(self):
        """
        The left operand has shape (2, 3), while the right operand has shape
        (3,), so the right operand is treated as a single column. The result
        has shape (2,).

        There are no leading axes. The result index (1,) selects row 1 in
        the left-hand matrix. The right operand contributes no visible
        column position, so column_index is None.
        """
        result = get_matmul_result_index_parts((1,), (2, 3), (3,))
        self.assertEqual(result, ((), (), 1, None))

    def test_returns_parts_for_2D_operands(self):
        """
        The operands have shapes (2, 3) and (3, 4), so the result has shape
        (2, 4).

        There are no leading axes. The result index (1, 2) identifies row 1
        from the left-hand matrix and column 2 from the right-hand matrix.
        """
        result = get_matmul_result_index_parts((1, 2), (2, 3), (3, 4))
        self.assertEqual(result, ((), (), 1, 2))

    def test_returns_parts_for_higher_rank_operands(self):
        """
        In the first case, both operands have one leading axis: (5,). The
        result index (3, 1, 2) therefore starts with leading index (3,),
        followed by row 1 from the left-hand matrix and column 2 from the
        right-hand matrix.

        In the second case, both operands have two leading axes: (6, 5).
        The result index (4, 3, 1, 0) therefore starts with leading index
        (4, 3), followed by row 1 from the left-hand matrix and column 0
        from the right-hand matrix.
        """
        cases = (
            ((3, 1, 2), (5, 2, 3), (5, 3, 4), ((3,), (3,), 1, 2)),
            (
                (4, 3, 1, 0),
                (6, 5, 2, 3),
                (6, 5, 3, 4),
                ((4, 3), (4, 3), 1, 0),
            ),
        )
        for result_index, a_shape, b_shape, expected in cases:
            with self.subTest(result_index=result_index):
                result = get_matmul_result_index_parts(result_index, a_shape, b_shape)
                self.assertEqual(result, expected)

    def test_maps_broadcasted_leading_axes_to_zero(self):
        """
        The left operand has leading shape (5, 1), while the right operand
        has leading shape (1, 4). These broadcast to the result leading
        shape (5, 4).

        The result index (2, 3, 0, 1) starts with leading index (2, 3).
        The left operand's second leading axis has length 1, so it must read
        from index 0 on that axis, giving leading index (2, 0). The right
        operand's first leading axis has length 1, so it must read from
        index 0 on that axis, giving leading index (0, 3).
        """
        result = get_matmul_result_index_parts(
            (2, 3, 0, 1),
            (5, 1, 2, 3),
            (1, 4, 3, 2),
        )
        self.assertEqual(result, ((2, 0), (0, 3), 0, 1))
