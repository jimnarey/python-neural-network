import unittest
from src.tensors.shared.validation import (
    validate_tensor_has_values,
    validate_shapes_match_except_axis,
    validate_shape_not_rank_0,
    validate_shape_has_no_negative_dimensions,
    validate_reduction_has_values,
    validate_scalar_is_not_bool,
    validate_axes_are_unique,
    validate_axes_are_permutation,
    validate_tensor_conversion_root_is_sequence,
    validate_matmul_operand_ranks,
    validate_matmul_core_dimensions,
    parse_tensor_data,
)

#
# shape_size is untested. That's fine, given the callers are simple and
# thoroughly tested.
#


class TestValidateShapeNotRank0(unittest.TestCase):

    def test_validate_shape_not_rank_0_accepts_rank_1_or_higher_shape(self):
        cases = (
            (3,),
            (2, 3),
            (2, 3, 4),
        )
        for shape in cases:
            with self.subTest():
                validate_shape_not_rank_0(shape)

    def test_validate_shape_not_rank_0_accepts_shape_with_zero_length_dimension(self):
        cases = (
            (0,),
            (2, 0),
            (2, 0, 3),
        )
        for shape in cases:
            with self.subTest():
                validate_shape_not_rank_0(shape)

    def test_validate_shape_not_rank_0_raises_when_shape_is_empty_tuple(self):
        with self.assertRaisesRegex(ValueError, "require a non-empty shape"):
            validate_shape_not_rank_0(())


class TestValidateShapeHasNoNegativeDimensions(unittest.TestCase):

    def test_validate_shape_has_no_negative_dimensions_accepts_non_negative_shape(self):
        cases = (
            (3,),
            (2, 3),
            (2, 3, 4),
        )
        for shape in cases:
            with self.subTest():
                validate_shape_has_no_negative_dimensions(shape, "reshape")

    def test_validate_shape_has_no_negative_dimensions_accepts_shape_with_zero_length_dimension(
        self,
    ):
        cases = (
            (0,),
            (2, 0),
            (2, 0, 3),
        )
        for shape in cases:
            with self.subTest():
                validate_shape_has_no_negative_dimensions(shape, "reshape")

    def test_validate_shape_has_no_negative_dimensions_raises_when_shape_contains_negative_dimension(
        self,
    ):
        cases = (
            (-1,),
            (2, -1),
            (2, -1, 0),
        )
        for shape in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "negative values"):
                    validate_shape_has_no_negative_dimensions(shape, "reshape")


class TestValidateScalarIsNotBool(unittest.TestCase):

    def test_validate_scalar_is_not_bool_accepts_non_bool_scalars(self):
        cases = (0, 1, -5, 100, 0.0, -2.5, 3.14)
        for value in cases:
            with self.subTest():
                validate_scalar_is_not_bool(value)

    def test_validate_scalar_is_not_bool_raises_when_value_is_bool(self):
        cases = (True, False)
        for value in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "must not be a bool"):
                    validate_scalar_is_not_bool(value)


class TestValidateReductionHasValues(unittest.TestCase):

    def test_accepts_reduction_when_reduced_axes_contain_values(self):
        cases = (
            ((5,), (0,)),
            ((4, 6), (0,)),
            ((4, 6), (1,)),
            ((2, 5, 7), (0, 2)),
            ((2, 1, 7), (1,)),
        )
        for shape, reduced_axes in cases:
            with self.subTest():
                validate_reduction_has_values(shape, reduced_axes)

    def test_accepts_reduction_when_axes_tuple_is_empty(self):
        cases = (
            (5,),
            (4, 6),
            (2, 1, 7),
            (2, 0, 7),
        )
        for shape in cases:
            with self.subTest():
                validate_reduction_has_values(shape, ())

    def test_raises_when_reduced_axes_contain_no_values(self):
        cases = (
            ((0,), (0,)),
            ((2, 0), (1,)),
            ((2, 0, 3), (1,)),
            ((2, 0, 3), (0, 1)),
        )
        for shape, reduced_axes in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "has no values"):
                    validate_reduction_has_values(shape, reduced_axes)


class TestValidateTensorHasValues(unittest.TestCase):

    def test_accepts_tensor_shape_with_values(self):
        cases = (
            (5,),
            (4, 6),
            (2, 1, 7),
        )
        for shape in cases:
            with self.subTest():
                validate_tensor_has_values(shape)

    def test_raises_when_tensor_shape_has_no_values(self):
        cases = (
            (0,),
            (2, 0),
            (2, 0, 3),
            (2, 3, 0),
        )
        for shape in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "has no values"):
                    validate_tensor_has_values(shape)


class TestValidateShapesMatchExceptAxis(unittest.TestCase):

    def test_accepts_single_shape(self):
        cases = (
            (((3,),), 0),
            (((2, 3),), 1),
            (((2, 3, 4),), 2),
        )
        for shapes, axis in cases:
            with self.subTest():
                validate_shapes_match_except_axis(shapes, axis)

    def test_accepts_shapes_which_match_except_axis(self):
        """
        This proves, amongst other things, that zero length dimensions
        are permissable either when specified in the axis argument or
        not. Zero is not a special case when determining tensor
        compatibility.
        """
        cases = (
            (((3,), (5,), (1,)), 0),
            (((2, 3), (2, 5), (2, 1)), 1),
            (((2, 3, 4), (2, 5, 4), (2, 1, 4)), 1),
            (((0,), (3,)), 0),
            (((2, 0, 4), (2, 3, 4)), 1),
            (((2, 0, 3), (2, 0, 5)), 2),
        )
        for shapes, axis in cases:
            with self.subTest():
                validate_shapes_match_except_axis(shapes, axis)

    def test_raises_when_no_shapes_are_passed(self):
        with self.assertRaisesRegex(ValueError, "at least one shape"):
            validate_shapes_match_except_axis((), 0)

    def test_raises_when_axis_is_outside_shape_rank(self):
        cases = (
            (((3,),), 1),
            (((2, 3),), 2),
            (((2, 3, 4),), 3),
            (((2, 3),), -1),
        )
        for shapes, axis in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "outside the valid range"):
                    validate_shapes_match_except_axis(shapes, axis)

    def test_raises_when_shapes_have_different_ranks(self):
        cases = (
            (((3,), (1, 3)), 0),
            (((2, 3), (2, 3, 4)), 1),
            (((2, 3, 4), (2, 3)), 2),
        )
        for shapes, axis in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "same rank"):
                    validate_shapes_match_except_axis(shapes, axis)

    def test_raises_when_shapes_differ_outside_axis(self):
        cases = (
            (((2, 3), (3, 5)), 1),
            (((2, 3, 4), (2, 5, 6)), 1),
            (((2, 0, 3), (2, 1, 5)), 2),
        )
        for shapes, axis in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "match except"):
                    validate_shapes_match_except_axis(shapes, axis)


class TestValidateMatmulOperandRanks(unittest.TestCase):

    def test_accepts_operands_with_rank_1_or_higher(self):
        cases = (
            ((3,), (3,)),
            ((2, 3), (3, 4)),
            ((2, 2, 3), (3,)),
            ((3,), (2, 3, 4)),
            ((2, 1, 2, 3), (1, 2, 3, 2)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                validate_matmul_operand_ranks(a_shape, b_shape)

    def test_raises_when_left_operand_is_rank_0(self):
        cases = (
            ((), (3,)),
            ((), (2, 3)),
            ((), (2, 3, 4)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                with self.assertRaisesRegex(ValueError, "at least one dimension"):
                    validate_matmul_operand_ranks(a_shape, b_shape)

    def test_raises_when_right_operand_is_rank_0(self):
        cases = (
            ((3,), ()),
            ((2, 3), ()),
            ((2, 3, 4), ()),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                with self.assertRaisesRegex(ValueError, "at least one dimension"):
                    validate_matmul_operand_ranks(a_shape, b_shape)


class TestValidateMatmulCoreDimensions(unittest.TestCase):

    def test_accepts_two_1D_operands_with_matching_lengths(self):
        cases = (
            ((0,), (0,)),
            ((1,), (1,)),
            ((3,), (3,)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                validate_matmul_core_dimensions(a_shape, b_shape)

    def test_accepts_1D_left_operand_and_rank_2_or_higher_right_operand(self):
        """
        This pins down the rule for vector @ matrix-style matmul.

        The 1D left operand is treated as a temporary row vector. Its only
        dimension must match the row count of each matrix in the right
        operand, which is the second-to-last dimension for any rank-2-or-
        higher right operand.
        """
        cases = (
            ((3,), (3, 2)),
            ((3,), (2, 3, 4)),
            ((3,), (2, 1, 3, 4)),
            ((0,), (0, 2)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                validate_matmul_core_dimensions(a_shape, b_shape)

    def test_accepts_rank_2_or_higher_left_operand_and_1D_right_operand(self):
        """
        This pins down the rule for matrix @ vector-style matmul.

        The 1D right operand is treated as a temporary column vector. Its
        only dimension must match the column count of each matrix in the
        left operand, which is the final dimension for any rank-2-or-higher
        left operand.
        """
        cases = (
            ((2, 3), (3,)),
            ((2, 4, 3), (3,)),
            ((2, 1, 4, 3), (3,)),
            ((2, 0), (0,)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                validate_matmul_core_dimensions(a_shape, b_shape)

    def test_accepts_rank_2_or_higher_operands_with_matching_core_dimensions(self):
        """
        This pins down the rule for matrix @ matrix-style matmul.

        For operands with rank 2 or higher, only the final two axes form the
        matrices being multiplied. The final dimension of the left operand
        must match the second-to-last dimension of the right operand. Earlier
        axes are not checked by this guard.
        """
        cases = (
            ((2, 3), (3, 4)),
            ((2, 2, 3), (3, 4)),
            ((2, 2, 3), (2, 3, 4)),
            ((2, 1, 2, 3), (1, 2, 3, 2)),
            ((2, 0), (0, 3)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                validate_matmul_core_dimensions(a_shape, b_shape)

    def test_raises_when_two_1D_operands_have_different_lengths(self):
        """
        These fail because the two vector lengths are different.
        """
        cases = (
            ((2,), (3,)),
            ((0,), (1,)),
            ((1,), (0,)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                with self.assertRaisesRegex(ValueError, "core dimensions"):
                    validate_matmul_core_dimensions(a_shape, b_shape)

    def test_raises_when_1D_left_operand_length_does_not_match_right_core_dimension(
        self,
    ):
        """
        These fail because the vector length does not match the row count
        of each right-hand matrix.
        """
        cases = (
            ((3,), (4, 2)),
            ((3,), (2, 4, 5)),
            ((0,), (1, 2)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                with self.assertRaisesRegex(ValueError, "core dimensions"):
                    validate_matmul_core_dimensions(a_shape, b_shape)

    def test_raises_when_1D_right_operand_length_does_not_match_left_core_dimension(
        self,
    ):
        """
        These fail because the vector length does not match the column
        count of each left-hand matrix.
        """
        cases = (
            ((2, 3), (4,)),
            ((2, 4, 3), (4,)),
            ((2, 1), (0,)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                with self.assertRaisesRegex(ValueError, "core dimensions"):
                    validate_matmul_core_dimensions(a_shape, b_shape)

    def test_raises_when_rank_2_or_higher_operands_have_different_core_dimensions(
        self,
    ):
        """
        These fail because the left-hand matrix column count does not match
        the right-hand matrix row count.
        """
        cases = (
            ((2, 3), (4, 5)),
            ((2, 2, 3), (2, 4, 5)),
            ((2, 2, 0), (2, 1, 3)),
        )
        for a_shape, b_shape in cases:
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                with self.assertRaisesRegex(ValueError, "core dimensions"):
                    validate_matmul_core_dimensions(a_shape, b_shape)


class TestValidateAxesAreUnique(unittest.TestCase):

    def test_validate_axes_are_unique_accepts_axes_with_no_duplicates(self):
        cases = (
            (),
            (0,),
            (1, 0),
            (2, 0, 1),
            (5, -3, 9),
        )
        for axes in cases:
            with self.subTest():
                validate_axes_are_unique(axes)

    def test_validate_axes_are_unique_accepts_non_contiguous_axes(self):
        """
        This function only checks for duplicates. It has no ndim argument,
        so it cannot check whether axes forms a complete permutation of any
        particular range. These cases each have a missing axis or an
        out-of-bounds value, and should still be accepted because none of
        the values repeat.
        """
        cases = (
            (0, 2),
            (0, 1, 3),
            (0, 5),
            (-1, 0, 1),
        )
        for axes in cases:
            with self.subTest():
                validate_axes_are_unique(axes)

    def test_validate_axes_are_unique_raises_when_axes_contain_duplicates(self):
        cases = (
            (0, 0),
            (0, 0, 1),
            (2, 1, 1),
            (5, 3, 5, 7),
        )
        for axes in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    validate_axes_are_unique(axes)


class TestValidateTransposeAxesArePermutation(unittest.TestCase):

    def test_validate_transpose_axes_are_permutation_accepts_complete_axes_permutation(
        self,
    ):
        cases = (
            ((0,), 1),
            ((1, 0), 2),
            ((2, 0, 1), 3),
            ((3, 1, 0, 2), 4),
        )
        for axes, ndim in cases:
            with self.subTest():
                validate_axes_are_permutation(axes, ndim)

    def test_validate_transpose_axes_are_permutation_raises_when_axes_tuple_is_too_short(
        self,
    ):
        cases = (
            ((), 1),
            ((0,), 2),
            ((2, 1), 3),
        )
        for axes, ndim in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    validate_axes_are_permutation(axes, ndim)

    def test_validate_transpose_axes_are_permutation_raises_when_axes_tuple_is_too_long(
        self,
    ):
        cases = (
            ((0, 1), 1),
            ((1, 0, 2), 2),
            ((2, 1, 0, 3), 3),
        )
        for axes, ndim in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    validate_axes_are_permutation(axes, ndim)

    def test_validate_transpose_axes_are_permutation_raises_when_axes_tuple_contains_duplicate_axis(
        self,
    ):
        """
        Rejection of axes arguments containing duplicates is delegated, so we check
        for a different error message.

        This behaviour is thoroughly tested in TestValidateAxesAreUnique but the
        way the permutation is validated using sets is the sort of thing it's easy
        to go wrong with, so test it properly here.
        """
        cases = (
            ((0, 0), 2),
            ((0, 0, 1), 3),
            ((2, 1, 1), 3),
        )
        for axes, ndim in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    validate_axes_are_permutation(axes, ndim)

    def test_validate_transpose_axes_are_permutation_raises_when_axes_tuple_omits_axis(
        self,
    ):
        cases = (
            ((0, 2), 2),
            ((0, 1, 3), 3),
            ((3, 2, 1), 4),
        )
        for axes, ndim in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    validate_axes_are_permutation(axes, ndim)


class TestValidateTensorConversionRootIsSequence(unittest.TestCase):

    def test_validate_tensor_conversion_root_is_sequence_accepts_list_or_tuple(self):
        cases = (
            [],
            (),
            [1.0, 2.0],
            (1.0, 2.0),
        )
        for data in cases:
            with self.subTest():
                validate_tensor_conversion_root_is_sequence(data)

    def test_validate_tensor_conversion_root_is_sequence_raises_when_data_is_not_list_or_tuple(
        self,
    ):
        cases = (
            1.0,
            "data",
            True,
            None,
            dict(),
            set(),
        )
        for data in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "requires a list or tuple"):
                    validate_tensor_conversion_root_is_sequence(data)


class TestParseTensorData(unittest.TestCase):

    def test_returns_rank_0_shape_and_data_in_list_when_data_is_float(self):
        cases = (1.0, 5.7, 100.6)
        for data in cases:
            with self.subTest():
                shape, values = parse_tensor_data(data)
                self.assertEqual(values, [data])
                self.assertIs(type(values[0]), float)
                self.assertEqual(shape, ())

    def test_returns_rank_0_shape_and_float_in_list_when_data_is_int(self):
        cases = (1, 5, 100, 0, -1, -5, -100)
        for data in cases:
            with self.subTest():
                shape, values = parse_tensor_data(data)
                self.assertEqual(values, [float(data)])
                self.assertIs(type(values[0]), float)
                self.assertEqual(shape, ())

    def test_raises_when_data_is_or_contains_invalid_type(self):
        cases = (
            "data",
            True,
            None,
            dict(),
            set(),
            bytes(),
            bytearray(),
            [1.0, "data"],
            [[1.0, True]],
            [[[1.0], [None]]],
        )
        for data in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "requires numeric values"):
                    parse_tensor_data(data)

    def test_returns_rank_1_shape_with_zero_length_dimension_and_no_values_when_data_is_empty_list_tuple(
        self,
    ):
        cases = (
            [],
            (),
        )
        for data in cases:
            with self.subTest():
                shape, values = parse_tensor_data(data)
                self.assertEqual(shape, (0,))
                self.assertEqual(values, [])

    def test_returns_expected_shape_and_no_values_when_data_has_nested_empty_sequences(
        self,
    ):
        cases = (
            ([[]], (1, 0)),
            ([[], []], (2, 0)),
            ([[[]], [[]]], (2, 1, 0)),
        )
        for data, expected_shape in cases:
            with self.subTest():
                shape, values = parse_tensor_data(data)
                self.assertEqual(shape, expected_shape)
                self.assertEqual(values, [])

    def test_raises_when_data_is_not_rectangular(self):
        cases = (
            [[1, 2], [3]],
            [[[1, 2], [3, 4]], [[5, 6]]],
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]]],
            ],
        )
        for data in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "requires rectangular input"):
                    parse_tensor_data(data)

    def test_raises_when_sibling_elements_have_different_nesting_depth(self):
        cases = (
            [[1, 2], 3],
            [1, [2, 3]],
            [[[1, 2], [3, 4]], [1, 2]],
            [[1, 2], [[3, 4], [5, 6]]],
        )
        for data in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "requires rectangular input"):
                    parse_tensor_data(data)

    def test_raises_when_data_mixes_empty_and_non_empty_sequences_at_same_level(self):
        cases = (
            [[], [1.0]],
            [[[]], [[1.0]]],
            [[], [], [1.0]],
        )
        for data in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "requires rectangular input"):
                    parse_tensor_data(data)

    def test_returns_expected_shape_and_flat_list_when_data_is_1D(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (5,))
        self.assertEqual(values, data)
        self.assertTrue(all(type(value) is float for value in values))

    def test_returns_expected_shape_and_flat_list_when_data_is_2D_and_rectangular(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (2, 3))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertTrue(all(type(value) is float for value in values))

    def test_returns_expected_shape_and_flat_list_when_data_is_3D_and_rectangular(self):
        data = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (2, 2, 2))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertTrue(all(type(value) is float for value in values))

    def test_returns_expected_shape_and_flat_list_when_data_is_4D_and_rectangular(self):
        data = [
            [
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            [
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        ]
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (2, 1, 2, 2))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertTrue(all(type(value) is float for value in values))

    def test_returns_expected_shape_and_flat_list_when_data_mixes_lists_and_tuples(
        self,
    ):
        """
        Test that we get valid return values when lists and tuples are mixed
        in the same input data.

        This is to pin down the behaviour. Mixing lists and tuples is not
        especially desirable but it isn't forbidden so we should test for it.
        """
        data = ([1.0, 2.0], (3.0, 4.0))
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (2, 2))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(all(type(value) is float for value in values))
