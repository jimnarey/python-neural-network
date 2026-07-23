import unittest
from src.tensors.validation import (
    validate_shape_not_rank_0,
    validate_shape_has_no_negative_dimensions,
    validate_transpose_axes_are_permutation,
    validate_tensor_conversion_root_is_sequence,
    parse_tensor_data,
)


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
                validate_transpose_axes_are_permutation(axes, ndim)

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
                    validate_transpose_axes_are_permutation(axes, ndim)

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
                    validate_transpose_axes_are_permutation(axes, ndim)

    def test_validate_transpose_axes_are_permutation_raises_when_axes_tuple_contains_duplicate_axis(
        self,
    ):
        cases = (
            ((0, 0), 2),
            ((0, 0, 1), 3),
            ((2, 1, 1), 3),
        )
        for axes, ndim in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    validate_transpose_axes_are_permutation(axes, ndim)

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
                    validate_transpose_axes_are_permutation(axes, ndim)


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
