import unittest

from fnn.tensors.shared.broadcasting import (
    get_target_dimension,
    get_target_shape,
    get_target_stride,
    get_target_strides,
    left_pad,
)


class TestLeftPad(unittest.TestCase):

    def test_returns_values_unchanged_when_target_rank_matches(self):
        values = (2, 3)
        result = left_pad(values, 2, 1)
        self.assertEqual(result, (2, 3))

    def test_adds_fill_values_on_the_left_when_target_rank_is_larger(self):
        cases = (
            ((3,), 2, 1, (1, 3)),
            ((2, 3), 4, 1, (1, 1, 2, 3)),
            ((4,), 3, 0, (0, 0, 4)),
        )
        for values, target_rank, fill_value, expected in cases:
            with self.subTest():
                result = left_pad(values, target_rank, fill_value)
                self.assertEqual(result, expected)

    def test_raises_when_target_rank_is_smaller_than_values_rank(self):
        with self.assertRaisesRegex(ValueError, "target rank"):
            left_pad((2, 3), 1, 1)


class TestGetTargetDimension(unittest.TestCase):

    def test_returns_dimension_when_dimensions_are_equal(self):
        cases = (0, 1, 2, 5)
        for dimension in cases:
            with self.subTest():
                result = get_target_dimension(dimension, dimension)
                self.assertEqual(result, dimension)

    def test_returns_right_dimension_when_left_dimension_is_1(self):
        right_dimensions = (0, 2, 5)
        for right_dimension in right_dimensions:
            with self.subTest():
                result = get_target_dimension(1, right_dimension)
                self.assertEqual(result, right_dimension)

    def test_returns_left_dimension_when_right_dimension_is_1(self):
        left_dimensions = (0, 2, 5)
        for left_dimension in left_dimensions:
            with self.subTest():
                result = get_target_dimension(left_dimension, 1)
                self.assertEqual(result, left_dimension)

    def test_returns_zero_when_one_dimension_is_0_and_the_other_is_1(self):
        """
        This covers an edge case where broadcasting produces a zero-length
        axis.

        A tensor with a zero-length dimension has no values, so this case
        has no direct practical value. It is still useful to test because
        shape logic should remain consistent for empty tensors rather than
        adding special cases which are easy for later backend implementations
        to get wrong.
        """
        cases = ((0, 1), (1, 0))
        for left_dimension, right_dimension in cases:
            with self.subTest():
                result = get_target_dimension(left_dimension, right_dimension)
                self.assertEqual(result, 0)

    def test_raises_when_dimensions_are_incompatible(self):
        cases = ((3, 2), (2, 5), (8, 4), (5, 6), (0, 2), (4, 0))
        for left_dimension, right_dimension in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "not broadcast compatible"):
                    get_target_dimension(left_dimension, right_dimension)


class TestGetTargetStride(unittest.TestCase):

    def test_returns_existing_stride_when_source_dimension_matches_target_dimension(
        self,
    ):
        cases = ((0, 7, 0), (1, 3, 1), (4, 2, 4), (100, 5, 100))
        for source_dimension, stride, target_dimension in cases:
            with self.subTest():
                result = get_target_stride(source_dimension, stride, target_dimension)
                self.assertEqual(result, stride)

    def test_returns_zero_when_source_dimension_is_broadcast(self):
        cases = ((1, 3, 2), (1, 5, 4), (1, 8, 9))
        for source_dimension, stride, target_dimension in cases:
            with self.subTest():
                result = get_target_stride(source_dimension, stride, target_dimension)
                self.assertEqual(result, 0)

    def test_returns_zero_when_source_dimension_is_1_and_target_dimension_is_0(self):
        strides = (0, 1, 5, 20)
        for stride in strides:
            with self.subTest():
                result = get_target_stride(1, stride, 0)
                self.assertEqual(result, 0)

    def test_raises_when_source_dimension_cannot_broadcast_to_target_dimension(self):
        cases = ((2, 1, 3), (3, 2, 2), (0, 4, 5), (4, 6, 0))
        for source_dimension, stride, target_dimension in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "not broadcast compatible"):
                    get_target_stride(source_dimension, stride, target_dimension)


class TestGetTargetShape(unittest.TestCase):

    def test_returns_expected_shape_when_both_inputs_are_1D(self):
        cases = (
            ((3,), (3,), (3,)),
            ((1,), (3,), (3,)),
            ((3,), (1,), (3,)),
            ((1,), (5,), (5,)),
            ((5,), (1,), (5,)),
            ((0,), (1,), (0,)),
            ((1,), (0,), (0,)),
        )
        for left_shape, right_shape, expected in cases:
            with self.subTest():
                result = get_target_shape(left_shape, right_shape)
                self.assertEqual(result, expected)

    def test_returns_expected_shape_when_both_inputs_are_2D(self):
        cases = (
            ((2, 5), (2, 5), (2, 5)),
            ((4, 1), (4, 6), (4, 6)),
            ((1, 7), (5, 7), (5, 7)),
            ((1, 1), (6, 8), (6, 8)),
            ((0, 9), (1, 9), (0, 9)),
        )
        for left_shape, right_shape, expected in cases:
            with self.subTest():
                result = get_target_shape(left_shape, right_shape)
                self.assertEqual(result, expected)

    def test_returns_expected_shape_when_inputs_are_3D_or_higher(self):
        cases = (
            ((4, 1, 6), (4, 5, 6), (4, 5, 6)),
            ((1, 7, 1), (8, 7, 9), (8, 7, 9)),
            ((2, 1, 5, 1), (1, 6, 5, 7), (2, 6, 5, 7)),
            (
                (2, 1, 4, 1, 6, 1),
                (1, 3, 4, 5, 1, 7),
                (2, 3, 4, 5, 6, 7),
            ),
            ((0, 4, 1), (1, 4, 8), (0, 4, 8)),
        )
        for left_shape, right_shape, expected in cases:
            with self.subTest():
                result = get_target_shape(left_shape, right_shape)
                self.assertEqual(result, expected)

    def test_returns_expected_shape_when_inputs_have_different_ranks(self):
        cases = (
            ((5,), (4, 5), (4, 5)),
            ((6,), (7, 4, 6), (7, 4, 6)),
            ((6,), (7, 4, 1), (7, 4, 6)),
            ((1, 8), (5, 7, 8), (5, 7, 8)),
            ((4, 9), (6, 5, 4, 9), (6, 5, 4, 9)),
            ((8,), (7, 6, 5, 4, 8), (7, 6, 5, 4, 8)),
            ((9,), (0, 9), (0, 9)),
        )
        for left_shape, right_shape, expected in cases:
            with self.subTest():
                result = get_target_shape(left_shape, right_shape)
                self.assertEqual(result, expected)

    def test_raises_when_shapes_are_incompatible(self):
        """
        These cases cover different combinations of compatible and
        incompatible axes, so that an incompatible axis is detected
        wherever it occurs in the shape.
        """
        cases = (
            ((4,), (5,)),
            ((4, 5), (4, 6)),
            ((0, 7), (5, 7)),
            ((4, 5), (3, 4, 6)),
            ((4, 5, 6), (4, 7, 6)),
        )
        for left_shape, right_shape in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "not broadcast compatible"):
                    get_target_shape(left_shape, right_shape)


class TestGetTargetStrides(unittest.TestCase):

    def test_returns_expected_strides_when_source_and_target_are_1D(self):
        cases = (
            ((3,), (1,), (3,), (1,)),
            ((1,), (4,), (1,), (4,)),
            ((0,), (4,), (0,), (4,)),
            ((1,), (3,), (5,), (0,)),
            ((1,), (7,), (0,), (0,)),
        )
        for source_shape, source_strides, target_shape, expected in cases:
            with self.subTest():
                result = get_target_strides(source_shape, source_strides, target_shape)
                self.assertEqual(result, expected)

    def test_returns_expected_strides_when_source_and_target_are_2D(self):
        cases = (
            ((2, 5), (5, 1), (2, 5), (5, 1)),
            ((4, 1), (1, 1), (4, 6), (1, 0)),
            ((1, 7), (7, 1), (5, 7), (0, 1)),
            ((1, 1), (1, 1), (6, 8), (0, 0)),
            ((0, 9), (9, 1), (0, 9), (9, 1)),
            ((5, 0), (0, 1), (5, 0), (0, 1)),
            ((5, 1), (1, 1), (5, 0), (1, 0)),
        )
        for source_shape, source_strides, target_shape, expected in cases:
            with self.subTest():
                result = get_target_strides(source_shape, source_strides, target_shape)
                self.assertEqual(result, expected)

    def test_returns_expected_strides_when_source_and_target_are_3D_or_higher(self):
        cases = (
            ((4, 1, 6), (6, 6, 1), (4, 5, 6), (6, 0, 1)),
            ((1, 7, 1), (7, 1, 1), (8, 7, 9), (0, 1, 0)),
            ((5, 0, 7), (0, 7, 1), (5, 0, 7), (0, 7, 1)),
            ((5, 1, 7), (7, 7, 1), (5, 0, 7), (7, 0, 1)),
            ((2, 1, 5, 1), (5, 5, 1, 1), (2, 6, 5, 7), (5, 0, 1, 0)),
            (
                (2, 1, 4, 1, 6, 1),
                (24, 24, 6, 6, 1, 1),
                (2, 3, 4, 5, 6, 7),
                (24, 0, 6, 0, 1, 0),
            ),
            ((0, 4, 1), (4, 1, 1), (0, 4, 8), (4, 1, 0)),
        )
        for source_shape, source_strides, target_shape, expected in cases:
            with self.subTest():
                result = get_target_strides(source_shape, source_strides, target_shape)
                self.assertEqual(result, expected)

    def test_returns_expected_strides_when_source_and_target_have_different_ranks(
        self,
    ):
        cases = (
            ((5,), (1,), (4, 5), (0, 1)),
            ((6,), (2,), (7, 4, 6), (0, 0, 2)),
            ((1, 8), (8, 1), (5, 7, 8), (0, 0, 1)),
            ((4, 9), (9, 1), (6, 5, 4, 9), (0, 0, 9, 1)),
            ((8,), (1,), (7, 6, 5, 4, 8), (0, 0, 0, 0, 1)),
            ((9,), (1,), (0, 9), (0, 1)),
        )
        for source_shape, source_strides, target_shape, expected in cases:
            with self.subTest():
                result = get_target_strides(source_shape, source_strides, target_shape)
                self.assertEqual(result, expected)

    def test_raises_when_source_shape_and_strides_have_different_ranks(self):
        """
        This doesn't test for a shape mismatch but where somehow the strides
        tuple and the shape tuple, used to calculate the strides for the
        target tensor, have a different lengths.
        """
        cases = (
            ((2, 5), (1,), (2, 5)),
            ((4,), (4, 1), (4,)),
            ((3, 4, 5), (20, 5), (3, 4, 5)),
        )
        for source_shape, source_strides, target_shape in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "same rank"):
                    get_target_strides(source_shape, source_strides, target_shape)

    def test_raises_when_target_shape_has_fewer_dimensions_than_source_shape(self):
        """
        These cases have a target_shape with fewer axes than the source_shape.
        Broadcasting only ever adds leading axes to a shape; it never
        removes them, so the shapes are incompatible.
        """
        cases = (
            ((3, 4, 5), (20, 5, 1), (4, 5)),
            ((4, 5), (5, 1), (5,)),
            ((2, 3, 4, 5), (60, 20, 5, 1), (3, 4, 5)),
        )
        for source_shape, source_strides, target_shape in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "not broadcast compatible"):
                    get_target_strides(source_shape, source_strides, target_shape)

    def test_raises_when_aligned_target_dimensions_are_not_broadcast_compatible(
        self,
    ):
        """
        These cases cover different combinations of compatible and
        incompatible axes, so that an incompatible axis is detected
        wherever it occurs in the shape — not only when it is the sole
        axis, or the first one checked.
        """
        cases = (
            ((4,), (1,), (5,)),
            ((4, 5), (5, 1), (4, 6)),
            ((0, 7), (7, 1), (5, 7)),
            ((4, 5), (5, 1), (3, 4, 6)),
            ((4, 5, 6), (30, 6, 1), (4, 7, 6)),
            ((1, 5), (5, 1), (4, 6)),
        )
        for source_shape, source_strides, target_shape in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "not broadcast compatible"):
                    get_target_strides(source_shape, source_strides, target_shape)
