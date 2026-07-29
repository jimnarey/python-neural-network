import unittest

from src.tensors.shared.reductions import (
    get_reduction_axes_and_target_shape,
    get_reduction_target_index,
    get_reduction_target_shape,
    normalise_axis_argument,
)


class TestNormaliseAxisArgument(unittest.TestCase):

    def test_returns_all_axes_when_axis_is_none(self):
        cases = (
            (1, (0,)),
            (2, (0, 1)),
            (4, (0, 1, 2, 3)),
        )
        for ndim, expected in cases:
            with self.subTest():
                result = normalise_axis_argument(None, ndim)
                self.assertEqual(result, expected)

    def test_returns_single_item_tuple_when_axis_is_int(self):
        cases = (
            (0, (0,)),
            (2, (2,)),
            (-1, (-1,)),
        )
        for axis, expected in cases:
            with self.subTest():
                result = normalise_axis_argument(axis, 3)
                self.assertEqual(result, expected)

    def test_returns_axis_tuple_unchanged_when_axis_is_tuple(self):
        cases = (
            ((),),
            ((0,),),
            ((2, 0),),
            ((1, -1),),
        )
        for (axis,) in cases:
            with self.subTest():
                result = normalise_axis_argument(axis, 3)
                self.assertEqual(result, axis)

    def test_raises_when_axis_argument_has_unsupported_type(self):
        cases = (
            [0],
            "0",
            1.0,
        )
        for axis in cases:
            with self.subTest():
                with self.assertRaisesRegex(TypeError, "axis must be"):
                    normalise_axis_argument(axis, 3)


class TestGetReductionTargetShape(unittest.TestCase):

    def test_returns_shape_with_reduced_axis_removed_when_keepdims_is_false(self):
        cases = (
            ((5,), (0,), ()),
            ((4, 6), (0,), (6,)),
            ((4, 6), (1,), (4,)),
            ((2, 1, 7), (1,), (2, 7)),
            ((2, 0, 7), (1,), (2, 7)),
        )
        for shape, reduced_axes, expected in cases:
            with self.subTest():
                result = get_reduction_target_shape(shape, reduced_axes)
                self.assertEqual(result, expected)

    def test_returns_shape_with_multiple_reduced_axes_removed_when_keepdims_is_false(
        self,
    ):
        cases = (
            ((4, 6), (0, 1), ()),
            ((2, 5, 7), (0, 2), (5,)),
            ((2, 5, 7), (1, 2), (2,)),
            ((2, 1, 7, 9), (1, 3), (2, 7)),
            ((2, 0, 7, 9), (1, 3), (2, 7)),
        )
        for shape, reduced_axes, expected in cases:
            with self.subTest():
                result = get_reduction_target_shape(shape, reduced_axes)
                self.assertEqual(result, expected)

    def test_returns_shape_with_reduced_axis_set_to_1_when_keepdims_is_true(self):
        cases = (
            ((5,), (0,), (1,)),
            ((4, 6), (0,), (1, 6)),
            ((4, 6), (1,), (4, 1)),
            ((2, 1, 7), (1,), (2, 1, 7)),
            ((2, 0, 7), (1,), (2, 1, 7)),
        )
        for shape, reduced_axes, expected in cases:
            with self.subTest():
                result = get_reduction_target_shape(shape, reduced_axes, keepdims=True)
                self.assertEqual(result, expected)

    def test_returns_shape_with_multiple_reduced_axes_set_to_1_when_keepdims_is_true(
        self,
    ):
        cases = (
            ((4, 6), (0, 1), (1, 1)),
            ((2, 5, 7), (0, 2), (1, 5, 1)),
            ((2, 5, 7), (1, 2), (2, 1, 1)),
            ((2, 1, 7, 9), (1, 3), (2, 1, 7, 1)),
            ((2, 0, 7, 9), (1, 3), (2, 1, 7, 1)),
        )
        for shape, reduced_axes, expected in cases:
            with self.subTest():
                result = get_reduction_target_shape(shape, reduced_axes, keepdims=True)
                self.assertEqual(result, expected)

    def test_returns_original_shape_when_axes_tuple_is_empty(self):
        cases = (
            ((5,), False),
            ((4, 6), False),
            ((2, 1, 7), False),
            ((2, 0, 7), False),
            ((5,), True),
            ((4, 6), True),
            ((2, 1, 7), True),
            ((2, 0, 7), True),
        )
        for shape, keepdims in cases:
            with self.subTest():
                result = get_reduction_target_shape(shape, (), keepdims=keepdims)
                self.assertEqual(result, shape)


class TestGetReductionAxesAndTargetShape(unittest.TestCase):

    def test_returns_reduced_axes_and_target_shape_when_axis_is_none(self):
        cases = (
            ((5,), False, ((0,), ())),
            ((5,), True, ((0,), (1,))),
            ((4, 6), False, ((0, 1), ())),
            ((2, 5, 7), False, ((0, 1, 2), ())),
            ((2, 0, 7), True, ((0, 1, 2), (1, 1, 1))),
        )
        for shape, keepdims, expected in cases:
            with self.subTest():
                result = get_reduction_axes_and_target_shape(
                    shape,
                    None,
                    keepdims=keepdims,
                )
                self.assertEqual(result, expected)

    def test_returns_reduced_axes_and_target_shape_when_axis_is_int(self):
        cases = (
            ((5,), 0, False, ((0,), ())),
            ((4, 6), 0, False, ((0,), (6,))),
            ((4, 6), 1, False, ((1,), (4,))),
            ((2, 0, 7), 1, True, ((1,), (2, 1, 7))),
        )
        for shape, axis, keepdims, expected in cases:
            with self.subTest():
                result = get_reduction_axes_and_target_shape(
                    shape,
                    axis,
                    keepdims=keepdims,
                )
                self.assertEqual(result, expected)

    def test_returns_reduced_axes_and_target_shape_when_axis_is_tuple(self):
        cases = (
            ((4, 6), (0, 1), False, ((0, 1), ())),
            ((2, 5, 7), (0, 2), False, ((0, 2), (5,))),
            ((2, 0, 7), (1, 2), False, ((1, 2), (2,))),
            ((2, 5, 7), (1, 2), True, ((1, 2), (2, 1, 1))),
            ((2, 0, 7, 9), (1, 3), True, ((1, 3), (2, 1, 7, 1))),
            ((2, 5, 7, 9, 11), (1, 3), False, ((1, 3), (2, 7, 11))),
        )
        for shape, axis, keepdims, expected in cases:
            with self.subTest():
                result = get_reduction_axes_and_target_shape(
                    shape,
                    axis,
                    keepdims=keepdims,
                )
                self.assertEqual(result, expected)

    def test_normalises_negative_axes_before_returning_reduced_axes(self):
        cases = (
            ((5,), -1, False, ((0,), ())),
            ((4, 6), -1, False, ((1,), (4,))),
            ((2, 5, 7), (0, -1), False, ((0, 2), (5,))),
            ((2, 5, 7), (-2, -1), True, ((1, 2), (2, 1, 1))),
        )
        for shape, axis, keepdims, expected in cases:
            with self.subTest():
                result = get_reduction_axes_and_target_shape(
                    shape,
                    axis,
                    keepdims=keepdims,
                )
                self.assertEqual(result, expected)

    def test_raises_when_normalised_axes_contain_duplicates(self):
        cases = (
            ((5,), (0, 0)),
            ((4, 6), (1, -1)),
            ((2, 5, 7), (-1, 2)),
        )
        for shape, axis in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    get_reduction_axes_and_target_shape(shape, axis)


class TestGetReductionTargetIndex(unittest.TestCase):

    def test_returns_index_with_reduced_axis_removed_when_keepdims_is_false(self):
        cases = (
            ((4,), (0,), ()),
            ((2, 5), (0,), (5,)),
            ((2, 5), (1,), (2,)),
            ((2, 5, 7), (1,), (2, 7)),
            ((2, 0, 7), (1,), (2, 7)),
        )
        for source_index, reduced_axes, expected in cases:
            with self.subTest():
                result = get_reduction_target_index(source_index, reduced_axes)
                self.assertEqual(result, expected)

    def test_returns_index_with_multiple_reduced_axes_removed_when_keepdims_is_false(
        self,
    ):
        cases = (
            ((2, 5), (0, 1), ()),
            ((2, 5, 7), (0, 2), (5,)),
            ((2, 5, 7), (1, 2), (2,)),
            ((2, 5, 7, 9), (1, 3), (2, 7)),
            ((2, 0, 7, 9), (1, 3), (2, 7)),
        )
        for source_index, reduced_axes, expected in cases:
            with self.subTest():
                result = get_reduction_target_index(source_index, reduced_axes)
                self.assertEqual(result, expected)

    def test_returns_index_with_reduced_axis_set_to_0_when_keepdims_is_true(self):
        cases = (
            ((4,), (0,), (0,)),
            ((2, 5), (0,), (0, 5)),
            ((2, 5), (1,), (2, 0)),
            ((2, 5, 7), (1,), (2, 0, 7)),
            ((2, 0, 7), (1,), (2, 0, 7)),
        )
        for source_index, reduced_axes, expected in cases:
            with self.subTest():
                result = get_reduction_target_index(
                    source_index,
                    reduced_axes,
                    keepdims=True,
                )
                self.assertEqual(result, expected)

    def test_returns_index_with_multiple_reduced_axes_set_to_0_when_keepdims_is_true(
        self,
    ):
        cases = (
            ((2, 5), (0, 1), (0, 0)),
            ((2, 5, 7), (0, 2), (0, 5, 0)),
            ((2, 5, 7), (1, 2), (2, 0, 0)),
            ((2, 5, 7, 9), (1, 3), (2, 0, 7, 0)),
            ((2, 0, 7, 9), (1, 3), (2, 0, 7, 0)),
        )
        for source_index, reduced_axes, expected in cases:
            with self.subTest():
                result = get_reduction_target_index(
                    source_index,
                    reduced_axes,
                    keepdims=True,
                )
                self.assertEqual(result, expected)

    def test_returns_original_index_when_axes_tuple_is_empty(self):
        cases = (
            ((5,), False),
            ((4, 6), False),
            ((2, 1, 7), False),
            ((2, 0, 7), False),
            ((5,), True),
            ((4, 6), True),
            ((2, 1, 7), True),
            ((2, 0, 7), True),
        )
        for source_index, keepdims in cases:
            with self.subTest():
                result = get_reduction_target_index(
                    source_index,
                    (),
                    keepdims=keepdims,
                )
                self.assertEqual(result, source_index)
