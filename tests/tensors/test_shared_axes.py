import unittest
from src.tensors.axes import (
    normalise_axis,
    normalise_axes,
)


class TestAxesNormalisation(unittest.TestCase):
    """
    Tests the shared helpers for converting axis arguments to
    non-negative indices.

    We do not test ndim == 0 because the backend contract does
    not allow rank-0 tensors, so axis normalisation should only
    be needed for tensors with one or more axes.
    """

    def test_normalise_axis_returns_positive_axis_unchanged_when_axis_in_bounds(self):
        cases = (
            (0, 1),
            (0, 3),
            (1, 3),
            (2, 3),
        )
        for axis, ndim in cases:
            with self.subTest():
                self.assertEqual(normalise_axis(axis, ndim), axis)

    def test_normalise_axis_converts_negative_axis_to_positive_equivalent(self):
        cases = (
            (-1, 1, 0),
            (-1, 3, 2),
            (-2, 3, 1),
            (-3, 3, 0),
        )
        for axis, ndim, expected in cases:
            with self.subTest():
                self.assertEqual(normalise_axis(axis, ndim), expected)

    def test_normalise_axis_raises_when_positive_axis_out_of_bounds(self):
        cases = (
            (1, 1),
            (3, 3),
            (4, 3),
        )
        for axis, ndim in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "axis is out of bounds"):
                    normalise_axis(axis, ndim)

    def test_normalise_axis_raises_when_negative_axis_out_of_bounds(self):
        cases = (
            (-2, 1),
            (-4, 3),
            (-5, 3),
        )
        for axis, ndim in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "axis is out of bounds"):
                    normalise_axis(axis, ndim)

    def test_normalise_axes_returns_tuple_with_each_axis_normalised(self):
        cases = (
            ((0,), 1, (0,)),
            ((0, -1), 2, (0, 1)),
            ((-1, -2, -3), 3, (2, 1, 0)),
            ((2, -1, 0), 3, (2, 2, 0)),
        )
        for axes, ndim, expected in cases:
            with self.subTest():
                self.assertEqual(normalise_axes(axes, ndim), expected)

    def test_normalise_axes_raises_when_any_axis_out_of_bounds(self):
        cases = (
            ((0, 1), 1),
            ((0, 1, 3), 3),
            ((0, -4, 2), 3),
        )
        for axes, ndim in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "axis is out of bounds"):
                    normalise_axes(axes, ndim)
