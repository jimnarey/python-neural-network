import unittest

from fnn.tensors.shared.composition import get_concatenate_shape, get_stack_shape


class TestGetConcatenateShape(unittest.TestCase):

    def test_returns_1D_shape_with_axis_size_from_all_tensors(self):
        """
        The input shapes are (2,), (3,) and (1,). Concatenating them on
        their only axis gives one 1D result whose length is 2 + 3 + 1.
        """
        result = get_concatenate_shape(((2,), (3,), (1,)), 0)
        self.assertEqual(result, (6,))

    def test_returns_2D_shape_when_concatenating_axis_0(self):
        """
        The input shapes are (2, 3) and (4, 3). Concatenating on axis 0
        combines the leading dimension and leaves the second dimension
        unchanged.
        """
        result = get_concatenate_shape(((2, 3), (4, 3)), 0)
        self.assertEqual(result, (6, 3))

    def test_returns_2D_shape_when_concatenating_axis_1(self):
        """
        The input shapes are (2, 3) and (2, 4). Concatenating on axis 1
        keeps the leading dimension and combines the second dimension.
        """
        result = get_concatenate_shape(((2, 3), (2, 4)), 1)
        self.assertEqual(result, (2, 7))

    def test_returns_3D_shape_when_concatenating_middle_axis(self):
        """
        The input shapes are (2, 3, 4) and (2, 5, 4). Concatenating on
        axis 1 keeps the outer axes and combines the middle axis, giving
        result shape (2, 8, 4).
        """
        result = get_concatenate_shape(((2, 3, 4), (2, 5, 4)), 1)
        self.assertEqual(result, (2, 8, 4))

    def test_accepts_singleton_sequence(self):
        """
        A singleton sequence is still a valid concatenation. Because there
        is only one source shape, the result shape is the source shape.
        """
        result = get_concatenate_shape(((2, 3),), 1)
        self.assertEqual(result, (2, 3))

    def test_returns_shape_when_axis_lengths_include_zero(self):
        """
        Zero-length dimensions are included in the axis-size calculation.
        Concatenating shapes (2, 0, 3) and (2, 4, 3) on axis 1 gives a
        result shape whose middle axis has length 0 + 4.
        """
        result = get_concatenate_shape(((2, 0, 3), (2, 4, 3)), 1)
        self.assertEqual(result, (2, 4, 3))


class TestGetStackShape(unittest.TestCase):

    def test_returns_shape_when_stacking_1D_tensors_on_axis_0(self):
        """
        Each input shape is (3,), representing a 1D tensor with three
        values. Stacking two of them on axis 0 creates a new leading
        axis of length 2, giving result shape (2, 3).
        """
        result = get_stack_shape(((3,), (3,)), 0)
        self.assertEqual(result, (2, 3))

    def test_returns_shape_when_stacking_1D_tensors_on_axis_1(self):
        """
        Each input shape is (3,). Stacking two such tensors on axis 1
        keeps the original length-3 axis first and adds a new second
        axis of length 2, giving result shape (3, 2).
        """
        result = get_stack_shape(((3,), (3,)), 1)
        self.assertEqual(result, (3, 2))

    def test_returns_shape_when_stacking_2D_tensors_on_axis_0(self):
        """
        Each input shape is (2, 3). Stacking two such tensors on axis
        0 creates a new leading axis of length 2, followed by the
        original axes, giving result shape (2, 2, 3).
        """
        result = get_stack_shape(((2, 3), (2, 3)), 0)
        self.assertEqual(result, (2, 2, 3))

    def test_returns_shape_when_stacking_2D_tensors_on_axis_1(self):
        """
        Each input shape is (2, 3). Stacking two such tensors on axis
        1 keeps the original leading length-2 axis first, inserts a
        new axis of length 2, and leaves the original length-3 axis
        last.
        """
        result = get_stack_shape(((2, 3), (2, 3)), 1)
        self.assertEqual(result, (2, 2, 3))

    def test_returns_shape_when_stacking_2D_tensors_on_axis_2(self):
        """
        Each input shape is (2, 3). Stacking two such tensors on axis 2
        keeps both original axes first and adds a new trailing axis of
        length 2, giving result shape (2, 3, 2).
        """
        result = get_stack_shape(((2, 3), (2, 3)), 2)
        self.assertEqual(result, (2, 3, 2))

    def test_accepts_singleton_sequence(self):
        """
        A singleton sequence still creates a stack axis. Because there is
        only one source shape, the new axis has length 1.
        """
        result = get_stack_shape(((2, 3),), 0)
        self.assertEqual(result, (1, 2, 3))

    def test_returns_shape_when_source_shape_has_zero_length_dimension(self):
        """
        A zero-length source dimension is preserved. Stacking still
        inserts a new axis whose length is the number of source shapes
        (tensors).
        """
        cases = (
            ((0,), 0, (2, 0)),
            ((2, 0), 1, (2, 2, 0)),
            ((2, 0, 3), 3, (2, 0, 3, 2)),
        )
        for source_shape, axis, expected in cases:
            with self.subTest(source_shape=source_shape, axis=axis):
                result = get_stack_shape((source_shape, source_shape), axis)
                self.assertEqual(result, expected)
