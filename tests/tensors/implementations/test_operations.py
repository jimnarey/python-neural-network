import operator
import unittest
from array import array

from src.tensors.python_backend.operations import (
    argmax_to_scalar,
    argmax_to_tensor,
    concatenate_tensors,
    copy_sequence_values,
    divide_reduction_result,
    first_max_index,
    get_concatenate_shape,
    get_stack_shape,
    map_binary,
    map_unary,
    reduce_to_scalar,
    reduce_to_tensor,
    stack_tensors,
)
from src.tensors.python_backend.python_tensor import PythonTensor


class TestFirstMaxIndex(unittest.TestCase):

    def test_returns_index_of_largest_value(self):
        values = [2.0, 9.0, 4.0, 7.0]
        result = first_max_index(values)
        self.assertEqual(result, 1)

    def test_returns_first_index_when_maximum_value_is_tied(self):
        values = [2.0, 9.0, 4.0, 9.0]
        result = first_max_index(values)
        self.assertEqual(result, 1)

    def test_consumes_one_pass_iterable(self):
        values = (value for value in [2.0, 4.0, 9.0, 7.0])
        result = first_max_index(values)
        self.assertEqual(result, 2)


class TestArgmaxToScalar(unittest.TestCase):

    def test_returns_flat_index_for_1D_tensor(self):
        tensor = PythonTensor((4,), array("d", [2.0, 9.0, 4.0, 7.0]))
        result = argmax_to_scalar(tensor)
        self.assertEqual(result, 1)

    def test_returns_flat_index_for_2D_tensor(self):
        tensor = PythonTensor((2, 3), array("d", [2.0, 4.0, 7.0, 9.0, 6.0, 8.0]))
        result = argmax_to_scalar(tensor)
        self.assertEqual(result, 3)

    def test_uses_tensor_layout_when_tensor_is_view(self):
        """
        The parent tensor has shape (2, 3) and values:
            [[1.0, 9.0, 3.0],
             [4.0, 5.0, 6.0]]

        The view has shape (3, 2) and strides (1, 3), so it reads the same
        buffer as:
            [[1.0, 4.0],
             [9.0, 5.0],
             [3.0, 6.0]]

        In that logical view, 9.0 is the third value encountered, so its
        flattened index is 2.
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 9.0, 3.0, 4.0, 5.0, 6.0]))
        view = tensor.view((3, 2), strides=(1, 3))
        result = argmax_to_scalar(view)
        self.assertEqual(result, 2)

    def test_returns_first_flat_index_when_maximum_value_is_tied(self):
        tensor = PythonTensor((2, 3), array("d", [2.0, 9.0, 7.0, 9.0, 6.0, 8.0]))
        result = argmax_to_scalar(tensor)
        self.assertEqual(result, 1)


class TestArgmaxToTensor(unittest.TestCase):

    def test_returns_indices_when_reducing_2D_tensor_axis_0(self):
        """
        The tensor has shape (2, 3) and values:
            [[1.0, 5.0, 3.0],
             [4.0, 2.0, 6.0]]

        Reducing axis 0 means searching down each column. The maximum values
        are at row index 1 for column 0, row index 0 for column 1, and row
        index 1 for column 2.

        The result is therefore [1, 0, 1].
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]))
        result = argmax_to_tensor(tensor, 0, (3,))
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.data.tolist(), [1, 0, 1])

    def test_returns_indices_when_reducing_2D_tensor_axis_1(self):
        """
        The tensor has shape (2, 3) and values:
            [[1.0, 5.0, 3.0],
             [4.0, 2.0, 6.0]]

        Reducing axis 1 means searching across each row. The maximum values
        are at column index 1 for row 0 and column index 2 for row 1.

        The result is therefore [1, 2].
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]))
        result = argmax_to_tensor(tensor, 1, (2,))
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.data.tolist(), [1, 2])

    def test_returns_indices_when_reducing_3D_tensor_middle_axis(self):
        """
        The tensor has shape (2, 3, 2). It can be read as two 2D tensors,
        one for each position on axis 0:

            axis 0, position 0:
                [[1.0, 5.0],
                 [7.0, 2.0],
                 [3.0, 9.0]]

            axis 0, position 1:
                [[4.0, 8.0],
                 [6.0, 1.0],
                 [2.0, 10.0]]

        Reducing axis 1 means searching down the rows inside each of those
        2D tensors, once for each fixed axis-0 and axis-2 position.

        For axis-0 position 0 and axis-2 position 0, the searched values are
        1.0, 7.0 and 3.0, so the maximum is at axis-1 index 1. For axis-0
        position 0 and axis-2 position 1, the searched values are 5.0, 2.0
        and 9.0, so the maximum is at axis-1 index 2.

        The same calculation is then repeated for axis-0 position 1, giving
        axis-1 indices 1 and 2.

        The result is therefore [[1, 2], [1, 2]].
        """
        tensor = PythonTensor(
            (2, 3, 2),
            array(
                "d",
                [1.0, 5.0, 7.0, 2.0, 3.0, 9.0, 4.0, 8.0, 6.0, 1.0, 2.0, 10.0],
            ),
        )
        result = argmax_to_tensor(tensor, 1, (2, 2))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.tolist(), [1, 2, 1, 2])

    def test_returns_int_valued_tensor(self):
        tensor = PythonTensor((2, 3), array("d", [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]))
        result = argmax_to_tensor(tensor, 0, (3,))
        self.assertEqual(result.data.typecode, PythonTensor.INT)

    def test_returns_first_axis_index_when_maximum_value_is_tied(self):
        """
        The tensor has shape (2, 3) and values:
            [[4.0, 5.0, 6.0],
             [4.0, 2.0, 6.0]]

        It is reduced over axis 0, so column 0 compares 4.0 and 4.0,
        column 1 compares 5.0 and 2.0, and column 2 compares 6.0 and
        6.0. In the tied columns, the first maximum value is at row index 0.

        The result is therefore [0, 0, 0].
        """
        tensor = PythonTensor((2, 3), array("d", [4.0, 5.0, 6.0, 4.0, 2.0, 6.0]))
        result = argmax_to_tensor(tensor, 0, (3,))
        self.assertEqual(result.data.tolist(), [0, 0, 0])

    def test_uses_tensor_layout_when_tensor_is_view(self):
        """
        The parent tensor has shape (2, 3) and values:
            [[1.0, 9.0, 3.0],
             [4.0, 5.0, 6.0]]

        The view has shape (3, 2) and strides (1, 3), so it reads the same
        buffer as:
            [[1.0, 4.0],
             [9.0, 5.0],
             [3.0, 6.0]]

        Reducing axis 0 of that view means searching down each column. The
        first column holds 1.0, 9.0 and 3.0, so its maximum is at row index
        1. The second column holds 4.0, 5.0 and 6.0, so its maximum is at
        row index 2.

        The result is therefore [1, 2].
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 9.0, 3.0, 4.0, 5.0, 6.0]))
        view = tensor.view((3, 2), strides=(1, 3))
        result = argmax_to_tensor(view, 0, (2,))
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.data.tolist(), [1, 2])


class TestGetConcatenateShape(unittest.TestCase):

    def test_returns_1D_shape_with_axis_size_from_all_tensors(self):
        """
        The input tensors have shapes (2,), (3,) and (1,). Concatenating
        them on their only axis gives one 1D result whose length is
        2 + 3 + 1.
        """
        first = PythonTensor((2,), array("d", [1.0, 2.0]))
        second = PythonTensor((3,), array("d", [3.0, 4.0, 5.0]))
        third = PythonTensor((1,), array("d", [6.0]))
        result = get_concatenate_shape((first, second, third), 0)
        self.assertEqual(result, (6,))

    def test_returns_2D_shape_when_concatenating_axis_0(self):
        """
        The input tensors have shapes (2, 3) and (4, 3). Concatenating on
        axis 0 combines the leading dimension and leaves the second
        dimension unchanged.
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor(
            (4, 3),
            array(
                "d",
                [
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                ],
            ),
        )
        result = get_concatenate_shape((first, second), 0)
        self.assertEqual(result, (6, 3))

    def test_returns_2D_shape_when_concatenating_axis_1(self):
        """
        The input tensors have shapes (2, 3) and (2, 4). Concatenating on
        axis 1 keeps the leading dimension and combines the second
        dimension.
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor(
            (2, 4), array("d", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        )
        result = get_concatenate_shape((first, second), 1)
        self.assertEqual(result, (2, 7))

    def test_returns_3D_shape_when_concatenating_middle_axis(self):
        """
        The input tensors have shapes (2, 3, 4) and (2, 5, 4).
        Concatenating on axis 1 keeps the outer axes and combines the middle
        axis, giving result shape (2, 8, 4).
        """
        first = PythonTensor((2, 3, 4), array("d", (float(i) for i in range(24))))
        second = PythonTensor((2, 5, 4), array("d", (float(i) for i in range(40))))
        result = get_concatenate_shape((first, second), 1)
        self.assertEqual(result, (2, 8, 4))

    def test_accepts_singleton_sequence(self):
        """
        A singleton sequence is still a valid concatenation. Because there
        is only one source tensor, the result shape is the source shape.
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        result = get_concatenate_shape((tensor,), 1)
        self.assertEqual(result, (2, 3))

    def test_returns_shape_when_axis_lengths_include_zero(self):
        """
        Zero-length dimensions are included in the axis-size calculation.
        Concatenating shapes (2, 0, 3) and (2, 4, 3) on axis 1 gives a
        result shape whose middle axis has length 0 + 4.
        """
        first = PythonTensor((2, 0, 3), array("d"))
        second = PythonTensor((2, 4, 3), array("d", (float(i) for i in range(24))))
        result = get_concatenate_shape((first, second), 1)
        self.assertEqual(result, (2, 4, 3))


class TestCopySequenceValues(unittest.TestCase):

    def test_copies_values_using_target_index_function(self):
        first = PythonTensor((2,), array("d", [1.0, 2.0]))
        second = PythonTensor((2,), array("d", [3.0, 4.0]))
        result = PythonTensor((4,))
        copy_sequence_values(
            (first, second),
            result,
            lambda tensor_index, source_index: (tensor_index * 2 + source_index[0],),
        )
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_passes_source_tensor_position_to_target_index_function(self):
        """
        This proves that the target-index function is told which source
        tensor is currently being copied, not just the index within that
        source tensor.
        """
        first = PythonTensor((1,), array("d", [1.0]))
        second = PythonTensor((1,), array("d", [2.0]))
        tensor_indices = []
        result = PythonTensor((2,))
        copy_sequence_values(
            (first, second),
            result,
            lambda tensor_index, _: tensor_indices.append(tensor_index)
            or (tensor_index,),
        )
        self.assertEqual(tensor_indices, [0, 1])
        self.assertEqual(result.data.tolist(), [1.0, 2.0])

    def test_uses_source_tensor_layout_when_copying_values(self):
        """
        The parent tensor has shape (2, 3) and values:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

        The view has shape (3, 2) and strides (1, 3), so it reads the same
        buffer as:
            [[1.0, 4.0],
             [2.0, 5.0],
             [3.0, 6.0]]

        This demonstrates that values are copied from the source tensor's
        logical layout, rather than from the underlying buffer order.
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        view = tensor.view((3, 2), strides=(1, 3))
        result = PythonTensor((3, 2))
        copy_sequence_values((view,), result, lambda _, source_index: source_index)
        self.assertEqual(result.data.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_returns_result_tensor(self):
        tensor = PythonTensor((1,), array("d", [1.0]))
        result = PythonTensor((1,))
        returned = copy_sequence_values(
            (tensor,), result, lambda _, source_index: source_index
        )
        self.assertIs(returned, result)


class TestConcatenateTensors(unittest.TestCase):

    def test_concatenates_1D_tensors(self):
        first = PythonTensor((2,), array("d", [1.0, 2.0]))
        second = PythonTensor((3,), array("d", [3.0, 4.0, 5.0]))
        result = concatenate_tensors((first, second), 0, (5,))
        self.assertEqual(result.shape, (5,))
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_concatenates_2D_tensors_along_axis_0(self):
        """
        The first tensor has shape (2, 3) and values:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

        The second tensor has shape (1, 3) and values:
            [[7.0, 8.0, 9.0]]

        Concatenating on axis 0 appends rows, so the result is:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor(
            (1, 3),
            array("d", [7.0, 8.0, 9.0]),
        )
        result = concatenate_tensors((first, second), 0, (3, 3))
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(
            result.data.tolist(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )

    def test_concatenates_2D_tensors_along_axis_1(self):
        """
        The first tensor has shape (2, 2) and values:
            [[1.0, 2.0],
             [3.0, 4.0]]

        The second tensor has shape (2, 1) and values:
            [[5.0],
             [6.0]]

        Concatenating on axis 1 appends columns, so the result is:
            [[1.0, 2.0, 5.0],
             [3.0, 4.0, 6.0]]
        """
        first = PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0]))
        second = PythonTensor((2, 1), array("d", [5.0, 6.0]))
        result = concatenate_tensors((first, second), 1, (2, 3))
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 5.0, 3.0, 4.0, 6.0])

    def test_concatenates_more_than_two_3D_tensors(self):
        """
        Each input tensor has shape (2, 1, 2):
            first:  [[[1.0, 2.0]], [[3.0, 4.0]]]
            second: [[[5.0, 6.0]], [[7.0, 8.0]]]
            third:  [[[9.0, 10.0]], [[11.0, 12.0]]]

        Concatenating on axis 1 joins the values inside each outer group,
        giving:
            [[[1.0, 2.0],
              [5.0, 6.0],
              [9.0, 10.0]],
             [[3.0, 4.0],
              [7.0, 8.0],
              [11.0, 12.0]]]

        Shape: (2, 3, 2)
        """
        first = PythonTensor((2, 1, 2), array("d", [1.0, 2.0, 3.0, 4.0]))
        second = PythonTensor((2, 1, 2), array("d", [5.0, 6.0, 7.0, 8.0]))
        third = PythonTensor((2, 1, 2), array("d", [9.0, 10.0, 11.0, 12.0]))
        result = concatenate_tensors((first, second, third), 1, (2, 3, 2))
        self.assertEqual(result.shape, (2, 3, 2))
        self.assertEqual(
            result.data.tolist(),
            [
                1.0,
                2.0,
                5.0,
                6.0,
                9.0,
                10.0,
                3.0,
                4.0,
                7.0,
                8.0,
                11.0,
                12.0,
            ],
        )

    def test_accepts_singleton_sequence(self):
        tensor = PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0]))
        result = concatenate_tensors((tensor,), 1, (2, 2))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_uses_tensor_layout_when_a_source_tensor_is_a_view(self):
        """
        The parent tensor has shape (2, 3) and values:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

        The view has shape (3, 2) and strides (1, 3), so it reads the same
        buffer as:
            [[1.0, 4.0],
             [2.0, 5.0],
             [3.0, 6.0]]

        The second tensor has shape (3, 1) and values:
            [[10.0],
             [20.0],
             [30.0]]

        Concatenating on axis 1 appends the second tensor as a final column,
        giving:
            [[1.0, 4.0, 10.0],
             [2.0, 5.0, 20.0],
             [3.0, 6.0, 30.0]]
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        view = tensor.view((3, 2), strides=(1, 3))
        second = PythonTensor((3, 1), array("d", [10.0, 20.0, 30.0]))
        result = concatenate_tensors((view, second), 1, (3, 3))
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(
            result.data.tolist(),
            [1.0, 4.0, 10.0, 2.0, 5.0, 20.0, 3.0, 6.0, 30.0],
        )

    def test_concatenates_tensors_with_zero_length_axis(self):
        """
        The first two cases concatenate along axis 1. One tensor has shape
        (2, 0, 3), so it has no values along the concatenated axis. The
        other has shape (2, 1, 3) and values:
            [[[1.0, 2.0, 3.0]],
             [[4.0, 5.0, 6.0]]]

        Concatenating on axis 1 keeps only the non-empty values, regardless
        of whether the empty tensor comes first or second. The result has
        shape (2, 1, 3) and values:
            [[[1.0, 2.0, 3.0]],
             [[4.0, 5.0, 6.0]]]

        The final case concatenates on axis 0 while axis 1 has length 0 in
        both tensors. The result shape is (5, 0, 3), but it has no values
        because one of its dimensions still has length 0.
        """
        cases = (
            (
                PythonTensor((2, 0, 3), array("d")),
                PythonTensor((2, 1, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
                1,
                (2, 1, 3),
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            (
                PythonTensor((2, 1, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
                PythonTensor((2, 0, 3), array("d")),
                1,
                (2, 1, 3),
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            (
                PythonTensor((2, 0, 3), array("d")),
                PythonTensor((3, 0, 3), array("d")),
                0,
                (5, 0, 3),
                [],
            ),
        )
        for first, second, axis, shape, expected in cases:
            with self.subTest(first_shape=first.shape, second_shape=second.shape):
                result = concatenate_tensors((first, second), axis, shape)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.data.tolist(), expected)


class TestGetStackShape(unittest.TestCase):

    def test_returns_shape_when_stacking_1D_tensors_on_axis_0(self):
        """
        Each input tensor has shape (3,), so each is a 1D tensor with three
        values. Stacking two of them on axis 0 creates a new leading axis of
        length 2, giving result shape (2, 3).
        """
        first = PythonTensor((3,), array("d", [1.0, 2.0, 3.0]))
        second = PythonTensor((3,), array("d", [4.0, 5.0, 6.0]))
        result = get_stack_shape((first, second), 0)
        self.assertEqual(result, (2, 3))

    def test_returns_shape_when_stacking_1D_tensors_on_axis_1(self):
        """
        Each input tensor has shape (3,). Stacking two of them on axis 1
        keeps the original length-3 axis first and adds a new second axis of
        length 2, giving result shape (3, 2).
        """
        first = PythonTensor((3,), array("d", [1.0, 2.0, 3.0]))
        second = PythonTensor((3,), array("d", [4.0, 5.0, 6.0]))
        result = get_stack_shape((first, second), 1)
        self.assertEqual(result, (3, 2))

    def test_returns_shape_when_stacking_2D_tensors_on_axis_0(self):
        """
        Each input tensor has shape (2, 3). Stacking two of them on axis 0
        creates a new leading axis of length 2, followed by the original
        axes, giving result shape (2, 2, 3).
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor((2, 3), array("d", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
        result = get_stack_shape((first, second), 0)
        self.assertEqual(result, (2, 2, 3))

    def test_returns_shape_when_stacking_2D_tensors_on_axis_1(self):
        """
        Each input tensor has shape (2, 3). Stacking two of them on axis 1
        keeps the original leading length-2 axis first, inserts a new axis
        of length 2, and leaves the original length-3 axis last.
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor((2, 3), array("d", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
        result = get_stack_shape((first, second), 1)
        self.assertEqual(result, (2, 2, 3))

    def test_returns_shape_when_stacking_2D_tensors_on_axis_2(self):
        """
        Each input tensor has shape (2, 3). Stacking two of them on axis 2
        keeps both original axes first and adds a new trailing axis of
        length 2, giving result shape (2, 3, 2).
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor((2, 3), array("d", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
        result = get_stack_shape((first, second), 2)
        self.assertEqual(result, (2, 3, 2))

    def test_accepts_singleton_sequence(self):
        """
        A singleton sequence still creates a stack axis. Because there is
        only one source tensor, the new axis has length 1.
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        result = get_stack_shape((tensor,), 0)
        self.assertEqual(result, (1, 2, 3))

    def test_returns_shape_when_source_shape_has_zero_length_dimension(self):
        """
        A zero-length source dimension is preserved. Stacking still inserts
        a new axis whose length is the number of source tensors.
        """
        cases = (
            ((0,), 0, (2, 0)),
            ((2, 0), 1, (2, 2, 0)),
            ((2, 0, 3), 3, (2, 0, 3, 2)),
        )
        for source_shape, axis, expected in cases:
            with self.subTest(source_shape=source_shape, axis=axis):
                first = PythonTensor(source_shape, array("d"))
                second = PythonTensor(source_shape, array("d"))
                result = get_stack_shape((first, second), axis)
                self.assertEqual(result, expected)


class TestStackTensors(unittest.TestCase):

    def test_stacks_1D_tensors_on_axis_0(self):
        """
        The input tensors are [1.0, 2.0] and [3.0, 4.0]. Stacking on axis
        0 makes each input tensor a row, giving:
            [[1.0, 2.0],
             [3.0, 4.0]]
        """
        first = PythonTensor((2,), array("d", [1.0, 2.0]))
        second = PythonTensor((2,), array("d", [3.0, 4.0]))
        result = stack_tensors((first, second), 0, (2, 2))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_stacks_1D_tensors_on_axis_1(self):
        """
        The input tensors are [1.0, 2.0] and [3.0, 4.0]. Stacking on axis
        1 pairs values at the same original position, giving:
            [[1.0, 3.0],
             [2.0, 4.0]]
        """
        first = PythonTensor((2,), array("d", [1.0, 2.0]))
        second = PythonTensor((2,), array("d", [3.0, 4.0]))
        result = stack_tensors((first, second), 1, (2, 2))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.tolist(), [1.0, 3.0, 2.0, 4.0])

    def test_stacks_2D_tensors_on_axis_0(self):
        """
        The first tensor has shape (2, 3) and values:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

        The second tensor has shape (2, 3) and values:
            [[7.0, 8.0, 9.0],
             [10.0, 11.0, 12.0]]

        Stacking on axis 0 makes each input tensor one item on the new
        leading axis, giving:
            [[[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]],
             [[7.0, 8.0, 9.0],
              [10.0, 11.0, 12.0]]]
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor((2, 3), array("d", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
        result = stack_tensors((first, second), 0, (2, 2, 3))
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(
            result.data.tolist(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )

    def test_stacks_2D_tensors_on_axis_1(self):
        """
        The input tensors are:
            first:  [[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]]
            second: [[7.0, 8.0, 9.0],
                     [10.0, 11.0, 12.0]]

        Stacking on axis 1 groups rows from the same original row position,
        giving:
            [[[1.0, 2.0, 3.0],
              [7.0, 8.0, 9.0]],
             [[4.0, 5.0, 6.0],
              [10.0, 11.0, 12.0]]]
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor((2, 3), array("d", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
        result = stack_tensors((first, second), 1, (2, 2, 3))
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(
            result.data.tolist(),
            [1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
        )

    def test_stacks_2D_tensors_on_axis_2(self):
        """
        The input tensors are:
            first:  [[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]]
            second: [[7.0, 8.0, 9.0],
                     [10.0, 11.0, 12.0]]

        Stacking on axis 2 pairs individual values from the same original
        row and column position, giving:
            [[[1.0, 7.0],
              [2.0, 8.0],
              [3.0, 9.0]],
             [[4.0, 10.0],
              [5.0, 11.0],
              [6.0, 12.0]]]
        """
        first = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        second = PythonTensor((2, 3), array("d", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
        result = stack_tensors((first, second), 2, (2, 3, 2))
        self.assertEqual(result.shape, (2, 3, 2))
        self.assertEqual(
            result.data.tolist(),
            [1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0],
        )

    def test_stacks_more_than_two_3D_tensors(self):
        """
        Each input tensor has shape (2, 1, 2):
            first:  [[[1.0, 2.0]], [[3.0, 4.0]]]
            second: [[[5.0, 6.0]], [[7.0, 8.0]]]
            third:  [[[9.0, 10.0]], [[11.0, 12.0]]]

        Stacking on axis 2 inserts a new axis inside each inner group,
        giving:
            [[[[1.0, 2.0],
               [5.0, 6.0],
               [9.0, 10.0]]],
             [[[3.0, 4.0],
               [7.0, 8.0],
               [11.0, 12.0]]]]
        """
        first = PythonTensor((2, 1, 2), array("d", [1.0, 2.0, 3.0, 4.0]))
        second = PythonTensor((2, 1, 2), array("d", [5.0, 6.0, 7.0, 8.0]))
        third = PythonTensor((2, 1, 2), array("d", [9.0, 10.0, 11.0, 12.0]))
        result = stack_tensors((first, second, third), 2, (2, 1, 3, 2))
        self.assertEqual(result.shape, (2, 1, 3, 2))
        self.assertEqual(
            result.data.tolist(),
            [
                1.0,
                2.0,
                5.0,
                6.0,
                9.0,
                10.0,
                3.0,
                4.0,
                7.0,
                8.0,
                11.0,
                12.0,
            ],
        )

    def test_accepts_singleton_sequence(self):
        tensor = PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0]))
        result = stack_tensors((tensor,), 2, (2, 2, 1))
        self.assertEqual(result.shape, (2, 2, 1))
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_uses_tensor_layout_when_a_source_tensor_is_a_view(self):
        """
        The parent tensor has shape (2, 3) and values:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

        The view has shape (3, 2) and strides (1, 3), so it reads the same
        buffer as:
            [[1.0, 4.0],
             [2.0, 5.0],
             [3.0, 6.0]]

        Stacking the view above the second tensor proves that stack reads
        the source tensor's logical layout, not its raw buffer order.
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        view = tensor.view((3, 2), strides=(1, 3))
        second = PythonTensor((3, 2), array("d", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        result = stack_tensors((view, second), 0, (2, 3, 2))
        self.assertEqual(result.shape, (2, 3, 2))
        self.assertEqual(
            result.data.tolist(),
            [
                1.0,
                4.0,
                2.0,
                5.0,
                3.0,
                6.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
            ],
        )

    def test_stacks_tensors_with_zero_length_dimension(self):
        """
        Each source tensor has shape (2, 0, 3). Stacking two of them on
        axis 0 inserts a new leading axis of length 2, giving shape
        (2, 2, 0, 3). Stacking two of them on axis 2 inserts the new axis
        between the existing 0 and 3 dimensions, giving shape (2, 0, 2, 3).

        In each case none of the input or output tensors can have any values.
        """
        cases = (
            ((2, 0, 3), 0, (2, 2, 0, 3)),
            ((2, 0, 3), 2, (2, 0, 2, 3)),
        )
        for source_shape, axis, result_shape in cases:
            with self.subTest(source_shape=source_shape, axis=axis):
                first = PythonTensor(source_shape, array("d"))
                second = PythonTensor(source_shape, array("d"))
                result = stack_tensors((first, second), axis, result_shape)
                self.assertEqual(result.shape, result_shape)
                self.assertEqual(result.data.tolist(), [])


class TestMapUnary(unittest.TestCase):

    def test_applies_operation_to_each_value(self):
        tensor = PythonTensor((2, 2), array("d", [1.0, 2.5, 4.0, 8.5]))
        result = map_unary(tensor, lambda value: value * 2.0 + 0.5)
        self.assertEqual(result.data.tolist(), [2.5, 5.5, 8.5, 17.5])

    def test_returns_tensor_with_same_shape(self):
        cases = (
            PythonTensor((3,), array("d", [1.0, 2.0, 3.0])),
            PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0])),
            PythonTensor((2, 2, 1), array("d", [1.0, 2.0, 3.0, 4.0])),
        )
        for tensor in cases:
            with self.subTest(shape=tensor.shape):
                result = map_unary(tensor, lambda value: value + 1.0)
                self.assertEqual(result.shape, tensor.shape)

    def test_uses_tensor_layout_when_tensor_is_view(self):
        """
        The parent tensor has shape (2, 3) and values:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

        The view has shape (3, 2) and strides (1, 3), so it reads the same
        buffer as:
            [[1.0, 4.0],
             [2.0, 5.0],
             [3.0, 6.0]]

        The operation is applied to values in that logical view order, so the
        result contains those values plus 10.0.
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        view = tensor.view((3, 2), strides=(1, 3))
        result = map_unary(view, lambda value: value + 10.0)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.data.tolist(), [11.0, 14.0, 12.0, 15.0, 13.0, 16.0])

    def test_returns_empty_tensor_with_same_shape(self):
        cases = (
            PythonTensor((0,), array("d")),
            PythonTensor((2, 0), array("d")),
            PythonTensor((2, 0, 3), array("d")),
        )
        for tensor in cases:
            with self.subTest(shape=tensor.shape):
                result = map_unary(tensor, lambda value: value + 1.0)
                self.assertEqual(result.shape, tensor.shape)
                self.assertEqual(result.data.tolist(), [])


class TestMapBinary(unittest.TestCase):

    def test_applies_operation_to_tensors_with_same_shape(self):
        left = PythonTensor((2, 2), array("d", [1.0, 2.5, 4.0, 8.5]))
        right = PythonTensor((2, 2), array("d", [10.0, 20.0, 30.0, 40.0]))
        result = map_binary(left, right, lambda a, b: a + b)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.tolist(), [11.0, 22.5, 34.0, 48.5])

    def test_applies_operation_to_tensor_and_scalar(self):
        tensor = PythonTensor((2, 2), array("d", [1.0, 2.5, 4.0, 8.5]))
        result = map_binary(tensor, 2.0, lambda a, b: a * b)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.tolist(), [2.0, 5.0, 8.0, 17.0])

    def test_broadcasts_1D_tensor_against_2D_tensor(self):
        """
        The left tensor has shape (2, 3) and values:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

        The right tensor has shape (3,) and values [10.0, 20.0, 30.0].
        It is reused across each row, so the result is:
            [[11.0, 22.0, 33.0],
             [14.0, 25.0, 36.0]]
        """
        left = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        right = PythonTensor((3,), array("d", [10.0, 20.0, 30.0]))
        result = map_binary(left, right, lambda a, b: a + b)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data.tolist(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])

    def test_broadcasts_2D_tensors_when_one_axis_has_length_1(self):
        """
        The left tensor has shape (2, 3) and values:
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

        The right tensor has shape (2, 1) and values:
            [[10.0],
             [20.0]]

        Each right-hand value is reused across its row, so the result is:
            [[11.0, 12.0, 13.0],
             [24.0, 25.0, 26.0]]
        """
        left = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        right = PythonTensor((2, 1), array("d", [10.0, 20.0]))
        result = map_binary(left, right, lambda a, b: a + b)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data.tolist(), [11.0, 12.0, 13.0, 24.0, 25.0, 26.0])

    def test_broadcasts_3D_tensors_when_one_axis_has_length_1(self):
        """
        The left tensor has shape (2, 1, 3). It has one 1D row for each
        position on axis 0:
            [[[1.0, 2.0, 3.0]],
             [[4.0, 5.0, 6.0]]]

        The right tensor has shape (2, 2, 3):
            [[[10.0, 20.0, 30.0],
              [40.0, 50.0, 60.0]],
             [[70.0, 80.0, 90.0],
              [100.0, 110.0, 120.0]]]

        The left tensor's middle axis is reused to match the right tensor's
        middle axis of length 2. It is treated as if it had shape
        (2, 2, 3) and values:
            [[[1.0, 2.0, 3.0],
              [1.0, 2.0, 3.0]],
             [[4.0, 5.0, 6.0],
              [4.0, 5.0, 6.0]]]

        For axis-0 position 0, [1.0, 2.0, 3.0] is therefore added to both
        [10.0, 20.0, 30.0] and [40.0, 50.0, 60.0]. For axis-0 position 1,
        [4.0, 5.0, 6.0] is added to both [70.0, 80.0, 90.0] and
        [100.0, 110.0, 120.0].

        The result is therefore:
            [[[11.0, 22.0, 33.0],
              [41.0, 52.0, 63.0]],
             [[74.0, 85.0, 96.0],
              [104.0, 115.0, 126.0]]]
        """
        left = PythonTensor((2, 1, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        right = PythonTensor(
            (2, 2, 3),
            array(
                "d",
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                    70.0,
                    80.0,
                    90.0,
                    100.0,
                    110.0,
                    120.0,
                ],
            ),
        )
        result = map_binary(left, right, lambda a, b: a + b)
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(
            result.data.tolist(),
            [
                11.0,
                22.0,
                33.0,
                41.0,
                52.0,
                63.0,
                74.0,
                85.0,
                96.0,
                104.0,
                115.0,
                126.0,
            ],
        )

    def test_uses_tensor_layout_when_left_tensor_is_view(self):
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        view = tensor.view((3, 2), strides=(1, 3))
        right = PythonTensor((3, 2), array("d", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        result = map_binary(view, right, lambda a, b: a + b)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.data.tolist(), [11.0, 24.0, 32.0, 45.0, 53.0, 66.0])

    def test_uses_tensor_layout_when_right_tensor_is_view(self):
        """
        Mirrors test_uses_tensor_layout_when_left_tensor_is_view, but with
        the view on the right-hand operand instead, so both sides of
        map_binary are proven to read through a tensor's layout rather than
        its raw buffer order.
        """
        left = PythonTensor((3, 2), array("d", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        view = tensor.view((3, 2), strides=(1, 3))
        result = map_binary(left, view, lambda a, b: a + b)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.data.tolist(), [11.0, 24.0, 32.0, 45.0, 53.0, 66.0])

    def test_returns_empty_tensor_when_broadcast_result_has_zero_length_dimension(self):
        left = PythonTensor((2, 0, 3), array("d"))
        right = PythonTensor((1, 0, 3), array("d"))
        result = map_binary(left, right, lambda a, b: a + b)
        self.assertEqual(result.shape, (2, 0, 3))
        self.assertEqual(result.data.tolist(), [])

    def test_raises_when_scalar_is_bool(self):
        tensor = PythonTensor((2,), array("d", [1.0, 2.0]))
        for scalar in (True, False):
            with self.subTest(scalar=scalar):
                with self.assertRaisesRegex(
                    ValueError, "scalar value must not be a bool"
                ):
                    map_binary(tensor, scalar, lambda a, b: a + b)


class TestDivideReductionResult(unittest.TestCase):

    def test_divides_scalar_result(self):
        result = divide_reduction_result(9.0, 3.0)
        self.assertEqual(result, 3.0)

    def test_divides_each_value_in_tensor_result(self):
        tensor = PythonTensor((2, 2), array("d", [2.0, 4.0, 6.0, 8.0]))
        result = divide_reduction_result(tensor, 2.0)
        self.assertIsInstance(result, PythonTensor)
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_returns_same_tensor_instance_when_result_is_tensor(self):
        tensor = PythonTensor((2, 2), array("d", [2.0, 4.0, 6.0, 8.0]))
        result = divide_reduction_result(tensor, 2.0)
        self.assertIs(result, tensor)


class TestReduceToScalar(unittest.TestCase):

    def test_accumulates_all_values_into_scalar(self):
        tensor = PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0]))
        result = reduce_to_scalar(
            tensor,
            0.0,
            lambda total, value: total + value,
        )
        self.assertEqual(result, 10.0)

    def test_uses_initial_value_when_accumulating_values(self):
        """
        The accumulator starts from the initial value passed by the caller,
        so that value is included before any tensor values are combined.
        """
        tensor = PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0]))
        result = reduce_to_scalar(
            tensor,
            10.0,
            lambda total, value: total + value,
        )
        self.assertEqual(result, 20.0)

    def test_accepts_non_sum_accumulation_function(self):
        cases = (
            (
                PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0])),
                20.0,
                lambda total, value: total - value,
                10.0,
            ),
            (
                PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0])),
                1.0,
                operator.mul,
                24.0,
            ),
        )
        for tensor, initial_value, accumulate_fn, expected in cases:
            with self.subTest():
                result = reduce_to_scalar(
                    tensor,
                    initial_value,
                    accumulate_fn,
                )
                self.assertEqual(result, expected)


class TestReduceToTensor(unittest.TestCase):

    def test_accumulates_values_into_tensor_when_reducing_single_axis(self):
        """
        This tests reducing a 2D tensor over axis 0.

        Values whose positions differ only on axis 0 are combined, so the
        result values are calculated from the columns:
        - 1.0 + 4.0 = 5.0
        - 2.0 + 5.0 = 7.0
        - 3.0 + 6.0 = 9.0

        The result is a 1D tensor, containing these values.
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        result = reduce_to_tensor(
            tensor,
            (0,),
            (3,),
            False,
            0.0,
            lambda total, value: total + value,
        )
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.data.tolist(), [5.0, 7.0, 9.0])

    def test_accumulates_values_into_tensor_when_reducing_middle_axis(self):
        """
        This tests reducing a 3D tensor over its middle axis.

        Values whose positions differ only on axis 1 are combined:
        - (0, 0, 0) and (0, 1, 0) -> 1.0 + 3.0 = 4.0
        - (0, 0, 1) and (0, 1, 1) -> 2.0 + 4.0 = 6.0
        - (1, 0, 0) and (1, 1, 0) -> 5.0 + 7.0 = 12.0
        - (1, 0, 1) and (1, 1, 1) -> 6.0 + 8.0 = 14.0

        The result is a 2D tensor containing these values.
        """
        tensor = PythonTensor(
            (2, 2, 2),
            array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        result = reduce_to_tensor(
            tensor,
            (1,),
            (2, 2),
            False,
            0.0,
            lambda total, value: total + value,
        )
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.tolist(), [4.0, 6.0, 12.0, 14.0])

    def test_accumulates_values_into_tensor_when_reducing_multiple_axes(self):
        """
        This tests reducing a 3D tensor over axes 0 and 2.

        Axis 1 is not reduced, so the result has one value for each position
        on axis 1. The first result value combines 1.0, 2.0, 3.0, 7.0, 8.0
        and 9.0. The second combines 4.0, 5.0, 6.0, 10.0, 11.0 and 12.0.
        """
        tensor = PythonTensor(
            (2, 2, 3),
            array(
                "d",
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ),
        )
        result = reduce_to_tensor(
            tensor,
            (0, 2),
            (2,),
            False,
            0.0,
            lambda total, value: total + value,
        )
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.data.tolist(), [30.0, 48.0])

    def test_preserves_reduced_axes_as_length_1_when_keepdims_is_true(self):
        """
        This tests reducing a 3D tensor over axis 1 with keepdims=True.

        The same values are combined as when reducing the middle axis, but
        the reduced axis is retained with length 1. The result therefore has
        shape (2, 1, 3) rather than (2, 3).
        """
        tensor = PythonTensor(
            (2, 2, 3),
            array(
                "d",
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ),
        )
        result = reduce_to_tensor(
            tensor,
            (1,),
            (2, 1, 3),
            True,
            0.0,
            lambda total, value: total + value,
        )
        self.assertEqual(result.shape, (2, 1, 3))
        self.assertEqual(
            result.data.tolist(),
            [5.0, 7.0, 9.0, 17.0, 19.0, 21.0],
        )

    def test_returns_tensor_with_original_shape_when_axes_tuple_is_empty(self):
        """
        This tests the control case where no axes are reduced.

        Each source value maps to the same position in the result tensor, so
        the result has the original shape and the original values.
        """
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        result = reduce_to_tensor(
            tensor,
            (),
            (2, 3),
            False,
            0.0,
            lambda total, value: total + value,
        )
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
