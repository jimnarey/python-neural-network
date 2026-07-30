import operator
import unittest
from array import array

from src.tensors.python_backend.operations import (
    argmax_to_scalar,
    argmax_to_tensor,
    divide_reduction_result,
    first_max_index,
    reduce_to_scalar,
    reduce_to_tensor,
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
