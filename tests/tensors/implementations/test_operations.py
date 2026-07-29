import operator
import unittest
from array import array

from src.tensors.python_backend.operations import (
    divide_reduction_result,
    reduce_to_scalar,
    reduce_to_tensor,
)
from src.tensors.python_backend.python_tensor import PythonTensor


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
