import unittest
import math
from array import array

from fnn.tensors.python_backend.python_tensor import PythonTensor
from fnn.tensors.python_backend.validation import (
    require_non_empty_tensor_sequence,
    validate_stack_shapes,
)


def make_tensor(shape):
    return PythonTensor(
        shape, array("d", (float(index) for index in range(math.prod(shape))))
    )


class TestRequireNonEmptyTensorSequence(unittest.TestCase):

    def test_returns_tuple_containing_tensors(self):
        first = PythonTensor((2,), array("d", [1.0, 2.0]))
        second = PythonTensor((2,), array("d", [3.0, 4.0]))
        result = require_non_empty_tensor_sequence([first, second])
        self.assertEqual(result, (first, second))

    def test_accepts_non_list_sequence(self):
        first = PythonTensor((2,), array("d", [1.0, 2.0]))
        second = PythonTensor((2,), array("d", [3.0, 4.0]))
        result = require_non_empty_tensor_sequence((first, second))
        self.assertEqual(result, (first, second))

    def test_raises_when_sequence_is_empty(self):
        with self.assertRaisesRegex(ValueError, "tensor sequence must not be empty"):
            require_non_empty_tensor_sequence([])


class TestValidateStackShapes(unittest.TestCase):

    def test_accepts_tensors_with_same_shape(self):
        cases = ((2,), (2, 3), (2, 1, 3), (2, 3, 4))
        for shape in cases:
            with self.subTest(shape=shape):
                first = make_tensor(shape)
                second_shape = tuple(dim for dim in shape)
                second = make_tensor(second_shape)
                self.assertIsNot(first.shape, second.shape)
                result = validate_stack_shapes((first, second))
                self.assertIsNone(result)

    def test_accepts_single_tensor(self):
        cases = ((1,), (2, 3), (2, 1, 3), (2, 0))
        for shape in cases:
            with self.subTest(shape=shape):
                tensor = make_tensor(shape)
                result = validate_stack_shapes((tensor,))
                self.assertIsNone(result)

    def test_accepts_tensors_with_same_zero_length_shape(self):
        cases = ((0,), (2, 0), (0, 2), (2, 0, 3), (2, 3, 0))
        for shape in cases:
            with self.subTest(shape=shape):
                first = make_tensor(shape)
                second_shape = tuple(dim for dim in shape)
                second = make_tensor(second_shape)
                self.assertIsNot(first.shape, second.shape)
                result = validate_stack_shapes((first, second))
                self.assertIsNone(result)

    def test_raises_when_tensor_shapes_differ(self):
        cases = (
            ((2,), (3,)),
            ((2, 3), (3, 2)),
            ((2, 3), (2, 3, 1)),
            ((2, 1, 3), (2, 2, 3)),
            ((2, 0), (2, 1)),
        )
        for first_shape, second_shape in cases:
            with self.subTest(first_shape=first_shape, second_shape=second_shape):
                first = make_tensor(first_shape)
                second = make_tensor(second_shape)
                with self.assertRaisesRegex(
                    ValueError, "all tensors must have the same shape"
                ):
                    validate_stack_shapes((first, second))
