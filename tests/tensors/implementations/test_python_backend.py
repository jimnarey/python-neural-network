import unittest
from array import array

from tests.helpers.tensor_helpers import all_values_are_floats
from src.tensors.protocol import TensorBackend
from src.tensors.python_backend.python_backend import PythonBackend
from src.tensors.python_backend.python_tensor import PythonTensor

from tests.tensors.contract.argmax import (
    BackendContractArgMaxAxisArgumentMixin,
    BackendContractArgMaxKeepdimsMixin,
    BackendContractArgMaxSemanticsMixin,
    BackendContractArgMaxTieBehaviourMixin,
)

from tests.tensors.contract.composition import (
    BackendContractConcatenateSemanticsMixin,
    BackendContractStackSemanticsMixin,
)

from tests.tensors.contract.creation import (
    BackendContractCopyMixin,
    BackendContractCreationInputValidationMixin,
    BackendContractCreationZeroLengthDimensionMixin,
    BackendContractEmptyMixin,
    BackendContractEyeMixin,
    BackendContractLikeCreationMixin,
    BackendContractZerosOnesAndFullMixin,
)

from tests.tensors.contract.elementwise import (
    BackendContractElementwiseDualBroadcastingMixin,
    BackendContractElementwiseLeftPaddingBroadcastingMixin,
    BackendContractElementwiseLengthOneAxisBroadcastingMixin,
    BackendContractElementwiseSemanticsMixin,
)

from tests.tensors.contract.randn import BackendContractRandnMixin

from tests.tensors.contract.reduction import (
    BackendContractReductionBehaviourMixin,
    BackendContractReductionEmptyInputMixin,
    BackendContractReductionInvalidAxisMixin,
    BackendContractReductionKeepdimsMixin,
)

from tests.tensors.contract.reshape import BackendContractReshapeMixin

from tests.tensors.contract.transpose import BackendContractTransposeMixin

from tests.tensors.contract.unary import (
    BackendContractAbsoluteSemanticsMixin,
    BackendContractClipSemanticsMixin,
    BackendContractSignSemanticsMixin,
    BackendContractSqrtSemanticsMixin,
    BackendContractUnaryShapeMixin,
    BackendContractUnaryZeroLengthDimensionMixin,
)

from tests.tensors.reference.creation import (
    BackendReferenceCopyMixin,
    BackendReferenceCreationLikeValueTypeMixin,
    BackendReferenceCreationValueTypeMixin,
)

from tests.tensors.reference.elementwise import (
    BackendReferenceElementwiseArithmeticMixin,
    BackendReferenceElementwiseFloatValueMixin,
    BackendReferenceElementwiseSpecialValueMixin,
)

from tests.tensors.reference.randn import BackendReferenceRandnMixin

from tests.tensors.reference.reduction import (
    BackendReferenceReductionArithmeticMixin,
    BackendReferenceReductionFloatValueMixin,
)

from tests.tensors.reference.unary import (
    BackendReferenceExpArithmeticMixin,
    BackendReferenceLogArithmeticMixin,
    BackendReferenceLogSpecialValueMixin,
    BackendReferenceSqrtArithmeticMixin,
    BackendReferenceSqrtSpecialValueMixin,
    BackendReferenceUnaryValueTypeMixin,
)


class TestPythonBackendProtocolConformance(unittest.TestCase):
    """
    This is a safety check so that if the codebase ever temporarily or
    permanently does not pass PythonBackend to a layer or other consumer
    the type checker will still catch deviations from the protocol/contract
    """

    # It is essential to set the return type here if we want mypy to type check
    # the instantiation of PythonBackend
    def test_python_backend_implements_tensor_backend_protocol(self) -> None:
        # mypy check
        backend: TensorBackend[PythonTensor] = PythonBackend()
        # Test at runtime
        self.assertIsInstance(backend, TensorBackend)


class PythonBackendTestCase(unittest.TestCase):

    def make_backend(self, seed: int | None = None) -> TensorBackend[PythonTensor]:
        return PythonBackend(seed=seed)


class TestPythonBackendContract(
    PythonBackendTestCase,
    BackendContractArgMaxAxisArgumentMixin,
    BackendContractArgMaxKeepdimsMixin,
    BackendContractArgMaxSemanticsMixin,
    BackendContractArgMaxTieBehaviourMixin,
    BackendContractAbsoluteSemanticsMixin,
    BackendContractClipSemanticsMixin,
    BackendContractConcatenateSemanticsMixin,
    BackendContractStackSemanticsMixin,
    BackendContractCopyMixin,
    BackendContractCreationInputValidationMixin,
    BackendContractCreationZeroLengthDimensionMixin,
    BackendContractEmptyMixin,
    BackendContractEyeMixin,
    BackendContractLikeCreationMixin,
    BackendContractZerosOnesAndFullMixin,
    BackendContractRandnMixin,
    BackendContractReshapeMixin,
    BackendContractSignSemanticsMixin,
    BackendContractSqrtSemanticsMixin,
    BackendContractTransposeMixin,
    BackendContractUnaryShapeMixin,
    BackendContractUnaryZeroLengthDimensionMixin,
    BackendContractElementwiseDualBroadcastingMixin,
    BackendContractElementwiseLeftPaddingBroadcastingMixin,
    BackendContractElementwiseLengthOneAxisBroadcastingMixin,
    BackendContractElementwiseSemanticsMixin,
    BackendContractReductionBehaviourMixin,
    BackendContractReductionEmptyInputMixin,
    BackendContractReductionInvalidAxisMixin,
    BackendContractReductionKeepdimsMixin,
):
    pass


class TestPythonBackendReference(
    PythonBackendTestCase,
    BackendReferenceCopyMixin,
    BackendReferenceCreationLikeValueTypeMixin,
    BackendReferenceCreationValueTypeMixin,
    BackendReferenceRandnMixin,
    BackendReferenceExpArithmeticMixin,
    BackendReferenceLogArithmeticMixin,
    BackendReferenceLogSpecialValueMixin,
    BackendReferenceSqrtArithmeticMixin,
    BackendReferenceSqrtSpecialValueMixin,
    BackendReferenceUnaryValueTypeMixin,
    BackendReferenceElementwiseArithmeticMixin,
    BackendReferenceElementwiseFloatValueMixin,
    BackendReferenceElementwiseSpecialValueMixin,
    BackendReferenceReductionArithmeticMixin,
    BackendReferenceReductionFloatValueMixin,
):
    pass


class TestPythonBackendFloatValuedTensorCreation(PythonBackendTestCase):

    def _assert_is_float_typed_python_tensor(self, tensor):
        self.assertIsInstance(tensor, PythonTensor)
        self.assertEqual(tensor.data.typecode, "d")

    def test_shape_based_creation_methods_return_float_typed_python_tensors(self):
        backend = self.make_backend(seed=0)

        creation_methods = [
            ("randn", lambda: backend.randn((2, 3))),
            ("zeros", lambda: backend.zeros((2, 3))),
            ("ones", lambda: backend.ones((2, 3))),
            ("full", lambda: backend.full((2, 3), 7)),
            ("empty", lambda: backend.empty((2, 3))),
            ("eye", lambda: backend.eye(3)),
        ]

        for method_name, call in creation_methods:
            with self.subTest(method=method_name):
                self._assert_is_float_typed_python_tensor(call())

    def test_tensor_based_creation_methods_return_float_typed_python_tensors(self):
        backend = self.make_backend()
        tensor = PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0]))

        creation_methods = [
            ("zeros_like", lambda: backend.zeros_like(tensor)),
            ("ones_like", lambda: backend.ones_like(tensor)),
            ("full_like", lambda: backend.full_like(tensor, 7)),
            ("empty_like", lambda: backend.empty_like(tensor)),
            ("copy", lambda: backend.copy(tensor)),
        ]

        for method_name, call in creation_methods:
            with self.subTest(method=method_name):
                self._assert_is_float_typed_python_tensor(call())


class TestPythonBackendToTensor(PythonBackendTestCase):

    def test_to_tensor_converts_1D_input_to_expected_python_tensor(self):
        backend = self.make_backend()
        result = backend.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.assertIsInstance(result, PythonTensor)
        self.assertEqual(result.shape, (4,))
        self.assertEqual(result.strides, (1,))
        self.assertEqual(result.offset, 0)
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_to_tensor_converts_2D_input_to_expected_python_tensor(self):
        backend = self.make_backend()
        result = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertIsInstance(result, PythonTensor)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.strides, (3, 1))
        self.assertEqual(result.offset, 0)
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_to_tensor_converts_3D_input_to_expected_python_tensor(self):
        backend = self.make_backend()
        result = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        self.assertIsInstance(result, PythonTensor)
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertEqual(result.strides, (4, 2, 1))
        self.assertEqual(result.offset, 0)
        self.assertEqual(
            result.data.tolist(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )

    def test_to_tensor_converts_4D_input_to_expected_python_tensor(self):
        backend = self.make_backend()
        result = backend.to_tensor(
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
            ]
        )

        self.assertIsInstance(result, PythonTensor)
        self.assertEqual(result.shape, (2, 1, 2, 3))
        self.assertEqual(result.strides, (6, 6, 3, 1))
        self.assertEqual(result.offset, 0)
        self.assertEqual(
            result.data.tolist(),
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
        )

    def test_to_tensor_allows_empty_lists_input(self):
        """
        Tests that a range of tensors with empty dimensions are handled
        by to_tensor. Not all all such tensors can be represented in nested
        Python lists but this covers a decent sample of those which can.
        """
        backend = self.make_backend()
        test_cases = (
            ([], (0,), (1,)),
            ([[]], (1, 0), (0, 1)),
            ([[], []], (2, 0), (0, 1)),
            ([[[]], [[]]], (2, 1, 0), (0, 0, 1)),
        )

        for data, expected_shape, expected_strides in test_cases:
            with self.subTest(data=data):
                result = backend.to_tensor(data)
                self.assertIsInstance(result, PythonTensor)
                self.assertEqual(result.shape, expected_shape)
                self.assertEqual(result.strides, expected_strides)
                self.assertEqual(result.offset, 0)
                self.assertEqual(result.data.tolist(), [])

    def test_to_tensor_returns_float_valued_python_tensor_when_given_integer_input(
        self,
    ):
        backend = self.make_backend()
        result = backend.to_tensor([1, 2, 3])
        self.assertIsInstance(result, PythonTensor)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0])
        self.assertTrue(all_values_are_floats(result.data.tolist()))

    def test_to_tensor_returns_float_valued_python_tensor_when_given_mixed_numeric_input(
        self,
    ):
        backend = self.make_backend()
        result = backend.to_tensor([1, 2.5, 3])
        self.assertIsInstance(result, PythonTensor)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.data.tolist(), [1.0, 2.5, 3.0])
        self.assertTrue(all_values_are_floats(result.data.tolist()))


class TestPythonBackendToPython(PythonBackendTestCase):

    def test_to_python_converts_1D_python_tensor_to_python_list(self):
        backend = self.make_backend()
        tensor = PythonTensor((4,), array("d", [1.0, 2.0, 3.0, 4.0]))
        result = backend.to_python(tensor)
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])

    def test_to_python_converts_2D_python_tensor_to_python_nested_list(self):
        backend = self.make_backend()
        tensor = PythonTensor(
            (2, 3),
            array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        result = backend.to_python(tensor)
        self.assertEqual(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_to_python_converts_3D_python_tensor_to_python_nested_list(self):
        backend = self.make_backend()
        tensor = PythonTensor(
            (2, 2, 2),
            array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        result = backend.to_python(tensor)
        self.assertEqual(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        )

    def test_to_python_converts_4D_python_tensor_to_python_nested_list(self):
        backend = self.make_backend()
        tensor = PythonTensor(
            (2, 1, 2, 3),
            array(
                "d",
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                ],
            ),
        )
        result = backend.to_python(tensor)
        self.assertEqual(
            result,
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
            ],
        )

    def test_to_python_returns_plain_python_float_values(self):
        backend = self.make_backend()
        tensor = PythonTensor((3,), array("d", [1.0, 2.0, 3.0]))
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_to_python_converts_empty_python_tensors_to_nested_lists(self):
        backend = self.make_backend()
        test_cases = (
            (PythonTensor((0,)), []),
            (PythonTensor((0, 3)), []),
            (PythonTensor((2, 0, 3)), [[], []]),
            (PythonTensor((2, 3, 0)), [[[], [], []], [[], [], []]]),
        )
        for tensor, expected in test_cases:
            with self.subTest(shape=tensor.shape):
                result = backend.to_python(tensor)
                self.assertEqual(result, expected)


class TestPythonBackendShape(PythonBackendTestCase):
    """
    Test the Python backend's shape method since it is relied upon in the
    backend contract tests.

    The value here is in covering native tensors created manually, especially
    those with zero-length dimensions that cannot always be distinguished
    through to_python.
    """

    def test_shape_returns_expected_tuple_for_1D_tensor(self):
        backend = self.make_backend()
        test_cases = [
            ("length_3", PythonTensor((3,)), (3,)),
            ("zero_length", PythonTensor((0,)), (0,)),
        ]
        for case_name, tensor, expected_shape in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, expected_shape)

    def test_shape_returns_expected_tuple_for_2D_tensor(self):
        backend = self.make_backend()
        test_cases = [
            ("two_by_three", PythonTensor((2, 3)), (2, 3)),
            ("two_by_zero", PythonTensor((2, 0)), (2, 0)),
        ]
        for case_name, tensor, expected_shape in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, expected_shape)

    def test_shape_returns_expected_tuple_for_3D_tensor(self):
        backend = self.make_backend()
        test_cases = [
            ("two_by_three_by_two", PythonTensor((2, 3, 2)), (2, 3, 2)),
            ("two_by_zero_by_three", PythonTensor((2, 0, 3)), (2, 0, 3)),
        ]
        for case_name, tensor, expected_shape in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, expected_shape)

    def test_shape_returns_expected_tuple_for_4D_tensor(self):
        backend = self.make_backend()
        test_cases = [
            (
                "one_by_two_by_three_by_four",
                PythonTensor((1, 2, 3, 4)),
                (1, 2, 3, 4),
            ),
            (
                "two_by_one_by_four_by_three",
                PythonTensor((2, 1, 4, 3)),
                (2, 1, 4, 3),
            ),
            (
                "three_by_two_by_zero_by_one",
                PythonTensor((3, 2, 0, 1)),
                (3, 2, 0, 1),
            ),
        ]
        for case_name, tensor, expected_shape in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, expected_shape)


class TestPythonBackendFirstMaxIndex(PythonBackendTestCase):

    def test_returns_index_of_largest_value(self):
        values = [2.0, 9.0, 4.0, 7.0]
        result = PythonBackend._first_max_index(values)
        self.assertEqual(result, 1)

    def test_returns_first_index_when_maximum_value_is_tied(self):
        values = [2.0, 9.0, 4.0, 9.0]
        result = PythonBackend._first_max_index(values)
        self.assertEqual(result, 1)

    def test_consumes_one_pass_iterable(self):
        values = (value for value in [2.0, 4.0, 9.0, 7.0])
        result = PythonBackend._first_max_index(values)
        self.assertEqual(result, 2)


class TestPythonBackendArgmaxToScalar(PythonBackendTestCase):

    def test_returns_flat_index_for_1D_tensor(self):
        tensor = PythonTensor((4,), array("d", [2.0, 9.0, 4.0, 7.0]))
        result = PythonBackend._argmax_to_scalar(tensor)
        self.assertEqual(result, 1)

    def test_returns_flat_index_for_2D_tensor(self):
        tensor = PythonTensor((2, 3), array("d", [2.0, 4.0, 7.0, 9.0, 6.0, 8.0]))
        result = PythonBackend._argmax_to_scalar(tensor)
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
        result = PythonBackend._argmax_to_scalar(view)
        self.assertEqual(result, 2)

    def test_returns_first_flat_index_when_maximum_value_is_tied(self):
        tensor = PythonTensor((2, 3), array("d", [2.0, 9.0, 7.0, 9.0, 6.0, 8.0]))
        result = PythonBackend._argmax_to_scalar(tensor)
        self.assertEqual(result, 1)


class TestPythonBackendArgmaxToTensor(PythonBackendTestCase):

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
        result = PythonBackend._argmax_to_tensor(tensor, 0, (3,))
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
        result = PythonBackend._argmax_to_tensor(tensor, 1, (2,))
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
        result = PythonBackend._argmax_to_tensor(tensor, 1, (2, 2))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.tolist(), [1, 2, 1, 2])

    def test_returns_int_valued_tensor(self):
        tensor = PythonTensor((2, 3), array("d", [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]))
        result = PythonBackend._argmax_to_tensor(tensor, 0, (3,))
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
        result = PythonBackend._argmax_to_tensor(tensor, 0, (3,))
        self.assertEqual(result.data.tolist(), [0, 0, 0])


class TestPythonBackendCopy(PythonBackendTestCase):

    def test_copy_does_not_share_values_with_original_after_original_is_mutated(self):
        backend = self.make_backend()
        source_tensor = PythonTensor(
            (2, 3),
            array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        copy_tensor = backend.copy(source_tensor)
        self.assertIsInstance(copy_tensor, PythonTensor)
        source_tensor.data[0] = 0.0
        self.assertEqual(
            copy_tensor.data.tolist(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

    def test_copy_does_not_share_values_with_original_after_copy_is_mutated(self):
        backend = self.make_backend()
        source_tensor = PythonTensor(
            (2, 3),
            array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        copy_tensor = backend.copy(source_tensor)
        self.assertIsInstance(copy_tensor, PythonTensor)
        copy_tensor.data[0] = 0.0
        self.assertEqual(
            source_tensor.data.tolist(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
