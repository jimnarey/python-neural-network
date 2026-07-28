import math
import operator
import unittest
from array import array

from tests.helpers.tensor_helpers import all_values_are_floats
from src.tensors.protocol import TensorBackend
from src.tensors.python_backend.python_backend import PythonBackend
from src.tensors.python_backend.python_tensor import PythonTensor

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
    BackendContractAbsoluteSemanticsMixin,
    BackendContractClipSemanticsMixin,
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


class TestPythonBackendDivideScalar(PythonBackendTestCase):

    def test_returns_ordinary_division_result_when_divisor_is_not_zero(self):
        cases = (
            (6.0, 3.0, 2.0),
            (-6.0, 3.0, -2.0),
            (6.0, -3.0, -2.0),
            (-6.0, -3.0, 2.0),
        )
        for left, right, expected in cases:
            with self.subTest():
                result = PythonBackend._divide_scalar(left, right)
                self.assertEqual(result, expected)

    def test_returns_nan_when_zero_is_divided_by_zero(self):
        result = PythonBackend._divide_scalar(0.0, 0.0)
        self.assertTrue(math.isnan(result))

    def test_returns_positive_inf_when_positive_value_is_divided_by_zero(self):
        result = PythonBackend._divide_scalar(1.0, 0.0)
        self.assertEqual(result, math.inf)

    def test_returns_negative_inf_when_negative_value_is_divided_by_zero(self):
        result = PythonBackend._divide_scalar(-1.0, 0.0)
        self.assertEqual(result, -math.inf)


class TestPythonBackendDivideReductionResult(PythonBackendTestCase):

    def test_divides_scalar_result(self):
        result = PythonBackend._divide_reduction_result(9.0, 3.0)
        self.assertEqual(result, 3.0)

    def test_divides_each_value_in_tensor_result(self):
        tensor = PythonTensor((2, 2), array("d", [2.0, 4.0, 6.0, 8.0]))
        result = PythonBackend._divide_reduction_result(tensor, 2.0)
        self.assertIsInstance(result, PythonTensor)
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_returns_same_tensor_instance_when_result_is_tensor(self):
        tensor = PythonTensor((2, 2), array("d", [2.0, 4.0, 6.0, 8.0]))
        result = PythonBackend._divide_reduction_result(tensor, 2.0)
        self.assertIs(result, tensor)


class TestPythonBackendReduceToScalar(PythonBackendTestCase):

    def test_accumulates_all_values_into_scalar(self):
        tensor = PythonTensor((2, 2), array("d", [1.0, 2.0, 3.0, 4.0]))
        result = PythonBackend._reduce_to_scalar(
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
        result = PythonBackend._reduce_to_scalar(
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
                result = PythonBackend._reduce_to_scalar(
                    tensor,
                    initial_value,
                    accumulate_fn,
                )
                self.assertEqual(result, expected)


class TestPythonBackendReduceToTensor(PythonBackendTestCase):

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
        result = PythonBackend._reduce_to_tensor(
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
        result = PythonBackend._reduce_to_tensor(
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
        result = PythonBackend._reduce_to_tensor(
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
        result = PythonBackend._reduce_to_tensor(
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
        result = PythonBackend._reduce_to_tensor(
            tensor,
            (),
            (2, 3),
            False,
            0.0,
            lambda total, value: total + value,
        )
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
