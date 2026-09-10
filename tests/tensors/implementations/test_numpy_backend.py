"""Test module for the NumPy backend

This contains a runner class for the shared backend contract tests
and one for the shared reference design tests.

It also has tests for to_tensor, to_python and shape, which it is
critical be tested thoroughly as the shared tests rely on them.

We import numpy within each test method individually to enable the
skipUnless checks which apply to each class. It also helps with test
isolation just in case code under test does something like set the
global NumPy seed (though this has been carefully avoided in the way
the backend is constructed).
"""

import importlib.util
import unittest

from fnn.tensors.protocol import TensorBackend
from tests.tensors.contract.argmax import (
    BackendContractArgMaxAxisArgumentMixin,
    BackendContractArgMaxKeepdimsMixin,
    BackendContractArgMaxSemanticsMixin,
    BackendContractArgMaxTieBehaviourMixin,
)
from tests.tensors.contract.chaining import BackendContractRank0ChainingMixin
from tests.tensors.contract.composition import (
    BackendContractConcatenateSemanticsMixin,
    BackendContractStackSemanticsMixin,
)

from tests.tensors.contract.creation import (
    BackendContractCopyMixin,
    BackendContractCreationRankZeroShapeMixin,
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
from tests.tensors.contract.matmul import (
    BackendContractMatmulBroadcastingMixin,
    BackendContractMatmulSemanticsMixin,
)
from tests.tensors.contract.randn import BackendContractRandnMixin
from tests.tensors.contract.reduction import (
    BackendContractReductionBehaviourMixin,
    BackendContractReductionEmptyInputMixin,
    BackendContractReductionInvalidAxisMixin,
    BackendContractReductionKeepdimsMixin,
)
from tests.tensors.contract.reshape import BackendContractReshapeMixin
from tests.tensors.contract.shared import BackendContractConstructionMixin
from tests.tensors.contract.to_python import BackendContractToPythonMixin
from tests.tensors.contract.to_tensor import (
    BackendContractToTensorShapeInputMixin,
    BackendContractToTensorValidInputTypeMixin,
    BackendContractToTensorInvalidInputTypeMixin,
    BackendContractToTensorValueInputMixin,
)
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
from tests.tensors.reference.matmul import (
    BackendReferenceMatmulArithmeticMixin,
    BackendReferenceMatmulFloatValueMixin,
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

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendProtocolConformance(unittest.TestCase):
    """
    This is a safety check so that if the codebase ever temporarily or
    permanently does not pass NumpyBackend to a layer or other consumer
    the type checker will still catch deviations from the protocol/contract
    """

    # It is essential to set the return type here if we want mypy to type check
    # the instantiation of NumpyBackend
    def test_numpy_backend_implements_tensor_backend_protocol(self) -> None:
        from fnn.tensors import NumpyBackend
        from fnn.tensors.numpy_backend import NumpyTensor

        # mypy check
        backend: TensorBackend[NumpyTensor] = NumpyBackend()
        # Test at runtime
        self.assertIsInstance(backend, TensorBackend)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class NumpyBackendTestCase(unittest.TestCase):

    def make_backend(self, seed: int | None = None) -> TensorBackend:
        from fnn.tensors import NumpyBackend

        return NumpyBackend(seed=seed)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendContract(
    NumpyBackendTestCase,
    BackendContractArgMaxAxisArgumentMixin,
    BackendContractArgMaxKeepdimsMixin,
    BackendContractArgMaxSemanticsMixin,
    BackendContractArgMaxTieBehaviourMixin,
    BackendContractRank0ChainingMixin,
    BackendContractConcatenateSemanticsMixin,
    BackendContractStackSemanticsMixin,
    BackendContractAbsoluteSemanticsMixin,
    BackendContractClipSemanticsMixin,
    BackendContractConstructionMixin,
    BackendContractCopyMixin,
    BackendContractCreationRankZeroShapeMixin,
    BackendContractCreationZeroLengthDimensionMixin,
    BackendContractElementwiseDualBroadcastingMixin,
    BackendContractElementwiseLeftPaddingBroadcastingMixin,
    BackendContractElementwiseLengthOneAxisBroadcastingMixin,
    BackendContractElementwiseSemanticsMixin,
    BackendContractEmptyMixin,
    BackendContractEyeMixin,
    BackendContractLikeCreationMixin,
    BackendContractMatmulBroadcastingMixin,
    BackendContractMatmulSemanticsMixin,
    BackendContractRandnMixin,
    BackendContractReductionBehaviourMixin,
    BackendContractReductionEmptyInputMixin,
    BackendContractReductionInvalidAxisMixin,
    BackendContractReductionKeepdimsMixin,
    BackendContractReshapeMixin,
    BackendContractSignSemanticsMixin,
    BackendContractSqrtSemanticsMixin,
    BackendContractToPythonMixin,
    BackendContractToTensorShapeInputMixin,
    BackendContractToTensorValidInputTypeMixin,
    BackendContractToTensorInvalidInputTypeMixin,
    BackendContractToTensorValueInputMixin,
    BackendContractTransposeMixin,
    BackendContractUnaryShapeMixin,
    BackendContractUnaryZeroLengthDimensionMixin,
    BackendContractZerosOnesAndFullMixin,
):
    pass


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendReference(
    NumpyBackendTestCase,
    BackendReferenceCopyMixin,
    BackendReferenceCreationLikeValueTypeMixin,
    BackendReferenceCreationValueTypeMixin,
    BackendReferenceElementwiseArithmeticMixin,
    BackendReferenceElementwiseFloatValueMixin,
    BackendReferenceElementwiseSpecialValueMixin,
    BackendReferenceExpArithmeticMixin,
    BackendReferenceLogArithmeticMixin,
    BackendReferenceLogSpecialValueMixin,
    BackendReferenceMatmulArithmeticMixin,
    BackendReferenceMatmulFloatValueMixin,
    BackendReferenceRandnMixin,
    BackendReferenceReductionArithmeticMixin,
    BackendReferenceReductionFloatValueMixin,
    BackendReferenceSqrtArithmeticMixin,
    BackendReferenceSqrtSpecialValueMixin,
    BackendReferenceUnaryValueTypeMixin,
):
    pass


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendFloatValuedTensorCreation(NumpyBackendTestCase):
    """
    The backend contract does not require tensors to be float valued
    internally but our reference design does. The NumPy backend and
    any future backends designed with inference on x86 CPU/GPU and
    training (it's assumed this is on x86/GPU) should use float-valued
    tensors. This class ensures that this is what we get from the various
    NumPy backend methods which create tensors.

    The to_tensor method is tested separately, since it needs additional
    tests.
    """

    def _assert_is_float_typed_ndarray(self, tensor):
        import numpy as np

        self.assertIsInstance(tensor, np.ndarray)
        self.assertTrue(np.issubdtype(tensor.dtype, np.floating))

    def test_shape_based_creation_methods_return_float_typed_ndarrays(self):
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
                self._assert_is_float_typed_ndarray(call())

    def test_tensor_based_creation_methods_return_float_typed_ndarrays(self):
        import numpy as np

        backend = self.make_backend()
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]])

        creation_methods = [
            ("zeros_like", lambda: backend.zeros_like(tensor)),
            ("ones_like", lambda: backend.ones_like(tensor)),
            ("full_like", lambda: backend.full_like(tensor, 7)),
            ("empty_like", lambda: backend.empty_like(tensor)),
            ("copy", lambda: backend.copy(tensor)),
        ]

        for method_name, call in creation_methods:
            with self.subTest(method=method_name):
                self._assert_is_float_typed_ndarray(call())


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendRank0Handling(NumpyBackendTestCase):
    def test_argmax_returns_rank_0_int_ndarray_when_called_without_axis(self):
        """
        This ensures that the NumPy backend returns a rank-zero ndarray rather
        than a NumPy or Python scalar when argmax produces one index. We're
        looking for a specific NumPy representation here, so this is the right
        place for this test.
        """
        # Some of the assertions here are arguably duplicative of assertions in
        # the (still WIP) backend contract tests. This is fine for now and probably
        # fine forever but do a sense check once the backend contract tests are
        # complete.
        import numpy as np

        backend = self.make_backend()

        result = backend.argmax(np.array([[1.0, 4.0], [3.0, 2.0]]))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, ())
        self.assertTrue(np.issubdtype(result.dtype, np.integer))
        self.assertEqual(result.item(), 1)

    def test_reduction_methods_return_rank_0_ndarrays(self):
        import numpy as np

        backend = self.make_backend()
        tensor = np.array([[1.0, 1.0], [7.0, 7.0]])
        cases = (
            ("sum", lambda: backend.sum(tensor), 16.0),
            ("mean", lambda: backend.mean(tensor), 4.0),
            ("max", lambda: backend.max(tensor), 7.0),
            ("min", lambda: backend.min(tensor), 1.0),
            ("std", lambda: backend.std(tensor), 3.0),
        )
        for method_name, call, expected in cases:
            with self.subTest(method=method_name):
                result = call()
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, ())
                self.assertIs(result.dtype.type, np.float64)
                self.assertEqual(result.item(), expected)

    def test_matmul_returns_rank_0_float_ndarray_for_two_1D_tensors(self):
        import numpy as np

        backend = self.make_backend()
        left = np.array([1.0, 2.0, 3.0])
        right = np.array([4.0, 5.0, 6.0])
        result = backend.matmul(left, right)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, ())
        self.assertIs(result.dtype.type, np.float64)
        self.assertEqual(result.item(), 32.0)

    def test_like_creation_methods_accept_rank_0_tensor(self):
        import numpy as np

        backend = self.make_backend()
        tensor = np.array(3.0)
        cases = [
            ("zeros_like", lambda: backend.zeros_like(tensor), 0.0),
            ("ones_like", lambda: backend.ones_like(tensor), 1.0),
            ("full_like", lambda: backend.full_like(tensor, 7), 7.0),
        ]
        for method_name, call, expected in cases:
            with self.subTest(method=method_name):
                result = call()
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, ())
                self.assertIs(result.dtype.type, np.float64)
                self.assertEqual(result.item(), expected)

    def test_empty_like_accepts_rank_0_tensor(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.empty_like(np.array(3.0))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, ())
        self.assertIs(result.dtype.type, np.float64)

    def test_copy_accepts_rank_0_tensor(self):
        import numpy as np

        backend = self.make_backend()
        tensor = np.array(3.0)
        result = backend.copy(tensor)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, ())
        self.assertIs(result.dtype.type, np.float64)
        self.assertEqual(result.item(), 3.0)
        self.assertIsNot(result, tensor)

    def test_unary_methods_return_rank_0_tensor_when_passed_rank_0_tensor(self):
        import numpy as np

        backend = self.make_backend()
        tensor = np.array(4.0)
        cases = [
            ("exp", lambda: backend.exp(tensor), np.exp(4.0).item()),
            ("log", lambda: backend.log(tensor), np.log(4.0).item()),
            ("sqrt", lambda: backend.sqrt(tensor), 2.0),
            ("absolute", lambda: backend.absolute(np.array(-4.0)), 4.0),
            ("sign", lambda: backend.sign(np.array(-4.0)), -1.0),
            ("clip", lambda: backend.clip(tensor, 1.0, 3.0), 3.0),
        ]
        for method_name, call, expected in cases:
            with self.subTest(method=method_name):
                result = call()
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, ())
                self.assertIs(result.dtype.type, np.float64)
                self.assertEqual(result.item(), expected)

    def test_argmax_returns_rank_0_int_ndarray_when_passed_rank_0_tensor(self):
        import numpy as np

        backend = self.make_backend()
        tensor = np.array(4.0)
        result = backend.argmax(tensor)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, ())
        self.assertTrue(np.issubdtype(result.dtype, np.integer))
        self.assertEqual(result.item(), 0)

    def test_reduction_methods_return_rank_0_ndarrays_when_passed_rank_0_tensor(self):
        import numpy as np

        backend = self.make_backend()
        tensor = np.array(4.0)
        reduction_cases = [
            ("sum", lambda: backend.sum(tensor), 4.0),
            ("mean", lambda: backend.mean(tensor), 4.0),
            ("max", lambda: backend.max(tensor), 4.0),
            ("min", lambda: backend.min(tensor), 4.0),
            ("std", lambda: backend.std(tensor), 0.0),
        ]
        for method_name, call, expected in reduction_cases:
            with self.subTest(method=method_name):
                result = call()
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, ())
                self.assertIs(result.dtype.type, np.float64)
                self.assertEqual(result.item(), expected)

    def test_elementwise_methods_return_rank_0_tensor_when_passed_two_rank_0_tensors(
        self,
    ):
        import numpy as np

        backend = self.make_backend()
        a = np.array(6.0)
        b = np.array(3.0)
        cases = [
            ("add", backend.add, 9.0),
            ("subtract", backend.subtract, 3.0),
            ("multiply", backend.multiply, 18.0),
            ("divide", backend.divide, 2.0),
            ("maximum", backend.maximum, 6.0),
            ("minimum", backend.minimum, 3.0),
        ]
        for method_name, method, expected in cases:
            with self.subTest(method=method_name):
                result = method(a, b)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, ())
                self.assertIs(result.dtype.type, np.float64)
                self.assertEqual(result.item(), expected)

    def test_elementwise_methods_return_rank_0_tensor_when_passed_rank_0_tensor_and_scalar(
        self,
    ):
        import numpy as np

        backend = self.make_backend()
        a = np.array(6.0)
        cases = [
            ("add", backend.add, 9.0),
            ("subtract", backend.subtract, 3.0),
            ("multiply", backend.multiply, 18.0),
            ("divide", backend.divide, 2.0),
            ("maximum", backend.maximum, 6.0),
            ("minimum", backend.minimum, 3.0),
        ]
        for method_name, method, expected in cases:
            with self.subTest(method=method_name):
                result = method(a, 3.0)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, ())
                self.assertIs(result.dtype.type, np.float64)
                self.assertEqual(result.item(), expected)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendToTensor(NumpyBackendTestCase):
    """
    Implementation-level tests for to_tensor

    The backend contract tests are highly reliant on each backend's
    to_tensor implementation. This is an accepted trade-off to enable
    the shared contract tests to work with any backend. It does
    present a risk, however, so we thoroughly test to_tensor here
    where we can inspect the NumPy backend's internal tensor
    representation directly.

    There's a little duplication in here. E.g. an equality check
    between a to_tensor return value and an expected ndarray,
    declared within a test method tells us we have the right
    shape but it doesn't hurt to be explicit.
    """

    def test_to_tensor_converts_float_scalar_to_expected_rank_0_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        for data in (0.0, 3.5, -4.0):
            with self.subTest(data=data):
                result = backend.to_tensor(data)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, ())
                self.assertEqual(result.ndim, 0)
                self.assertEqual(result.size, 1)
                self.assertTrue(np.issubdtype(result.dtype, np.floating))
                self.assertEqual(result.item(), data)

    def test_to_tensor_normalises_int_scalar_in_rank_0_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        for data in (0, 3, -4):
            with self.subTest(data=data):
                result = backend.to_tensor(data)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, ())
                self.assertEqual(result.ndim, 0)
                self.assertEqual(result.size, 1)
                self.assertTrue(np.issubdtype(result.dtype, np.floating))
                self.assertEqual(result.item(), float(data))
                self.assertIs(type(result.item()), float)

    def test_to_tensor_rejects_bool_and_non_numeric_scalar_input(self):
        backend = self.make_backend()
        for data in (True, "data", None):
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, r"numeric values"):
                    backend.to_tensor(data)

    def test_to_tensor_converts_1D_input_to_expected_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([1.0, 2.0, 3.0, 4.0])
        expected = np.array([1.0, 2.0, 3.0, 4.0])

        self.assertIsInstance(result, np.ndarray)
        # Use ndarray's shape attribute, not our backend's shape method
        self.assertEqual(result.shape, (4,))
        self.assertTrue(np.array_equal(result, expected))

    def test_to_tensor_allows_empty_lists_input(self):
        """
        Some, but not all, tensors with empty dimensions can be
        represented in Python using lists and may therefore be
        passed to to_tensor.
        """
        import numpy as np

        backend = self.make_backend()
        test_cases = (
            ([], np.array([])),
            ([[]], np.array([[]])),
            ([[], []], np.array([[], []])),
            ([[[]], [[]]], np.array([[[]], [[]]])),
        )

        for data, expected in test_cases:
            with self.subTest(data=data):
                result = backend.to_tensor(data)

                self.assertIsInstance(result, np.ndarray)
                self.assertTrue(np.array_equal(result, expected))

    def test_to_tensor_does_not_return_rank_0_ndarray_when_given_empty_list_input(self):
        """
        Confirm that when we pass a single, empty list to to_tensor we
        get a rank 1 array with zero elements rather than a rank-zero
        ndarray, which would contain one value.
        """
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([])

        self.assertIsInstance(result, np.ndarray)
        self.assertNotEqual(result.shape, ())
        self.assertEqual(result.shape, (0,))

    def test_to_tensor_converts_2D_input_to_expected_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(np.array_equal(result, expected))

    def test_to_tensor_converts_3D_input_to_expected_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        expected = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertTrue(np.array_equal(result, expected))

    def test_to_tensor_converts_4D_input_to_expected_ndarray(self):
        """
        This test helps ensure that to_tensor is sufficiently
        generalised that it handles higher-dimension tensors.

        We use shape (2, 1, 2, 3) (rather than e.g. (2, 2, 2, 2))
        to ensure that dimensions are mapped properly (though using
        values which are unique within each tensor, then testing for
        equality does this too).

        In practice, we can be sure that this all works because
        to_tensor is a thin wrapper around NumPy's array() method
        but these tests act as a template for any custom, future
        backends and will have counterparts in those backends'
        test classes. It's also not completely impossible that
        future extensions to to_tensor (e.g. more guards) might
        mangle the input before it's passed to array().
        """
        import numpy as np

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
        expected = np.array(
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
            ]
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 1, 2, 3))
        self.assertTrue(np.array_equal(result, expected))

    def test_to_tensor_returns_float_dtype_ndarray_when_given_integer_input(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([1, 2, 3])
        expected = np.array([1.0, 2.0, 3.0])

        self.assertIsInstance(result, np.ndarray)
        # Checks that the ndarray is float typed which is not quite
        # the same as checking every value (but close enough)
        # np.floating is the superclass of np.float64 etc
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertTrue(np.array_equal(result, expected))

    def test_to_tensor_returns_float_dtype_ndarray_when_given_mixed_numeric_input(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([1, 2.5, 3])
        expected = np.array([1.0, 2.5, 3.0])

        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertTrue(np.array_equal(result, expected))

    def test_to_tensor_rejects_ndarray_input(self):
        """
        To catch a possible case whereby another part of the application
        creates a NumPy array and tries to pass it to to_tensor. This may
        not, in practice, be a problem but breaks the backend contract which
        is there to - amongst other things - keep the application as simple
        as possible.
        """
        import numpy as np

        backend = self.make_backend()

        invalid_inputs = [
            np.array([1.0, 2.0, 3.0]),
            np.matrix([[1.0, 2.0], [3.0, 4.0]]),
        ]

        for data in invalid_inputs:
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    r"numeric values",
                    msg=(
                        "to_tensor did not raise ValueError when given a NumPy array-like input"
                    ),
                ):
                    backend.to_tensor(data)

    def test_to_tensor_rejects_numpy_scalar_input(self):
        import numpy as np

        backend = self.make_backend()
        invalid_inputs = [
            np.float64(1.0),
            np.int64(2),
            np.bool_(True),
        ]
        for data in invalid_inputs:
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                    ValueError,
                    r"numeric values",
                    msg="to_tensor did not raise ValueError when given a NumPy scalar",
                ):
                    backend.to_tensor(data)

    def test_to_tensor_rejects_numpy_scalar_values_within_input(self):
        import numpy as np

        backend = self.make_backend()
        invalid_inputs = [
            [np.float64(1.0), 2.0],
            [1.0, np.int64(2)],
            [[1.0, 2.0], [3.0, np.float64(4.0)]],
            [[1.0], [np.bool_(True)]],
        ]

        for data in invalid_inputs:
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                    ValueError,
                    r"numeric values",
                    msg=(
                        "to_tensor did not raise ValueError when given NumPy scalar values within input"
                    ),
                ):
                    backend.to_tensor(data)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendToPython(NumpyBackendTestCase):
    def test_to_python_converts_1D_ndarray_to_python_list(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_python(np.array([1.0, 2.0, 3.0, 4.0]))

        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])

    def test_to_python_converts_2D_ndarray_to_python_nested_list(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_python(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        self.assertEqual(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_to_python_converts_3D_ndarray_to_python_nested_list(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_python(
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            )
        )

        self.assertEqual(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        )

    def test_to_python_converts_4D_ndarray_to_python_nested_list(self):
        """
        As with the counterpart to_tensor test we use an irregular shape
        to ensure that dimensions are mapped correctly.

        Again, we start with a high level of confidence given that to_python
        is a thin wrapper around ndarray.tolist() but the same testing
        considerations apply as they do to to_tensor.
        """
        import numpy as np

        backend = self.make_backend()
        result = backend.to_python(
            np.array(
                [
                    [
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    ],
                    [
                        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    ],
                ]
            )
        )

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
        """
        This test uses floats to create the ndarray passed to
        to_python, reflecting that to_tensor enforces float values
        on tensor creation. So there is no need to test that this
        converts ints. If we had an input tensor to to_python in
        the NumPy backend which contained ints, that would be a
        bug and should be caught elsewhere in the test suite.
        """
        import numpy as np

        backend = self.make_backend()
        result = backend.to_python(np.array([1.0, 2.0, 3.0]))

        for value in result:
            with self.subTest(value=value):
                self.assertIs(type(value), float)

    def test_to_python_converts_rank_0_array_to_python_float(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_python(np.array(1.0))
        self.assertIs(type(result), float)
        self.assertEqual(result, 1.0)

    def test_to_python_rejects_numpy_scalar_values(self):
        """
        This test checks that to_python rejects NumPy scalar values,
        which are not tensors. This doesn't test for non-NumPy types.
        """
        import numpy as np

        backend = self.make_backend()
        invalid_inputs = [
            np.float64(1.0),
            np.int64(2),
        ]
        for data in invalid_inputs:
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    r"NumPy scalar values",
                    msg=(
                        "to_python did not raise ValueError when given a "
                        "NumPy scalar value"
                    ),
                ):
                    backend.to_python(data)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendShape(NumpyBackendTestCase):
    """
    Test the NumPy backend's shape method since it is relied upon
    in the backend contract tests.

    We can be pretty certain that the NumPy backend's shape method works
    because it simply returns ``np.ndarray.shape``. This class also serves
    as a template for implementation-level shape tests in future backends.
    """

    def test_shape_returns_expected_tuple_for_1D_tensor(self):
        import numpy as np

        backend = self.make_backend()
        test_cases = [
            ("length_3", np.empty((3,), dtype=float), (3,)),
            ("zero_length", np.empty((0,), dtype=float), (0,)),
        ]
        for case_name, tensor, expected_shape in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, expected_shape)

    def test_shape_returns_expected_tuple_for_2D_tensor(self):
        import numpy as np

        backend = self.make_backend()
        test_cases = [
            ("two_by_three", np.empty((2, 3), dtype=float), (2, 3)),
            ("two_by_zero", np.empty((2, 0), dtype=float), (2, 0)),
        ]
        for case_name, tensor, expected_shape in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, expected_shape)

    def test_shape_returns_expected_tuple_for_3D_tensor(self):
        import numpy as np

        backend = self.make_backend()
        test_cases = [
            ("two_by_three_by_two", np.empty((2, 3, 2), dtype=float), (2, 3, 2)),
            ("two_by_zero_by_three", np.empty((2, 0, 3), dtype=float), (2, 0, 3)),
        ]
        for case_name, tensor, expected_shape in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, expected_shape)

    def test_shape_returns_expected_tuple_for_4D_tensor(self):
        import numpy as np

        backend = self.make_backend()
        test_cases = [
            (
                "one_by_two_by_three_by_four",
                np.empty((1, 2, 3, 4), dtype=float),
                (1, 2, 3, 4),
            ),
            (
                "two_by_one_by_four_by_three",
                np.empty((2, 1, 4, 3), dtype=float),
                (2, 1, 4, 3),
            ),
            (
                "three_by_two_by_zero_by_one",
                np.empty((3, 2, 0, 1), dtype=float),
                (3, 2, 0, 1),
            ),
        ]
        for case_name, tensor, expected_shape in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, expected_shape)

    def test_shape_returns_empty_tuple_for_rank_0_tensor(self):
        import numpy as np

        backend = self.make_backend()
        test_cases = [
            ("rank_0_float", np.empty((), dtype=float)),
            ("rank_0_int", np.empty((), dtype=int)),
        ]
        for case_name, tensor in test_cases:
            with self.subTest(case=case_name):
                result = backend.shape(tensor)
                self.assertEqual(result, ())

    # This is now covered in test_shape_returns_expected_tuple_for_1D_tensor
    # and can probably be removed.
    def test_shape_accepts_zero_length_tensor(self):
        import numpy as np

        backend = self.make_backend()
        tensor = np.empty((0,), dtype=float)
        result = backend.shape(tensor)
        self.assertEqual(result, (0,))


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendCopy(NumpyBackendTestCase):
    """
    We test that copy does not give us the same Python object in the contrac
    tests but this is not sufficient to know that we do not have separate
    objects referring to the same underlying memory/data.

    This class provides real assurance by mutating the source and copy tensors
    and checking that in each case the other is unchanged.
    """

    def test_copy_does_not_share_values_with_original_after_original_is_mutated(self):
        import numpy as np

        backend = self.make_backend()
        source_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        copy_tensor = backend.copy(source_tensor)
        source_tensor[0, 0] = 0
        self.assertEqual(
            copy_tensor.tolist(),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )

    def test_copy_does_not_share_values_with_original_after_copy_is_mutated(self):
        import numpy as np

        backend = self.make_backend()
        source_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        copy_tensor = backend.copy(source_tensor)
        copy_tensor[0, 0] = 0
        self.assertEqual(
            source_tensor.tolist(),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
