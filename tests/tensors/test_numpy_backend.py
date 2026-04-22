"""Test module for the NumPy backend

This contains a runner class for the shared backend contract tests
and one for the shared reference design tests.

It also has tests for to_tensor and to_python, which it is critical
be tested thoroughly as the shared tests rely on them.

We import numpy within each test method individually to enable the
skipUnless checks which apply to each class. It also helps with test
isolation just in case code under test does something like set the
global NumPy seed (though this has been carefully avoided in the way
the backend is constructed).
"""

import importlib.util
import unittest

from src.tensors.tensor_backend import TensorBackend
from tests.tensors.backend_contract_shared import BackendContractConstructionMixin
from tests.tensors.backend_contract_creation import BackendContractCreationMixin

from tests.tensors.backend_contract_to_tensor import (
    BackendContractToTensorTypeInputMixin,
    BackendContractToTensorShapeInputMixin,
)

from tests.tensors.backend_contract_to_python import BackendContractToPythonMixin

from tests.tensors.backend_contract_matmul import (
    BackendContractMatmulSemanticsMixin,
    BackendContractMatmulBroadcastingMixin,
)

from tests.tensors.backend_contract_randn import BackendContractRandnMixin

from tests.tensors.backend_contract_reduction import (
    BackendContractReductionBehaviourMixin,
    BackendContractReductionKeepdimsMixin,
    BackendContractReductionEmptyInputMixin,
    BackendContractReductionInvalidAxisMixin,
)

from tests.tensors.backend_contract_reshape import BackendContractReshapeMixin

from tests.tensors.backend_contract_argmax import BackendContractArgMaxMixin
from tests.tensors.backend_contract_transpose import BackendContractTransposeMixin

from tests.tensors.backend_reference_matmul import BackendReferenceMatmulArithmeticMixin
from tests.tensors.backend_reference_reduction import (
    BackendReferenceReductionArithmeticMixin,
    BackendReferenceReductionFloatValueMixin,
)
from tests.tensors.backend_reference_creation import (
    BackendReferenceCreationValueTypeMixin,
)
from tests.tensors.backend_reference_randn import BackendReferenceRandnMixin

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
        from src.tensors import NumpyBackend

        # mypy check
        backend: TensorBackend = NumpyBackend()
        # Test at runtime
        self.assertIsInstance(backend, TensorBackend)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class NumpyBackendTestCase(unittest.TestCase):

    def make_backend(self, seed: int | None = None) -> TensorBackend:
        from src.tensors import NumpyBackend

        return NumpyBackend(seed=seed)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendContract(
    NumpyBackendTestCase,
    BackendContractConstructionMixin,
    BackendContractCreationMixin,
    BackendContractToTensorTypeInputMixin,
    BackendContractToTensorShapeInputMixin,
    BackendContractToPythonMixin,
    BackendContractRandnMixin,
    BackendContractMatmulSemanticsMixin,
    BackendContractMatmulBroadcastingMixin,
    BackendContractReshapeMixin,
    BackendContractArgMaxMixin,
    BackendContractTransposeMixin,
    BackendContractReductionBehaviourMixin,
    BackendContractReductionKeepdimsMixin,
    BackendContractReductionEmptyInputMixin,
    BackendContractReductionInvalidAxisMixin,
):
    pass


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendReference(
    NumpyBackendTestCase,
    BackendReferenceMatmulArithmeticMixin,
    BackendReferenceReductionFloatValueMixin,
    BackendReferenceReductionArithmeticMixin,
    BackendReferenceCreationValueTypeMixin,
    BackendReferenceRandnMixin,
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
    def test_scalar_returning_methods_do_not_return_rank_0_tensors(self):
        """
        This ensures that the NumPy backend methods which return a single value
        do not do so in the form of a rank 0 array. We're looking for a specific
        NumPy type here, so this is the right place for this test.
        """
        # Some of the assertions here are arguably duplicative of assertions in
        # the (still WIP) backend contract tests. This is fine for now and probably
        # fine forever but do a sense check once the backend contract tests are
        # complete.
        import numpy as np

        backend = self.make_backend()

        scalar_methods = [
            ("argmax", lambda: backend.argmax(np.array([[1.0, 4.0], [3.0, 2.0]]))),
            ("sum", lambda: backend.sum(np.array([[1.0, 2.0], [3.0, 4.0]]))),
            ("mean", lambda: backend.mean(np.array([[1.0, 2.0], [3.0, 4.0]]))),
            ("max", lambda: backend.max(np.array([[1.0, 2.0], [3.0, 4.0]]))),
            ("min", lambda: backend.min(np.array([[1.0, 2.0], [3.0, 4.0]]))),
            ("std", lambda: backend.std(np.array([[1.0, 2.0], [3.0, 4.0]]))),
        ]

        for method_name, call in scalar_methods:
            with self.subTest(method=method_name):
                result = call()
                self.assertNotIsInstance(
                    result,
                    np.ndarray,
                    msg=f"{method_name} returned a rank 0 ndarray: {result!r}",
                )
                self.assertIsInstance(
                    result,
                    (int, float),
                    msg=(
                        f"{method_name} returned {result!r} of type "
                        f"{type(result).__name__}, not a plain Python scalar"
                    ),
                )

    def test_tensor_input_methods_reject_rank_0_tensors(self):
        """
        This test ensures that the methods in the NumPy backend do not
        accept NumPy rank 0 types. This a risk particular to the NumPy
        backend, so the tests go here rather than in the backend contract
        tests.
        """
        import numpy as np

        backend = self.make_backend()
        rank_0 = np.array(1.0)
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        single_tensor_methods = [
            ("zeros_like", lambda: backend.zeros_like(rank_0)),
            ("ones_like", lambda: backend.ones_like(rank_0)),
            ("full_like", lambda: backend.full_like(rank_0, 7)),
            ("empty_like", lambda: backend.empty_like(rank_0)),
            ("copy", lambda: backend.copy(rank_0)),
            ("shape", lambda: backend.shape(rank_0)),
            ("reshape", lambda: backend.reshape(rank_0, (1, 1))),
            ("transpose", lambda: backend.transpose(rank_0)),
            ("argmax", lambda: backend.argmax(rank_0)),
            ("exp", lambda: backend.exp(rank_0)),
            ("log", lambda: backend.log(rank_0)),
            ("sqrt", lambda: backend.sqrt(rank_0)),
            ("absolute", lambda: backend.absolute(rank_0)),
            ("sign", lambda: backend.sign(rank_0)),
            ("clip", lambda: backend.clip(rank_0, 0, 1)),
            ("sum", lambda: backend.sum(rank_0)),
            ("mean", lambda: backend.mean(rank_0)),
            ("max", lambda: backend.max(rank_0)),
            ("min", lambda: backend.min(rank_0)),
            ("std", lambda: backend.std(rank_0)),
        ]

        for method_name, call in single_tensor_methods:
            with self.subTest(method=method_name):
                with self.assertRaises(
                    ValueError,
                    msg=f"{method_name} accepted a rank 0 array when it should reject it",
                ):
                    call()

        binary_tensor_methods = [
            ("add_lhs", lambda: backend.add(rank_0, matrix)),
            ("add_rhs", lambda: backend.add(matrix, rank_0)),
            ("subtract_lhs", lambda: backend.subtract(rank_0, matrix)),
            ("subtract_rhs", lambda: backend.subtract(matrix, rank_0)),
            ("multiply_lhs", lambda: backend.multiply(rank_0, matrix)),
            ("multiply_rhs", lambda: backend.multiply(matrix, rank_0)),
            ("divide_lhs", lambda: backend.divide(rank_0, matrix)),
            ("divide_rhs", lambda: backend.divide(matrix, rank_0)),
            ("matmul_lhs", lambda: backend.matmul(rank_0, matrix)),
            ("matmul_rhs", lambda: backend.matmul(matrix, rank_0)),
            ("maximum_lhs", lambda: backend.maximum(rank_0, matrix)),
            ("maximum_rhs", lambda: backend.maximum(matrix, rank_0)),
            ("minimum_lhs", lambda: backend.minimum(rank_0, matrix)),
            ("minimum_rhs", lambda: backend.minimum(matrix, rank_0)),
        ]

        for method_name, call in binary_tensor_methods:
            with self.subTest(method=method_name):
                with self.assertRaises(
                    ValueError,
                    msg=f"{method_name} accepted a rank 0 array when it should reject it",
                ):
                    call()

        sequence_tensor_methods = [
            ("stack", lambda: backend.stack([rank_0, matrix])),
            ("concatenate", lambda: backend.concatenate([rank_0, matrix])),
            ("vstack", lambda: backend.vstack([rank_0, matrix])),
            ("hstack", lambda: backend.hstack([rank_0, matrix])),
        ]

        for method_name, call in sequence_tensor_methods:
            with self.subTest(method=method_name):
                with self.assertRaises(
                    ValueError,
                    msg=f"{method_name} accepted a rank 0 array when it should reject it",
                ):
                    call()


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
        get a rank 1 array with zero elements and not an empty rank 0
        ndarray.
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
                    r"list or tuple input",
                    msg=(
                        "to_tensor did not raise ValueError when given a NumPy array-like input"
                    ),
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

    def test_to_python_rejects_rank_0_arrays(self):
        import numpy as np

        backend = self.make_backend()

        with self.assertRaisesRegex(
            ValueError,
            r"rank 0 arrays",
            msg=("to_python did not raise ValueError when given a rank 0 ndarray"),
        ):
            backend.to_python(np.array(1.0))

    def test_to_python_rejects_numpy_scalar_values(self):
        """
        This test checks that we do not get a Python (nested) list
        containing e.g. np.float64 values. This doesn't test for
        non-numpy types.
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
