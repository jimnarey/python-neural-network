"""Test module for the NumPy backend

This contains a runner class for the shared backend contract tests.

It also has tests for to_tensor and to_python, which it is critical
be tested thoroughly as the shared tests rely on them.

We import numpy within each test method individually to enable the
skipUnless checks which apply to each class. It also helps with test
isolation just in case code under test does something like set the
global NumPy seed (this has been carefully avoided in the way the
backend is constructed).
"""

import importlib.util
import unittest

from src.tensors.tensor_backend import TensorBackend
from tests.tensors.backend_contract_shared import BackendContractConstructionMixin
from tests.tensors.backend_contract_creation import (
    BackendContractCreationMixin,
    BackendContractFloatCreationMixin,
)
from tests.tensors.backend_contract_to_tensor import (
    BackendContractToTensorTypeInputMixin,
    BackendContractToTensorShapeInputMixin,
)

from tests.tensors.backend_contract_matmul import (
    BackendContractMatmulReferenceArithmeticMixin,
    BackendContractMatmulSemanticsMixin,
    BackendContractMatmulBroadcastingMixin,
)

from tests.tensors.backend_contract_randn import BackendContractRandnMixin

from tests.tensors.backend_contract_reshape import BackendContractReshapeMixin
from tests.tensors.backend_contract_reduction import (
    BackendContractScalarReturnTypeMixin,
)
from tests.tensors.backend_contract_argmax import BackendContractArgMaxMixin
from tests.tensors.backend_contract_transpose import BackendContractTransposeMixin

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
    BackendContractFloatCreationMixin,
    BackendContractToTensorTypeInputMixin,
    BackendContractToTensorShapeInputMixin,
    BackendContractRandnMixin,
    BackendContractMatmulReferenceArithmeticMixin,
    BackendContractMatmulSemanticsMixin,
    BackendContractMatmulBroadcastingMixin,
    BackendContractReshapeMixin,
    BackendContractScalarReturnTypeMixin,
    BackendContractArgMaxMixin,
    BackendContractTransposeMixin,
):
    pass


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
    def test_to_tensor_preserves_1D_value_positions_in_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([1.0, 2.0, 3.0, 4.0])

        self.assertIsInstance(result, np.ndarray)
        # Use ndarray's shape attribute, not our backend's shape method
        self.assertEqual(result.shape, (4,))
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 2.0)
        self.assertEqual(result[2], 3.0)
        self.assertEqual(result[3], 4.0)

    def test_to_tensor_preserves_2D_value_positions_in_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result[0, 0], 1.0)
        self.assertEqual(result[0, 1], 2.0)
        self.assertEqual(result[0, 2], 3.0)
        self.assertEqual(result[1, 0], 4.0)
        self.assertEqual(result[1, 1], 5.0)
        self.assertEqual(result[1, 2], 6.0)

    def test_to_tensor_preserves_3D_value_positions_in_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertEqual(result[0, 0, 0], 1.0)
        self.assertEqual(result[0, 0, 1], 2.0)
        self.assertEqual(result[0, 1, 0], 3.0)
        self.assertEqual(result[0, 1, 1], 4.0)
        self.assertEqual(result[1, 0, 0], 5.0)
        self.assertEqual(result[1, 0, 1], 6.0)
        self.assertEqual(result[1, 1, 0], 7.0)
        self.assertEqual(result[1, 1, 1], 8.0)

    def test_to_tensor_preserves_4D_value_positions_in_ndarray(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                ],
                [
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
            ]
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 1, 2, 2))
        self.assertEqual(result[0, 0, 0, 0], 1.0)
        self.assertEqual(result[0, 0, 0, 1], 2.0)
        self.assertEqual(result[0, 0, 1, 0], 3.0)
        self.assertEqual(result[0, 0, 1, 1], 4.0)
        self.assertEqual(result[1, 0, 0, 0], 5.0)
        self.assertEqual(result[1, 0, 0, 1], 6.0)
        self.assertEqual(result[1, 0, 1, 0], 7.0)
        self.assertEqual(result[1, 0, 1, 1], 8.0)

    def test_to_tensor_returns_float_dtype_ndarray_when_given_integer_input(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([1, 2, 3])

        self.assertIsInstance(result, np.ndarray)
        # Checks that the ndarray is float typed but does not check
        # which is not quite the same as checking every value
        # (but close enough)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 2.0)
        self.assertEqual(result[2], 3.0)

    def test_to_tensor_returns_float_dtype_ndarray_when_given_mixed_numeric_input(self):
        import numpy as np

        backend = self.make_backend()
        result = backend.to_tensor([1, 2.5, 3])

        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 2.5)
        self.assertEqual(result[2], 3.0)

    def test_to_tensor_rejects_ndarray_input(self):
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
                        "to_tensor did not raise the expected ValueError from the "
                        "list-or-tuple input guard when given a NumPy array-like input"
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
        ]

        for data in invalid_inputs:
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                    ValueError,
                    r"numeric values",
                    msg=(
                        "to_tensor did not raise the expected ValueError from "
                        "the numeric-values guard when given NumPy scalar "
                        "values within input"
                    ),
                ):
                    backend.to_tensor(data)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendToPython(NumpyBackendTestCase):
    def test_to_python_converts_1D_ndarray_to_plain_python_list(self):
        pass

    def test_to_python_converts_2D_ndarray_to_plain_python_nested_list(self):
        pass

    def test_to_python_converts_3D_ndarray_to_plain_python_nested_list(self):
        pass

    def test_to_python_returns_plain_python_float_values(self):
        pass

    def test_to_python_rejects_rank_0_arrays(self):
        pass

    def test_to_python_rejects_numpy_scalar_values(self):
        pass
