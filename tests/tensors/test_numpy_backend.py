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
    BackendContractToTensorValueMixin,
)
from tests.tensors.backend_contract_randn import BackendContractRandnMixin
from tests.tensors.backend_contract_matmul import BackendContractMatmulMixin
from tests.tensors.backend_contract_reshape import BackendContractReshapeMixin
from tests.tensors.backend_contract_reduction import (
    BackendContractScalarReturnTypeMixin,
)
from tests.tensors.backend_contract_argmax import BackendContractArgMaxMixin
from tests.tensors.backend_contract_transpose import BackendContractTransposeMixin

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackendProtocolConformance(unittest.TestCase):
    # It is essential to set the return type here if we want mypy to type check
    # the instantiation of NumpyBackend
    def test_numpy_backend_implements_tensor_backend_protocol(self) -> None:
        from src.tensors import NumpyBackend

        # This is a safety check so that if the codebase ever temporarily or
        # permanently does not pass NumpyBackend to a layer or other consumer
        # the type checker will still catch deviations from the protocol/contract
        backend: TensorBackend = NumpyBackend()
        # Test at runtime.
        self.assertIsInstance(backend, TensorBackend)


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackend(
    BackendContractConstructionMixin,
    BackendContractCreationMixin,
    BackendContractFloatCreationMixin,
    BackendContractToTensorTypeInputMixin,
    BackendContractToTensorShapeInputMixin,
    BackendContractToTensorValueMixin,
    BackendContractRandnMixin,
    BackendContractMatmulMixin,
    BackendContractReshapeMixin,
    BackendContractScalarReturnTypeMixin,
    BackendContractArgMaxMixin,
    BackendContractTransposeMixin,
    unittest.TestCase,
):
    def make_backend(self, seed: int | None = None) -> TensorBackend:
        from src.tensors import NumpyBackend

        return NumpyBackend(seed=seed)

    def test_scalar_returning_methods_do_not_return_rank_0_arrays(self):
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

    def test_tensor_input_methods_reject_rank_0_arrays(self):
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
