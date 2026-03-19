import importlib.util
import unittest

from tests.tensors.backend_contract import BackendConstructionContractMixin
from tests.tensors.backend_contract_matmul import BackendContractMatmulMixin

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackend(
    BackendConstructionContractMixin, BackendContractMatmulMixin, unittest.TestCase
):
    def make_backend(self, seed: int | None = None):
        from src.tensors import NumpyBackend

        # Setting the RNG seed to zero (or another fixed value) means
        # that we get reproducible results when generating random values with NumPy
        return NumpyBackend(seed=0)
