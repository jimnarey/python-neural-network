import importlib.util
import unittest

from tests.tensors.backend_contract import BackendContractMixin

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestNumpyBackend(BackendContractMixin, unittest.TestCase):
    def make_backend(self):
        from src.tensors import NumpyBackend

        # Setting the RNG seed to zero (or another fixed value) means
        # that we get reproducible results when generating random values with NumPy
        return NumpyBackend(seed=0)
