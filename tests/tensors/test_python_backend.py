import unittest

from tests.tensors.backend_contract import (
    BackendContractConstructionMixin,
    BackendContractCreationMixin,
)
from tests.tensors.backend_contract_randn import BackendContractRandnMixin
from tests.tensors.backend_contract_matmul import BackendContractMatmulMixin


@unittest.skip(
    "PythonBackend contract tests will be enabled with the first implementation."
)
class TestPythonBackend(
    BackendContractConstructionMixin,
    BackendContractCreationMixin,
    BackendContractRandnMixin,
    BackendContractMatmulMixin,
    unittest.TestCase,
):
    def make_backend(self, seed: int | None = None):
        from src.tensors.python_backend import PythonBackend

        return PythonBackend(seed=seed)
