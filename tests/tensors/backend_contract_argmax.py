from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures


@EnforceSharedNumericFixtures()
class BackendContractArgMaxMixin(BackendContractBase):

    def test_argmax_returns_int_when_returning_a_scalar(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 4.0], [3.0, 2.0]])
        result = backend.argmax(tensor)
        self.assertIsInstance(result, int)
