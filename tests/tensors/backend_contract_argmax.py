from tests.tensors.backend_contract_shared import BackendContractBase


class BackendContractArgMaxMixin(BackendContractBase):

    def test_argmax_returns_int_when_returning_a_scalar(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1, 4], [3, 2]])
        result = backend.argmax(tensor)
        self.assertIsInstance(result, int)
