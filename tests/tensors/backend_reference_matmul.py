from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close


class BackendReferenceMatmulArithmeticMixin(BackendContractBase):

    def test_matmul_multiplies_two_square_2D_tensors_with_non_integer_values(self):
        backend = self.make_backend()

        a = backend.to_tensor([[1.5, 2.25], [3.75, 4.5]])
        b = backend.to_tensor([[2.0, 0.5], [1.25, 3.5]])

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [5.8125, 8.625],
            [13.125, 17.625],
        ]
        self.assertEqual(backend.shape(tensor), (2, 2))
        assert_nested_close(result, expected)
