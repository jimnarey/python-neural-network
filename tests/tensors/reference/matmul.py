"""
Tests the parts of the matmul reference design which go beyond the backend
contract.

These tests check that matmul returns float-valued tensors. This includes a
rank-zero tensor when multiplying two 1D tensors produces one value. They
also include a small arithmetic check using non-integer float fixtures where
the reference design is intentionally more specific than the backend contract.
"""

from tests.tensors.contract.shared import BackendContractBase
from tests.helpers.tensor_helpers import assert_nested_close, all_values_are_floats


class BackendReferenceMatmulFloatValueMixin(BackendContractBase):
    def test_matmul_returns_float_valued_tensor_when_result_is_a_tensor(self):
        """
        Test that matmul returns float-valued results even when every input
        value and every result value happens to be a whole number. This
        guards against an accumulator implementation that stays an int for
        as long as the values being summed are whole numbers.
        """
        backend = self.make_backend()
        a = backend.to_tensor([[1, 2], [3, 4]])
        b = backend.to_tensor([[5, 6], [7, 8]])

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_matmul_returns_rank_0_float_tensor_for_two_1D_tensors(self):
        """
        Test that matmul returns a rank-zero float tensor when multiplying
        two 1D tensors produces one value. After conversion by to_python,
        that value must be a Python float rather than an int.
        """
        backend = self.make_backend()
        a = backend.to_tensor([1, 2, 3])
        b = backend.to_tensor([4, 5, 6])

        result_tensor = backend.matmul(a, b)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), ())
        self.assertIsInstance(result, float)


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
