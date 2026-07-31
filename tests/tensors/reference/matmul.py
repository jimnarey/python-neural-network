"""
Tests the parts of the matmul reference design which go beyond the backend
contract.

These tests focus on float-valued outputs, in both of the forms matmul can
return them: float-valued tensors, and a plain Python float when the result
is a scalar. They also include a small arithmetic check using non-integer
float fixtures where the reference design is intentionally more specific
than the backend contract.
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

    def test_matmul_returns_float_scalar_when_result_is_a_scalar(self):
        """
        Test that matmul returns a plain Python float, not an int, when
        multiplying two 1D tensors of whole numbers together produces a
        scalar result.
        """
        backend = self.make_backend()
        a = backend.to_tensor([1, 2, 3])
        b = backend.to_tensor([4, 5, 6])

        result = backend.matmul(a, b)
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
