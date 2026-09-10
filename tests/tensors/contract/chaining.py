"""Test chaining of operations returning single values

These tests check that backends represent a result containing one value as
a rank-zero tensor. This allows the result to be passed directly to another
tensor operation without first converting it to a Python value.
"""

from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures
from tests.helpers.tensor_helpers import assert_nested_close
from tests.tensors.contract.shared import BackendContractBase


@EnforceSharedNumericFixtures()
class BackendContractRank0ChainingMixin(BackendContractBase):
    def test_reduction_rank_0_result_can_be_passed_to_elementwise_operation(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0, 3.0])
        result_tensor = backend.add(backend.sum(tensor), 2.0)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), ())
        assert_nested_close(result, 8.0, rel_tol=0, abs_tol=0)

    def test_vector_matmul_rank_0_result_can_be_passed_to_elementwise_operation(self):
        backend = self.make_backend()
        left = backend.to_tensor([1.0, 2.0])
        right = backend.to_tensor([3.0, 4.0])
        result_tensor = backend.multiply(backend.matmul(left, right), 2.0)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), ())
        assert_nested_close(result, 22.0, rel_tol=0, abs_tol=0)

    def test_argmax_rank_0_result_can_be_reshaped(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 3.0, 2.0])
        result_tensor = backend.reshape(backend.argmax(tensor), (1,))
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (1,))
        assert_nested_close(result, [1], rel_tol=0, abs_tol=0)
