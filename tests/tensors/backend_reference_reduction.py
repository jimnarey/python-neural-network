"""
Tests the parts of the reduction reference design which go beyond the
backend contract.

These tests focus on float-valued outputs: plain Python floats when a
reduction returns a scalar, and float-valued tensors when it returns a
tensor. They also include a small number of arithmetic checks using
non-integer float fixtures where the reference design is intentionally
more specific than the backend contract.
"""

from tests.tensors.backend_contract_shared import BackendContractBase


class BackendReferenceReductionFloatValueMixin(BackendContractBase):
    def test_reduction_methods_return_float_scalars(self):
        backend = self.make_backend()
        tensor = backend.ones((2, 2))

        scalar_methods = [
            ("sum", lambda: backend.sum(tensor)),
            ("mean", lambda: backend.mean(tensor)),
            ("max", lambda: backend.max(tensor)),
            ("min", lambda: backend.min(tensor)),
            ("std", lambda: backend.std(tensor)),
        ]

        for method_name, call in scalar_methods:
            with self.subTest(method=method_name):
                result = call()
                self.assertIsInstance(
                    result,
                    float,
                    msg=f"{method_name} returned {result!r} instead of a float",
                )

    def test_reduction_methods_return_float_valued_tensors(self):
        """
        This tests that the reduction methods return float-valued tensors when
        the result is not scalar.

        The reductions are chosen to produce a 1D tensor so that we can iterate
        through the returned values directly and check their types.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 3.0], [5.0, 7.0]])

        tensor_methods = [
            ("sum", lambda: backend.sum(tensor, axis=(0,))),
            ("mean", lambda: backend.mean(tensor, axis=(0,))),
            ("max", lambda: backend.max(tensor, axis=(0,))),
            ("min", lambda: backend.min(tensor, axis=(0,))),
            ("std", lambda: backend.std(tensor, axis=(0,))),
        ]

        for method_name, call in tensor_methods:
            with self.subTest(method=method_name):
                result_tensor = call()
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2,))
                self.assertIsInstance(result, list)

                for value in result:
                    with self.subTest(value=value):
                        self.assertIs(type(value), float)

    def test_reduction_methods_return_float_tensors_when_keepdims_is_true(self):
        """
        This tests that the reduction methods still return float-valued tensors
        when keepdims=True is used.

        The test flattens the result after converting it to Python so that it
        checks only the value types. The shape behaviour of keepdims is tested
        in the backend contract tests rather than here.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 3.0], [5.0, 7.0]])

        tensor_methods = [
            ("sum", lambda: backend.sum(tensor, axis=(1,), keepdims=True)),
            ("mean", lambda: backend.mean(tensor, axis=(1,), keepdims=True)),
            ("max", lambda: backend.max(tensor, axis=(1,), keepdims=True)),
            ("min", lambda: backend.min(tensor, axis=(1,), keepdims=True)),
            ("std", lambda: backend.std(tensor, axis=(1,), keepdims=True)),
        ]

        for method_name, call in tensor_methods:
            with self.subTest(method=method_name):
                result = backend.to_python(call())
                self.assertIsInstance(result, list)
                flattened_result = [value for row in result for value in row]

                for value in flattened_result:
                    with self.subTest(value=value):
                        self.assertIs(type(value), float)


class BackendReferenceReductionArithmeticMixin(BackendContractBase):
    """
    Only mean and std are included here, because they are the reduction
    methods which can naturally produce non-integer-valued results from
    integer-valued inputs. By contrast, sum, max and min preserve integer
    values for integer-valued inputs.
    """

    def test_mean_returns_float_scalar_when_result_is_fractional(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0])
        result = backend.mean(tensor)

        self.assertIsInstance(result, float)
        self.assertEqual(result, 1.5)

    def test_std_returns_float_scalar_when_result_is_fractional(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0])
        result = backend.std(tensor)

        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.5)
