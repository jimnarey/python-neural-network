"""
Reference tests for the unary tensor operations.

These tests cover unary behaviour which is required only of the
float-based reference backends, not of all backends in general.

This includes:
- arithmetic results which depend on non-integer float values
- the use of float-valued tensors for unary-operation results
- the surfacing of special float values such as nan and inf at the
  Python boundary where the reference design requires that behaviour

The universal contract tests cover only the unary behaviour which all
backends must share.
"""

import math

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_helpers import all_values_are_floats, assert_nested_close


class BackendReferenceExpArithmeticMixin(BackendContractBase):
    def test_exp_returns_expected_values_for_1D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([0.0, 1.0, 2.0])
        result_tensor = backend.exp(tensor)
        result = backend.to_python(result_tensor)
        expected = [math.exp(0.0), math.exp(1.0), math.exp(2.0)]
        self.assertEqual(backend.shape(result_tensor), (3,))
        assert_nested_close(result, expected)

    def test_exp_returns_expected_values_for_2D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[0.0, 1.0], [2.0, 3.0]])
        result_tensor = backend.exp(tensor)
        result = backend.to_python(result_tensor)
        expected = [
            [math.exp(0.0), math.exp(1.0)],
            [math.exp(2.0), math.exp(3.0)],
        ]
        self.assertEqual(backend.shape(result_tensor), (2, 2))
        assert_nested_close(result, expected)

    def test_exp_returns_expected_values_for_3D_tensors(self):
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                backend.to_tensor(
                    [
                        [[0.0, 1.0, 2.0]],
                        [[3.0, 4.0, 5.0]],
                    ]
                ),
                [
                    [[math.exp(0.0), math.exp(1.0), math.exp(2.0)]],
                    [[math.exp(3.0), math.exp(4.0), math.exp(5.0)]],
                ],
                (2, 1, 3),
            ),
            (
                "larger_middle_dimension",
                backend.to_tensor(
                    [
                        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                        [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
                    ]
                ),
                [
                    [
                        [math.exp(0.0), math.exp(1.0)],
                        [math.exp(2.0), math.exp(3.0)],
                        [math.exp(4.0), math.exp(5.0)],
                    ],
                    [
                        [math.exp(6.0), math.exp(7.0)],
                        [math.exp(8.0), math.exp(9.0)],
                        [math.exp(10.0), math.exp(11.0)],
                    ],
                ],
                (2, 3, 2),
            ),
        ]

        for case_name, tensor, expected, expected_shape in test_cases:
            result_tensor = backend.exp(tensor)
            result = backend.to_python(result_tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), expected_shape)
                assert_nested_close(result, expected)


class BackendReferenceLogArithmeticMixin(BackendContractBase):
    def test_log_returns_expected_values_for_1D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0, 4.0])
        result_tensor = backend.log(tensor)
        result = backend.to_python(result_tensor)
        expected = [math.log(1.0), math.log(2.0), math.log(4.0)]
        self.assertEqual(backend.shape(result_tensor), (3,))
        assert_nested_close(result, expected)

    def test_log_returns_expected_values_for_2D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0], [4.0, 8.0]])
        result_tensor = backend.log(tensor)
        result = backend.to_python(result_tensor)
        expected = [
            [math.log(1.0), math.log(2.0)],
            [math.log(4.0), math.log(8.0)],
        ]
        self.assertEqual(backend.shape(result_tensor), (2, 2))
        assert_nested_close(result, expected)

    def test_log_returns_expected_values_for_3D_tensors(self):
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                backend.to_tensor(
                    [
                        [[1.0, 2.0, 4.0]],
                        [[8.0, 16.0, 32.0]],
                    ]
                ),
                [
                    [[math.log(1.0), math.log(2.0), math.log(4.0)]],
                    [[math.log(8.0), math.log(16.0), math.log(32.0)]],
                ],
                (2, 1, 3),
            ),
            (
                "larger_middle_dimension",
                backend.to_tensor(
                    [
                        [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]],
                        [[64.0, 128.0], [256.0, 512.0], [1024.0, 2048.0]],
                    ]
                ),
                [
                    [
                        [math.log(1.0), math.log(2.0)],
                        [math.log(4.0), math.log(8.0)],
                        [math.log(16.0), math.log(32.0)],
                    ],
                    [
                        [math.log(64.0), math.log(128.0)],
                        [math.log(256.0), math.log(512.0)],
                        [math.log(1024.0), math.log(2048.0)],
                    ],
                ],
                (2, 3, 2),
            ),
        ]

        for case_name, tensor, expected, expected_shape in test_cases:
            result_tensor = backend.log(tensor)
            result = backend.to_python(result_tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), expected_shape)
                assert_nested_close(result, expected)


class BackendReferenceUnaryValueTypeMixin(BackendContractBase):
    def test_exp_log_and_sqrt_return_float_valued_tensors(self):
        backend = self.make_backend()
        unary_methods = [
            ("exp", backend.exp, backend.to_tensor([0.0, 1.0, 2.0])),
            ("log", backend.log, backend.to_tensor([1.0, 2.0, 4.0])),
            ("sqrt", backend.sqrt, backend.to_tensor([1.0, 4.0, 9.0])),
        ]

        for method_name, method, tensor in unary_methods:
            with self.subTest(method=method_name):
                result = backend.to_python(method(tensor))
                self.assertTrue(all_values_are_floats(result))

    def test_absolute_sign_and_clip_return_float_valued_tensors(self):
        backend = self.make_backend()
        unary_methods = [
            ("absolute", backend.absolute, backend.to_tensor([-3.0, 0.0, 2.0])),
            ("sign", backend.sign, backend.to_tensor([-3.0, 0.0, 2.0])),
            (
                "clip",
                lambda x: backend.clip(x, 0.0, 3.0),
                backend.to_tensor([-2.0, 1.0, 5.0]),
            ),
        ]

        for method_name, method, tensor in unary_methods:
            with self.subTest(method=method_name):
                result = backend.to_python(method(tensor))
                self.assertTrue(all_values_are_floats(result))


class BackendReferenceSqrtArithmeticMixin(BackendContractBase):
    """
    Test that the actual arithmetic for sqrt works. We test that
    return values are the right type (float) elsewhere in this
    module.
    """

    def test_sqrt_returns_expected_results_with_non_integer_values(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[0.25, 2.25], [6.25, 12.25]])
        result_tensor = backend.sqrt(tensor)
        result = backend.to_python(result_tensor)
        expected = [[0.5, 1.5], [2.5, 3.5]]

        self.assertEqual(backend.shape(result_tensor), (2, 2))
        assert_nested_close(result, expected)


class BackendReferenceLogSpecialValueMixin(BackendContractBase):
    """
    This tests how log behaves when one or more input values are not valid.

    For positive values, log should return the ordinary logarithm. But zero
    and negative values result in a conventionally forbidden operation and
    require the backend to handle it somehow. For the float-based reference
    backends, the design decision is that the operation completes and the
    result is surfaced at the Python boundary using the special float
    values -inf and nan.

    The purpose of this mixin is not to test value type (nan, inf, -inf are
    type float) but that this specific boundary behaviour is adhered to.
    """

    def test_log_surfaces_special_values_at_python_boundary(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 0.0, -1.0])
        result_tensor = backend.log(tensor)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (3,))
        self.assertEqual(result[0], 0.0)
        self.assertTrue(math.isinf(result[1]))
        self.assertLess(result[1], 0.0)
        self.assertTrue(math.isnan(result[2]))


class BackendReferenceSqrtSpecialValueMixin(BackendContractBase):
    """
    This tests how sqrt behaves when one or more input values are not valid.

    In the case of sqrt positive values or 0 are valid (sqrt(0) == 0).

    The purpose of this mixin is not to test value type (nan is type float)
    but that this specific boundary behaviour is adhered to.
    """

    def test_sqrt_surfaces_special_values_at_python_boundary(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([-1.0, 0.0, 4.0])
        result_tensor = backend.sqrt(tensor)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (3,))
        self.assertTrue(math.isnan(result[0]))
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[2], 2.0)
