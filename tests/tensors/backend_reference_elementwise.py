import math

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_helpers import assert_nested_close, all_values_are_floats


class BackendReferenceElementwiseFloatValueMixin(BackendContractBase):
    def test_elementwise_methods_return_float_valued_tensors_for_same_shape_inputs(
        self,
    ):
        """
        Test that in a simple case of elementwise operations, all values returned
        are floats. It checks this is the case when:
        - both operand values are floats
        - both are ints
        - there is one of each
        """
        backend = self.make_backend()
        a = backend.to_tensor([[2.0, 6], [12, 20]])
        b = backend.to_tensor([[1.0, 3], [4, 5.0]])

        elementwise_methods = [
            ("add", backend.add),
            ("subtract", backend.subtract),
            ("multiply", backend.multiply),
            ("divide", backend.divide),
            ("maximum", backend.maximum),
            ("minimum", backend.minimum),
        ]

        for method_name, method in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertTrue(all_values_are_floats(result))

    def test_elementwise_methods_return_float_valued_tensors_when_broadcasting(self):
        """
        Test that all values in a tensor returned from an operation involving
        broadcasting are floats. It checks this is the case when:
        - both operand values are floats
        - both are ints
        - there is one of each

        This test uses tensors which engage both types of elementwise broadcasting,
        left-padding and length-1-axis.

        The resulting tensor from the add operation is:
            [
                [
                    [13.0, 26.0, 39.0],
                    [14.0, 27.0, 42.0]
                ],
                [
                    [19.0, 32.0, 45.0],
                    [20.0, 33.0, 48.0]
                ]
            ]

        Like all tensors produced by the operations under test in this method it has
        shape (2, 2, 3). We don't test the shape here as we're not testing the
        broadcasting behaviour itself, which is covered in the contract tests.
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[12.0, 24, 36]],
                [[18, 30, 42.0]],
            ]
        )
        b = backend.to_tensor([[1.0, 2, 3], [2, 3, 6]])

        elementwise_methods = [
            ("add", backend.add),
            ("subtract", backend.subtract),
            ("multiply", backend.multiply),
            ("divide", backend.divide),
            ("maximum", backend.maximum),
            ("minimum", backend.minimum),
        ]

        for method_name, method in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertTrue(all_values_are_floats(result))

    def test_elementwise_methods_return_float_valued_tensors_when_given_int_scalar_rhs(
        self,
    ):
        """
        Test that when a scalar int is used as the right-hand operand in
        elementwise operations, all values returned are floats. It checks this
        is the case when:
        - the tensor value is a float
        - the tensor value is an int
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[4.5, 8], [12, 20.0]],
                [[6, 10.5], [14.0, 18]],
            ]
        )
        b = 2

        elementwise_methods = [
            ("add", backend.add),
            ("subtract", backend.subtract),
            ("multiply", backend.multiply),
            ("divide", backend.divide),
            ("maximum", backend.maximum),
            ("minimum", backend.minimum),
        ]

        for method_name, method in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertTrue(all_values_are_floats(result))

    def test_elementwise_methods_return_float_valued_tensors_when_given_float_scalar_rhs(
        self,
    ):
        """
        Test that when a scalar float is used as the right-hand operand in
        elementwise operations, all values returned are floats. It checks this
        is the case when:
        - the tensor value is a float
        - the tensor value is an int
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[4.5, 8], [12, 20.0]],
                [[6, 10.5], [14.0, 18]],
            ]
        )
        b = 2.0

        elementwise_methods = [
            ("add", backend.add),
            ("subtract", backend.subtract),
            ("multiply", backend.multiply),
            ("divide", backend.divide),
            ("maximum", backend.maximum),
            ("minimum", backend.minimum),
        ]

        for method_name, method in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertTrue(all_values_are_floats(result))


class BackendReferenceElementwiseArithmeticMixin(BackendContractBase):
    def test_add_and_subtract_return_expected_results_with_non_integer_values(self):
        backend = self.make_backend()
        a = backend.to_tensor([[2.5, 6.25], [12.75, 20.5]])
        b = backend.to_tensor([[1.25, 3.5], [4.5, 5.75]])

        elementwise_methods = [
            ("add", backend.add, [[3.75, 9.75], [17.25, 26.25]]),
            ("subtract", backend.subtract, [[1.25, 2.75], [8.25, 14.75]]),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                assert_nested_close(result, expected)

    def test_multiply_returns_expected_results_with_non_integer_values(self):
        backend = self.make_backend()
        a = backend.to_tensor([[2.5, 6.25], [12.75, 20.5]])
        b = backend.to_tensor([[1.25, 3.5], [4.5, 5.75]])

        result_tensor = backend.multiply(a, b)
        result = backend.to_python(result_tensor)

        expected = [
            [3.125, 21.875],
            [57.375, 117.875],
        ]
        assert_nested_close(result, expected)

    def test_divide_returns_expected_results_with_non_integer_values(self):
        backend = self.make_backend()
        a = backend.to_tensor([[2.5, 6.25], [12.75, 20.5]])
        b = backend.to_tensor([[1.25, 2.5], [4.25, 4.1]])

        result_tensor = backend.divide(a, b)
        result = backend.to_python(result_tensor)

        expected = [
            [2.0, 2.5],
            [3.0, 5.0],
        ]
        assert_nested_close(result, expected)

    def test_maximum_and_minimum_return_expected_results_with_non_integer_values(
        self,
    ):
        backend = self.make_backend()
        a = backend.to_tensor([[2.5, 6.25], [12.75, 20.5]])
        b = backend.to_tensor([[1.25, 7.5], [13.0, 5.75]])

        elementwise_methods = [
            ("maximum", backend.maximum, [[2.5, 7.5], [13.0, 20.5]]),
            ("minimum", backend.minimum, [[1.25, 6.25], [12.75, 5.75]]),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                assert_nested_close(result, expected)


class BackendReferenceElementwiseSpecialValueMixin(BackendContractBase):
    """
    Tests the part of the reference design which pins down how reference
    backends surface special floating-point values at the Python boundary.

    These tests use explicit calls to math.isnan(...) and math.isinf(...)
    rather than comparing the whole returned tensor with an expected nested
    list. This is because nan does not compare equal to itself, so an
    equality-style comparison would fail even when the returned value is
    exactly the special value we expect.
    """

    def test_divide_surfaces_special_values_when_dividing_by_zero_scalar(self):
        """
        This tests division by a scalar zero.

        The expected result is:
            [
                [inf, -inf],
                [nan, inf]
            ]

        The result should therefore contain positive infinity where a
        positive number is divided by zero, negative infinity where a
        negative number is divided by zero, and nan where zero is divided
        by zero.
        """
        backend = self.make_backend()
        a = backend.to_tensor([[4.0, -6.0], [0.0, 8.0]])

        result_tensor = backend.divide(a, 0.0)
        result = backend.to_python(result_tensor)
        self.assertTrue(all_values_are_floats(result))
        self.assertTrue(math.isinf(result[0][0]))
        self.assertGreater(result[0][0], 0.0)
        self.assertTrue(math.isinf(result[0][1]))
        self.assertLess(result[0][1], 0.0)
        self.assertTrue(math.isnan(result[1][0]))
        self.assertTrue(math.isinf(result[1][1]))
        self.assertGreater(result[1][1], 0.0)

    def test_divide_surfaces_special_values_when_dividing_by_zero_tensor_values(
        self,
    ):
        """
        This tests division by a tensor containing both ordinary values and
        zeros.

        The expected result is:
            [
                [2.0, -inf, nan],
                [inf, 2.0, -2.0]
            ]

        This tests that ordinary finite results and special floating-point
        values can both appear in the same returned tensor.
        """
        backend = self.make_backend()
        a = backend.to_tensor([[4.0, -6.0, 0.0], [9.0, 8.0, -10.0]])
        b = backend.to_tensor([[2.0, 0.0, 0.0], [0.0, 4.0, 5.0]])

        result_tensor = backend.divide(a, b)
        result = backend.to_python(result_tensor)

        self.assertTrue(all_values_are_floats(result))
        self.assertEqual(result[0][0], 2.0)
        self.assertTrue(math.isinf(result[0][1]))
        self.assertLess(result[0][1], 0.0)
        self.assertTrue(math.isnan(result[0][2]))
        self.assertTrue(math.isinf(result[1][0]))
        self.assertGreater(result[1][0], 0.0)
        self.assertEqual(result[1][1], 2.0)
        self.assertEqual(result[1][2], -2.0)
