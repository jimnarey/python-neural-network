import math
import unittest

from fnn.tensors.shared.scalar_ops import (
    divide_scalar,
    log_scalar,
    sign_scalar,
    sqrt_scalar,
)


class TestDivideScalar(unittest.TestCase):

    def test_returns_ordinary_division_result_when_divisor_is_not_zero(self):
        cases = (
            (8.0, 2.0, 4.0),
            (-7.0, 2.0, -3.5),
            (7.0, -8.0, -0.875),
            (-5.0, -2.0, 2.5),
        )
        for left, right, expected in cases:
            with self.subTest():
                result = divide_scalar(left, right)
                self.assertAlmostEqual(result, expected)

    def test_returns_nan_when_zero_is_divided_by_zero(self):
        result = divide_scalar(0.0, 0.0)
        self.assertTrue(math.isnan(result))

    def test_returns_positive_inf_when_positive_value_is_divided_by_zero(self):
        result = divide_scalar(1.0, 0.0)
        self.assertEqual(result, math.inf)

    def test_returns_negative_inf_when_negative_value_is_divided_by_zero(self):
        result = divide_scalar(-1.0, 0.0)
        self.assertEqual(result, -math.inf)


class TestLogScalar(unittest.TestCase):

    def test_returns_log_when_value_is_positive(self):
        cases = (
            (1.0, 0.0),
            (math.e, 1.0),
        )
        for value, expected in cases:
            with self.subTest():
                result = log_scalar(value)
                self.assertEqual(result, expected)

    def test_returns_negative_inf_when_value_is_zero(self):
        result = log_scalar(0.0)
        self.assertEqual(result, -math.inf)

    def test_returns_nan_when_value_is_negative(self):
        result = log_scalar(-1.0)
        self.assertTrue(math.isnan(result))


class TestSqrtScalar(unittest.TestCase):

    def test_returns_square_root_when_value_is_not_negative(self):
        cases = (
            (0.0, 0.0),
            (4.0, 2.0),
            (9.0, 3.0),
        )
        for value, expected in cases:
            with self.subTest():
                result = sqrt_scalar(value)
                self.assertEqual(result, expected)

    def test_returns_nan_when_value_is_negative(self):
        result = sqrt_scalar(-1.0)
        self.assertTrue(math.isnan(result))


class TestSignScalar(unittest.TestCase):

    def test_returns_negative_one_when_value_is_negative(self):
        result = sign_scalar(-3.0)
        self.assertEqual(result, -1.0)

    def test_returns_zero_when_value_is_zero(self):
        result = sign_scalar(0.0)
        self.assertEqual(result, 0.0)

    def test_returns_one_when_value_is_positive(self):
        result = sign_scalar(5.0)
        self.assertEqual(result, 1.0)
