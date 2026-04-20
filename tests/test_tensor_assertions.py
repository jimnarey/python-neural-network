"""Test classes for the backend contract helpers

As well as ensuring the helpers can convert whatever tensor objects the
individual backends use into nested lists these tests document the
behaviour of the float comparisons. This amounts, to a large extent, to
testing Python's math.to_close function. This isn't because we don't
trust the Python devs. It's to make absolutely sure we understand how
it works with different combinations of values and thresholds.
"""

from unittest import TestCase

from tests.helpers.tensor_assertions import assert_nested_close


class MockTensor:
    def tolist(self):
        return [[1.0, 2.0], [3.0, 4.0]]


class TestAssertNestedClose(TestCase):
    """
    Tests that the recursive unpacking of nested lists/tuples works
    as expected.

    Also documents how rel_tol and abs_tol passed to math.isclose interact:
    each defines an allowed difference and the larger of the two
    is used as the threshold.

    This is because When the values being compared are very small (close to zero),
    the relative tolerance (which scales with the value) becomes tiny, so
    the comparison instead uses the absolute tolerance as a minimum allowed
    difference.

    """

    def test_accepts_equal_values_in_2D_nested_sequences(self):
        assert_nested_close(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        )

    def test_accepts_equal_values_in_3D_nested_sequences(self):
        """
        Does not raise because the helper keeps recursing until it reaches
        matching scalar values.
        """
        assert_nested_close(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        )

    def test_accepts_different_scalar_values_within_tolerance(self):
        """
        Does not raise because the scalar difference (1.00000001 vs 1.0)
        is within the requested relative tolerance of 1e-7.
        """
        assert_nested_close(1.00000001, 1.0, rel_tol=1e-7, abs_tol=1e-8)

    def test_accepts_different_values_within_relative_tolerance_in_2D_nested_sequence(
        self,
    ):
        """
        Does not raise because the difference (1.0000001 vs 1.0)
        is within the requested relative tolerance of 1e-6 (0.000001).
        """
        assert_nested_close(
            [[1.0000001, 2.0]],
            [[1.0, 2.0]],
            rel_tol=1e-6,
        )

    def test_rejects_different_scalar_value_outside_tolerance(self):
        with self.assertRaises(AssertionError):
            assert_nested_close(1.1, 1.0, rel_tol=1e-7, abs_tol=1e-8)

    def test_rejects_2D_nested_sequences_with_one_different_scalar_value_outside_tolerance(
        self,
    ):
        """
        Raises because the difference (1.0000001 vs 1.0)
        exceeds the stricter relative tolerance of 1e-8 (0.00000001).
        """
        with self.assertRaises(AssertionError):
            assert_nested_close(
                [[1.0000001, 2.0]],
                [[1.0, 2.0]],
                rel_tol=1e-8,
            )

    def test_rejects_3D_nested_sequences_with_one_different_scalar_value_outside_tolerance(
        self,
    ):
        """
        Raises because one deeply nested scalar value does not match.
        """
        with self.assertRaises(AssertionError):
            assert_nested_close(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.1, 8.0]],
                ],
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                rel_tol=1e-7,
                abs_tol=1e-8,
            )

    def test_accepts_values_within_absolute_tolerance_for_near_zero_values(self):
        """
        Does not raise because for very small values the absolute tolerance (1e-8)
        is larger than the relative threshold:

        rel_tol * max(|a|,|b|) = 1e-7 * 1e-9 = 1e-16

        So abs_tol (1e-8) becomes the effective tolerance.
        """
        assert_nested_close(
            [[0.000000001, 2.0]], [[0.0, 2.0]], rel_tol=1e-7, abs_tol=1e-8
        )

    def test_rejects_when_absolute_tolerance_too_small_for_near_zero_values(self):
        """Raises because `abs_tol` is set smaller than the relative threshold:

        rel_tol * max(|a|,|b|) = 1e-7 * 1e-9 = 1e-16

        With abs_tol=1e-18 the effective tolerance remains the relative value
        (1e-16), and the difference (1e-9) is far larger, so the comparison fails.
        """
        with self.assertRaises(AssertionError):
            assert_nested_close(
                [[0.000000001, 2.0]], [[0.0, 2.0]], rel_tol=1e-7, abs_tol=1e-18
            )

    def test_rejects_different_actual_and_expected_sequence_lengths(self):
        with self.assertRaises(AssertionError):
            assert_nested_close([[1.0]], [[1.0], [2.0]], rel_tol=1e-7, abs_tol=1e-8)

    def test_rejects_different_actual_and_expected_inner_sequence_lengths(self):
        """
        Raises because one nested sequence contains more values than the
        corresponding sequence in the expected structure.
        """
        with self.assertRaises(AssertionError):
            assert_nested_close(
                [[[1.0, 2.0, 3.0]]],
                [[[1.0, 2.0]]],
                rel_tol=1e-7,
                abs_tol=1e-8,
            )

    def test_rejects_non_sequence_actual_when_expected_is_nested(self):
        with self.assertRaises(AssertionError):
            assert_nested_close(1.0, [1.0], rel_tol=1e-7, abs_tol=1e-8)

    def test_rejects_non_sequence_actual_when_expected_is_more_deeply_nested(self):
        """
        Raises because a nested sequence is expected, but the actual structure
        contains a scalar at that point.
        """
        with self.assertRaises(AssertionError):
            assert_nested_close(
                [[1.0, 2.0], 3.0],
                [[1.0, 2.0], [3.0, 4.0]],
                rel_tol=1e-7,
                abs_tol=1e-8,
            )

    def test_accepts_mixed_list_and_tuple_nesting(self):
        """
        Does not raise because the helper accepts both lists and tuples as
        nested sequences.
        """
        assert_nested_close(
            ([1.0, 2.0], (3.0, 4.0)),
            [[1.0, 2.0], [3.0, 4.0]],
            rel_tol=1e-7,
            abs_tol=1e-8,
        )
