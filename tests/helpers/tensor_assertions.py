"""Helper functions used by the backend contract tests

Because they contain non-trivial logic themselves they have their own
tests. All of the backend contract tests depend on these working as
expected.

The rel_to and abs_to values are calibrated according to the behaviour
of the individual tensor backends. I.e. the tolerances are large enough
to accomodate the differences in rounding behaviour but no larger.

"""

import math


def to_python(tensor):
    """
    Convert a backend-native tensor into plain Python values so the same
    assertions work for list-based and NumPy-based implementations.
    """
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return tensor


def assert_nested_close(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
    """
    Assert that two nested numeric structures are equal within a
    floating-point tolerance.
    """
    actual_value = to_python(actual)
    _assert_nested_close(actual_value, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def _assert_nested_close(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
    """
    Recurse through lists/tuples of expected and actual values until we
    get to a pair of scalar values, then check that they match within
    the provided tolerances.

    rel_tol: The relative tolerance. It is the maximum allowed difference
    between value a and b.

    abs_tol: The minimum absolute tolerance. It is used to compare values
    near 0. The value must be at least 0

    """
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            raise AssertionError(
                f"Expected a nested sequence, got {type(actual).__name__}."
            )
        if len(actual) != len(expected):
            raise AssertionError(
                f"Sequence lengths differ: {len(actual)} != {len(expected)}."
            )
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_close(
                actual_item,
                expected_item,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        return

    if not math.isclose(
        actual,
        expected,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        raise AssertionError(f"{actual!r} != {expected!r}")
