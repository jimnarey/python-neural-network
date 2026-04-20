"""Helper functions used by the backend contract tests

Because they contain non-trivial logic themselves they have their own
tests. All of the backend contract tests depend on these working as
expected.

The rel_to and abs_to values are calibrated according to the behaviour
of the individual tensor backends. I.e. the tolerances are large enough
to accomodate the differences in rounding behaviour but no larger.

"""

import math
from typing import Any, Final

# TODO - tighten type checking here once we can add a meaningful type
# or union for tensor

DEFAULT_REL_TOL: Final = 1e-7
DEFAULT_ABS_TOL: Final = 1e-8


def assert_nested_close(
    actual: Any,
    expected: Any,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> None:
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
            assert_nested_close(
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
