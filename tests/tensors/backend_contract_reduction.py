# These are not the hardest methods to understand but explaining them
# turned out not to be easy (vs. e.g. matmul which was both hard to
# understand and explain). Come back to the method docstrings later.

"""
Tests for the five reduction methods: sum, mean, max, min, std

For the following tensor, with each operation run across *all*
of the axes:

[
    [1.0, 1.0],
    [7.0, 7.0]
]

sum adds every value in the tensor, in this case returning 16.0.

mean returns the arithmetic average of all the values in the
tensor. I.e. 16.0 / 4 = 4.0.

max returns the largest value anywhere in the tensor: 7.0.

min returns the smallest value anywhere in the tensor: 1.0.

std calculates the standard deviation of all the values in the
tensor. This means:
- calculate the mean: 4.0
- calculate the difference between each value and the mean: -3.0, -3.0, 3.0, 3.0
- square each of the difference values: 9.0, 9.0, 9.0, 9.0
- calculate the mean of those: 9.0
- calculate the square root of that mean: 3.0

The values in these test classes have been chosen very carefully such
that they are integer-valued and produce integer-valued results when
the relevant operations are carried out on them. This means the tests
can be run against backends whether they are float- or int-based.
"""

from collections.abc import Sequence

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures


@EnforceSharedNumericFixtures()
class BackendContractReductionBehaviourMixin(BackendContractBase):
    """
    A class to pin down the most important behaviour of the reduction
    methods, encompassing both handling of different axis arguments
    and the accuracy of the underlying calculations.

    Note that the 'axis' argument for these methods is doing a different
    job than the 'axes' argument in transpose. The former is a selection
    of one or more axes to reduce/collapse. The latter is a full list
    of axes in the desired output order.

    Passing different values for axis results in the operations
    reducing 'over' the given axes. Reducing over an axis means:

    - choose one axis to reduce
    - group together values whose positions are the same on every other axis
    - combine the values in each such group

    For:

    [
        [1.0, 3.0],
        [5.0, 7.0]
    ]

    the positions are:

    - 1.0 at (0, 0)
    - 3.0 at (0, 1)
    - 5.0 at (1, 0)
    - 7.0 at (1, 1)

    If we reduce over axis 0, we group together values whose positions are
    the same on every other axis, and whose positions differ only in axis 0.
    I.e. in a 2D tensor we take everything in the same column:

    - (0, 0) and (1, 0) -> 1.0 and 5.0
    - (0, 1) and (1, 1) -> 3.0 and 7.0

    Result: [6.0, 10.0]

    If we reduce over axis 1, we group together values whose positions are
    the same on every other axis, and whose positions differ only in axis 1.
    I.e. in a 2D tensor we take everything in the same row:

    - (0, 0) and (0, 1) -> 1.0 and 3.0
    - (1, 0) and (1, 1) -> 5.0 and 7.0

    Result: [4.0, 12.0]
    """

    def test_reduction_methods_reduce_over_all_axes_with_2D_tensor(
        self,
    ):
        """
        Tests the application of the reduction operations when applied to all
        elements in a 2D tensor.

        The test uses three call variants to the reduction methods, each of which
        performs the respective operation across all axes, with:
        - no axis argument
        - None as the axis argument
        - a tuple listing all (both) of the axes

        Note that when passing a tuple to the axis argument we are passing
        a sequence of indices (zero-indexed), not a shape.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 1.0], [7.0, 7.0]])

        reduction_methods = [
            ("sum", backend.sum, 16.0),
            ("mean", backend.mean, 4.0),
            ("max", backend.max, 7.0),
            ("min", backend.min, 1.0),
            ("std", backend.std, 3.0),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                calls = [
                    ("omitted_axis", lambda: method(tensor)),
                    ("axis_none", lambda: method(tensor, axis=None)),
                    ("all_axes_tuple", lambda: method(tensor, axis=(0, 1))),
                ]

                for call_style, call in calls:
                    with self.subTest(call_style=call_style):
                        result = call()
                        self.assertNotIsInstance(result, Sequence)
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_reduce_over_a_tuple_of_axes_with_2D_tensor(self):
        """
        This tests running the reduction methods across one of the (two) possible
        axes in the tensor. In each case the axis we want is passed as a singleton
        tuple.

        In the first case, axis=(0,) means we reduce over the first axis of
        the tensor and return a tensor with a single axis. For a 2D tensor,
        axis 0 corresponds to the rows, so reducing over axis 0 means combining
        the values row by row within each column.

        For the input [[1.0, 3.0], [5.0, 7.0]], the first result value is
        calculated from the first column, i.e. from 1.0 and 5.0, and the
        second result value is calculated from the second column, i.e. from
        3.0 and 7.0. The result has shape (2,).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 3.0], [5.0, 7.0]])

        valid_cases = [
            (
                (0,),
                [
                    ("sum", backend.sum, [6.0, 10.0]),
                    ("mean", backend.mean, [3.0, 5.0]),
                    ("max", backend.max, [5.0, 7.0]),
                    ("min", backend.min, [1.0, 3.0]),
                    ("std", backend.std, [2.0, 2.0]),
                ],
            ),
            (
                (1,),
                [
                    ("sum", backend.sum, [4.0, 12.0]),
                    ("mean", backend.mean, [2.0, 6.0]),
                    ("max", backend.max, [3.0, 7.0]),
                    ("min", backend.min, [1.0, 5.0]),
                    ("std", backend.std, [1.0, 1.0]),
                ],
            ),
        ]

        for axes, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result_tensor = method(tensor, axis=axes)
                        result = backend.to_python(result_tensor)
                        self.assertEqual(
                            backend.shape(result_tensor),
                            (2,),
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                "reducing over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_reduce_over_all_axes_with_3D_tensor(self):
        """
        Tests the application of the reduction operations when applied to all
        elements in a 3D tensor.

        The test uses three call variants to the reduction methods, each of which
        performs the respective operation across all axes, with:
        - no axis argument
        - None as the axis argument
        - a tuple listing all three of the axes
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ]
        )

        reduction_methods = [
            ("sum", backend.sum, 24.0),
            ("mean", backend.mean, 3.0),
            ("max", backend.max, 5.0),
            ("min", backend.min, 1.0),
            ("std", backend.std, 2.0),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                calls = [
                    ("omitted_axis", lambda: method(tensor)),
                    ("axis_none", lambda: method(tensor, axis=None)),
                    ("all_axes_tuple", lambda: method(tensor, axis=(0, 1, 2))),
                ]

                for call_style, call in calls:
                    with self.subTest(call_style=call_style):
                        result = call()
                        self.assertNotIsInstance(result, Sequence)
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_reduce_over_a_tuple_of_axes_with_3D_tensor(self):
        """
        This tests reducing a 3D tensor over a tuple of axes.

        For the tensor:

        [
            [
                [10.0, 18.0],
                [16.0, 24.0]
            ],
            [
                [16.0, 24.0],
                [22.0, 30.0]
            ]
        ]

        the positions are:

        - 10.0 at (0, 0, 0)
        - 18.0 at (0, 0, 1)
        - 16.0 at (0, 1, 0)
        - 24.0 at (0, 1, 1)
        - 16.0 at (1, 0, 0)
        - 24.0 at (1, 0, 1)
        - 22.0 at (1, 1, 0)
        - 30.0 at (1, 1, 1)

        In the first case, axis=(0,) means we reduce over the first axis only.
        So we group together values whose positions are the same on axes 1 and 2.

        This leaves axes 1 and 2 in place, so the result has shape (2, 2).

        In the second case, axis=(1, 2) means we reduce over the second and
        third axes. So each value in the result comes from all the values with
        the same first position.

        The first result value comes from:

        - 10.0 at (0, 0, 0)
        - 18.0 at (0, 0, 1)
        - 16.0 at (0, 1, 0)
        - 24.0 at (0, 1, 1)

        The second result value comes from:

        - 16.0 at (1, 0, 0)
        - 24.0 at (1, 0, 1)
        - 22.0 at (1, 1, 0)
        - 30.0 at (1, 1, 1)

        This leaves only axis 0 in place, so the result has shape (2,).

        In the third case, axis=(0, 2) means we reduce over the first and
        third axes. So each value in the result comes from all the values with
        the same middle position.

        The first result value comes from:

        - 10.0 at (0, 0, 0)
        - 18.0 at (0, 0, 1)
        - 16.0 at (1, 0, 0)
        - 24.0 at (1, 0, 1)

        The second result value comes from:

        - 16.0 at (0, 1, 0)
        - 24.0 at (0, 1, 1)
        - 22.0 at (1, 1, 0)
        - 30.0 at (1, 1, 1)

        This again leaves one axis in place, so the result has shape (2,).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[10.0, 18.0], [16.0, 24.0]],
                [[16.0, 24.0], [22.0, 30.0]],
            ]
        )

        valid_cases = [
            (
                (0,),
                (2, 2),
                [
                    ("sum", backend.sum, [[26.0, 42.0], [38.0, 54.0]]),
                    ("mean", backend.mean, [[13.0, 21.0], [19.0, 27.0]]),
                    ("max", backend.max, [[16.0, 24.0], [22.0, 30.0]]),
                    ("min", backend.min, [[10.0, 18.0], [16.0, 24.0]]),
                    ("std", backend.std, [[3.0, 3.0], [3.0, 3.0]]),
                ],
            ),
            (
                (1, 2),
                (2,),
                [
                    ("sum", backend.sum, [68.0, 92.0]),
                    ("mean", backend.mean, [17.0, 23.0]),
                    ("max", backend.max, [24.0, 30.0]),
                    ("min", backend.min, [10.0, 16.0]),
                    ("std", backend.std, [5.0, 5.0]),
                ],
            ),
            (
                (0, 2),
                (2,),
                [
                    ("sum", backend.sum, [68.0, 92.0]),
                    ("mean", backend.mean, [17.0, 23.0]),
                    ("max", backend.max, [24.0, 30.0]),
                    ("min", backend.min, [10.0, 16.0]),
                    ("std", backend.std, [5.0, 5.0]),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result_tensor = method(tensor, axis=axes)
                        result = backend.to_python(result_tensor)
                        self.assertEqual(
                            backend.shape(result_tensor),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_reduce_over_all_axes_with_4D_tensor(self):
        """
        This tests the application of the reduction methods across all the
        axes of a 4D tensor.

        As in the 2D and 3D all-axes tests, we check three call variants:
        - no axis argument
        - None as the axis argument
        - a tuple listing all four axes

        In each case the result should be a scalar, because all the axes are
        reduced.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ],
                [
                    [[5.0, 5.0], [5.0, 5.0]],
                    [[5.0, 5.0], [5.0, 5.0]],
                ],
            ]
        )

        reduction_methods = [
            ("sum", backend.sum, 48.0),
            ("mean", backend.mean, 3.0),
            ("max", backend.max, 5.0),
            ("min", backend.min, 1.0),
            ("std", backend.std, 2.0),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                calls = [
                    ("omitted_axis", lambda: method(tensor)),
                    ("axis_none", lambda: method(tensor, axis=None)),
                    ("all_axes_tuple", lambda: method(tensor, axis=(0, 1, 2, 3))),
                ]

                for call_style, call in calls:
                    with self.subTest(call_style=call_style):
                        result = call()
                        self.assertNotIsInstance(result, Sequence)
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_reduce_over_a_tuple_of_axes_with_4D_tensor(self):
        # Some of the language here is taken from the NumPy docs on
        # ufunc.reduce. It's better than anything I could come up with
        # but still not great. It needs some work.
        """
        This tests reducing a 4D tensor over a tuple of axes.

        All cases in this test run across the same input tensor:

        [
            [
                [[8.0, 16.0], [16.0, 24.0]],
                [[14.0, 22.0], [22.0, 30.0]],
            ],
            [
                [[10.0, 18.0], [18.0, 26.0]],
                [[16.0, 24.0], [24.0, 32.0]],
            ],
        ]

        For the case axis=(1, 3), the second and fourth axes are reduced, and
        the first and third axes remain. So each value in the result comes from
        all the input values which share the same positions on axes 0 and 2.

        For example, the result value at position (0, 0) comes from:

        - 8.0 at (0, 0, 0, 0)
        - 16.0 at (0, 0, 0, 1)
        - 14.0 at (0, 1, 0, 0)
        - 22.0 at (0, 1, 0, 1)

        The result value at position (0, 1) comes from:

        - 16.0 at (0, 0, 1, 0)
        - 24.0 at (0, 0, 1, 1)
        - 22.0 at (0, 1, 1, 0)
        - 30.0 at (0, 1, 1, 1)

        The same pattern gives the result values at positions (1, 0) and
        (1, 1). Because axes 0 and 2 remain, the result has shape (2, 2).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [[8.0, 16.0], [16.0, 24.0]],
                    [[14.0, 22.0], [22.0, 30.0]],
                ],
                [
                    [[10.0, 18.0], [18.0, 26.0]],
                    [[16.0, 24.0], [24.0, 32.0]],
                ],
            ]
        )

        valid_cases = [
            (
                (0,),
                (2, 2, 2),
                [
                    (
                        "sum",
                        backend.sum,
                        [
                            [[18.0, 34.0], [34.0, 50.0]],
                            [[30.0, 46.0], [46.0, 62.0]],
                        ],
                    ),
                    (
                        "mean",
                        backend.mean,
                        [
                            [[9.0, 17.0], [17.0, 25.0]],
                            [[15.0, 23.0], [23.0, 31.0]],
                        ],
                    ),
                    (
                        "max",
                        backend.max,
                        [
                            [[10.0, 18.0], [18.0, 26.0]],
                            [[16.0, 24.0], [24.0, 32.0]],
                        ],
                    ),
                    (
                        "min",
                        backend.min,
                        [
                            [[8.0, 16.0], [16.0, 24.0]],
                            [[14.0, 22.0], [22.0, 30.0]],
                        ],
                    ),
                    (
                        "std",
                        backend.std,
                        [
                            [[1.0, 1.0], [1.0, 1.0]],
                            [[1.0, 1.0], [1.0, 1.0]],
                        ],
                    ),
                ],
            ),
            (
                (1, 2),
                (2, 2),
                [
                    ("sum", backend.sum, [[60.0, 92.0], [68.0, 100.0]]),
                    ("mean", backend.mean, [[15.0, 23.0], [17.0, 25.0]]),
                    ("max", backend.max, [[22.0, 30.0], [24.0, 32.0]]),
                    ("min", backend.min, [[8.0, 16.0], [10.0, 18.0]]),
                    ("std", backend.std, [[5.0, 5.0], [5.0, 5.0]]),
                ],
            ),
            (
                (1, 3),
                (2, 2),
                [
                    ("sum", backend.sum, [[60.0, 92.0], [68.0, 100.0]]),
                    ("mean", backend.mean, [[15.0, 23.0], [17.0, 25.0]]),
                    ("max", backend.max, [[22.0, 30.0], [24.0, 32.0]]),
                    ("min", backend.min, [[8.0, 16.0], [10.0, 18.0]]),
                    ("std", backend.std, [[5.0, 5.0], [5.0, 5.0]]),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result_tensor = method(tensor, axis=axes)
                        result = backend.to_python(result_tensor)
                        self.assertEqual(
                            backend.shape(result_tensor),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_reduce_over_a_single_integer_axis(self):
        """
        This tests that the reduction methods accept a single integer for the
        axis argument, rather than requiring a tuple.

        We use a 4D tensor and more than one axis value so that the test does
        not accidentally encourage an implementation which only handles one
        particular axis or only works for low-rank tensors.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [[8.0, 16.0], [16.0, 24.0]],
                    [[14.0, 22.0], [22.0, 30.0]],
                ],
                [
                    [[10.0, 18.0], [18.0, 26.0]],
                    [[16.0, 24.0], [24.0, 32.0]],
                ],
            ]
        )

        valid_cases = [
            (
                1,
                [
                    (
                        "sum",
                        backend.sum,
                        [
                            [[22.0, 38.0], [38.0, 54.0]],
                            [[26.0, 42.0], [42.0, 58.0]],
                        ],
                    ),
                    (
                        "mean",
                        backend.mean,
                        [
                            [[11.0, 19.0], [19.0, 27.0]],
                            [[13.0, 21.0], [21.0, 29.0]],
                        ],
                    ),
                    (
                        "max",
                        backend.max,
                        [
                            [[14.0, 22.0], [22.0, 30.0]],
                            [[16.0, 24.0], [24.0, 32.0]],
                        ],
                    ),
                    (
                        "min",
                        backend.min,
                        [
                            [[8.0, 16.0], [16.0, 24.0]],
                            [[10.0, 18.0], [18.0, 26.0]],
                        ],
                    ),
                    (
                        "std",
                        backend.std,
                        [
                            [[3.0, 3.0], [3.0, 3.0]],
                            [[3.0, 3.0], [3.0, 3.0]],
                        ],
                    ),
                ],
            ),
            (
                3,
                [
                    (
                        "sum",
                        backend.sum,
                        [
                            [[24.0, 40.0], [36.0, 52.0]],
                            [[28.0, 44.0], [40.0, 56.0]],
                        ],
                    ),
                    (
                        "mean",
                        backend.mean,
                        [
                            [[12.0, 20.0], [18.0, 26.0]],
                            [[14.0, 22.0], [20.0, 28.0]],
                        ],
                    ),
                    (
                        "max",
                        backend.max,
                        [
                            [[16.0, 24.0], [22.0, 30.0]],
                            [[18.0, 26.0], [24.0, 32.0]],
                        ],
                    ),
                    (
                        "min",
                        backend.min,
                        [
                            [[8.0, 16.0], [14.0, 22.0]],
                            [[10.0, 18.0], [16.0, 24.0]],
                        ],
                    ),
                    (
                        "std",
                        backend.std,
                        [
                            [[4.0, 4.0], [4.0, 4.0]],
                            [[4.0, 4.0], [4.0, 4.0]],
                        ],
                    ),
                ],
            ),
        ]

        for axis, reduction_methods in valid_cases:
            with self.subTest(axis=axis):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result_tensor = method(tensor, axis=axis)
                        result = backend.to_python(result_tensor)
                        self.assertEqual(
                            backend.shape(result_tensor),
                            (2, 2, 2),
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing over axis {axis}"
                            ),
                        )
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_treat_axis_1_and_axis_singleton_tuple_1_as_equivalent(
        self,
    ):
        """
        This tests the contract rule that, for the reduction methods,
        axis=1 and axis=(1,) are equivalent.

        We use a 3D tensor so that the test checks this equivalence in a
        such a way as to avoid backend implementations which treat 1D/2D
        tensorts as a special case.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[10.0, 18.0], [16.0, 24.0]],
                [[16.0, 24.0], [22.0, 30.0]],
            ]
        )

        reduction_methods = [
            (
                "sum",
                backend.sum,
                [[26.0, 42.0], [38.0, 54.0]],
            ),
            (
                "mean",
                backend.mean,
                [[13.0, 21.0], [19.0, 27.0]],
            ),
            (
                "max",
                backend.max,
                [[16.0, 24.0], [22.0, 30.0]],
            ),
            (
                "min",
                backend.min,
                [[10.0, 18.0], [16.0, 24.0]],
            ),
            (
                "std",
                backend.std,
                [[3.0, 3.0], [3.0, 3.0]],
            ),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                integer_axis_result_tensor = method(tensor, axis=1)
                integer_axis_result = backend.to_python(integer_axis_result_tensor)
                singleton_tuple_result_tensor = method(tensor, axis=(1,))
                singleton_tuple_result = backend.to_python(
                    singleton_tuple_result_tensor
                )
                self.assertEqual(
                    backend.shape(integer_axis_result_tensor),
                    backend.shape(singleton_tuple_result_tensor),
                    msg=(
                        f"{method_name} returned different shapes for axis=1 "
                        "and axis=(1,)"
                    ),
                )
                assert_nested_close(integer_axis_result, expected, rel_tol=0, abs_tol=0)
                assert_nested_close(
                    singleton_tuple_result, expected, rel_tol=0, abs_tol=0
                )

    def test_reduction_methods_accept_negative_axes(self):
        """
        This tests that negative axis values are accepted and interpreted by
        counting back from the end of the tensor shape.

        Each case compares a negative axis or axes tuple with the equivalent
        positive value and checks that both produce the same result.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[10.0, 18.0], [16.0, 24.0]],
                [[16.0, 24.0], [22.0, 30.0]],
            ]
        )

        valid_cases = [
            (
                -1,
                2,
                [
                    ("sum", backend.sum, [[28.0, 40.0], [40.0, 52.0]]),
                    ("mean", backend.mean, [[14.0, 20.0], [20.0, 26.0]]),
                    ("max", backend.max, [[18.0, 24.0], [24.0, 30.0]]),
                    ("min", backend.min, [[10.0, 16.0], [16.0, 22.0]]),
                    ("std", backend.std, [[4.0, 4.0], [4.0, 4.0]]),
                ],
            ),
            (
                (-3, -1),
                (0, 2),
                [
                    ("sum", backend.sum, [68.0, 92.0]),
                    ("mean", backend.mean, [17.0, 23.0]),
                    ("max", backend.max, [24.0, 30.0]),
                    ("min", backend.min, [10.0, 16.0]),
                    ("std", backend.std, [5.0, 5.0]),
                ],
            ),
        ]

        for negative_axes, positive_axes, reduction_methods in valid_cases:
            with self.subTest(negative_axes=negative_axes, positive_axes=positive_axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        negative_result_tensor = method(tensor, axis=negative_axes)
                        negative_result = backend.to_python(negative_result_tensor)
                        positive_result_tensor = method(tensor, axis=positive_axes)
                        positive_result = backend.to_python(positive_result_tensor)
                        self.assertEqual(
                            backend.shape(negative_result_tensor),
                            backend.shape(positive_result_tensor),
                            msg=(
                                f"{method_name} returned different shapes for "
                                f"axis={negative_axes} and axis={positive_axes}"
                            ),
                        )
                        assert_nested_close(
                            negative_result, expected, rel_tol=0, abs_tol=0
                        )
                        assert_nested_close(
                            positive_result, expected, rel_tol=0, abs_tol=0
                        )

    def test_reduction_methods_treat_axis_tuple_order_as_irrelevant(self):
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [[8.0, 16.0], [16.0, 24.0]],
                    [[14.0, 22.0], [22.0, 30.0]],
                ],
                [
                    [[10.0, 18.0], [18.0, 26.0]],
                    [[16.0, 24.0], [24.0, 32.0]],
                ],
            ]
        )

        reduction_methods = [
            ("sum", backend.sum, [[60.0, 92.0], [68.0, 100.0]]),
            ("mean", backend.mean, [[15.0, 23.0], [17.0, 25.0]]),
            ("max", backend.max, [[22.0, 30.0], [24.0, 32.0]]),
            ("min", backend.min, [[8.0, 16.0], [10.0, 18.0]]),
            ("std", backend.std, [[5.0, 5.0], [5.0, 5.0]]),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                axis_1_3_result_tensor = method(tensor, axis=(1, 3))
                axis_1_3_result = backend.to_python(axis_1_3_result_tensor)
                axis_3_1_result_tensor = method(tensor, axis=(3, 1))
                axis_3_1_result = backend.to_python(axis_3_1_result_tensor)
                self.assertEqual(
                    backend.shape(axis_1_3_result_tensor),
                    backend.shape(axis_3_1_result_tensor),
                    msg=(
                        f"{method_name} returned different shapes for "
                        "axis=(1, 3) and axis=(3, 1)"
                    ),
                )
                assert_nested_close(axis_1_3_result, expected, rel_tol=0, abs_tol=0)
                assert_nested_close(axis_3_1_result, expected, rel_tol=0, abs_tol=0)


class BackendContractReductionKeepdimsMixin(BackendContractBase):

    def test_reduction_methods_keep_reduced_single_axis_when_keepdims_is_true_with_2D_tensor(
        self,
    ):
        """
        When keepdims=True is passed, the reduced axis is not removed from
        the result. Instead, it stays in the result with size 1.

        The tensor here has shape (2, 2). Reducing over axis 1 without
        keepdims would give a result with shape (2,). With keepdims=True,
        axis 1 is retained with size 1, so the result has shape (2, 1).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 3.0], [5.0, 7.0]])

        valid_cases = [
            ("sum", backend.sum, [[4.0], [12.0]]),
            ("mean", backend.mean, [[2.0], [6.0]]),
            ("max", backend.max, [[3.0], [7.0]]),
            ("min", backend.min, [[1.0], [5.0]]),
            ("std", backend.std, [[1.0], [1.0]]),
        ]

        for method_name, method, expected in valid_cases:
            with self.subTest(method=method_name):
                result_tensor = method(tensor, axis=(1,), keepdims=True)
                result = backend.to_python(result_tensor)
                self.assertEqual(
                    backend.shape(result_tensor),
                    (2, 1),
                    msg=(
                        f"{method_name} did not return the expected shape when "
                        "reducing with keepdims=True over a single axis"
                    ),
                )
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_keep_reduced_axes_when_keepdims_is_true_with_3D_tensor(
        self,
    ):
        """
        This tests how the reduction methods behave when keepdims=True is
        used with a 3D tensor.

        When keepdims=True, the axes we reduce over are not removed from the
        result. Instead, they stay in the result with size 1.

        For the tensor:

        [
            [
                [10.0, 18.0],
                [16.0, 24.0]
            ],
            [
                [16.0, 24.0],
                [22.0, 30.0]
            ]
        ]

        In the first case, axis=(1, 2) means we reduce over the second and
        third axes and keep the first axis. So the result has shape (2, 1, 1)
        rather than (2,).

        In the second case, axis=(0, 2) means we reduce over the first and
        third axes and keep the middle axis. So the result has shape (1, 2, 1)
        rather than (2,).

        These tests therefore check both:
        - that the values are correct
        - that the reduced axes remain in the result with size 1
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[10.0, 18.0], [16.0, 24.0]],
                [[16.0, 24.0], [22.0, 30.0]],
            ]
        )

        valid_cases = [
            (
                (1, 2),
                (2, 1, 1),
                [
                    ("sum", backend.sum, [[[68.0]], [[92.0]]]),
                    ("mean", backend.mean, [[[17.0]], [[23.0]]]),
                    ("max", backend.max, [[[24.0]], [[30.0]]]),
                    ("min", backend.min, [[[10.0]], [[16.0]]]),
                    ("std", backend.std, [[[5.0]], [[5.0]]]),
                ],
            ),
            (
                (0, 2),
                (1, 2, 1),
                [
                    ("sum", backend.sum, [[[68.0], [92.0]]]),
                    ("mean", backend.mean, [[[17.0], [23.0]]]),
                    ("max", backend.max, [[[24.0], [30.0]]]),
                    ("min", backend.min, [[[10.0], [16.0]]]),
                    ("std", backend.std, [[[5.0], [5.0]]]),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result_tensor = method(tensor, axis=axes, keepdims=True)
                        result = backend.to_python(result_tensor)
                        self.assertEqual(
                            backend.shape(result_tensor),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing with keepdims=True over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_keep_reduced_axes_when_keepdims_is_true_with_4D_tensor(
        self,
    ):
        """
        This tests how keepdims=True behaves with a 4D tensor.

        When keepdims=True, the axes we reduce over are not removed from the
        result. Instead, they remain in the result with size 1.

        In the case axis=(1, 3), we reduce over the second and fourth axes and
        keep the first and third axes. Without keepdims, the result would have
        shape (2, 2). With keepdims=True, the result has shape (2, 1, 2, 1).

        This means the test checks both:
        - that the values are correct
        - that the reduced axes remain in the result with size 1
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [[8.0, 16.0], [16.0, 24.0]],
                    [[14.0, 22.0], [22.0, 30.0]],
                ],
                [
                    [[10.0, 18.0], [18.0, 26.0]],
                    [[16.0, 24.0], [24.0, 32.0]],
                ],
            ]
        )

        valid_cases = [
            (
                (1, 2),
                (2, 1, 1, 2),
                [
                    ("sum", backend.sum, [[[[60.0, 92.0]]], [[[68.0, 100.0]]]]),
                    ("mean", backend.mean, [[[[15.0, 23.0]]], [[[17.0, 25.0]]]]),
                    ("max", backend.max, [[[[22.0, 30.0]]], [[[24.0, 32.0]]]]),
                    ("min", backend.min, [[[[8.0, 16.0]]], [[[10.0, 18.0]]]]),
                    ("std", backend.std, [[[[5.0, 5.0]]], [[[5.0, 5.0]]]]),
                ],
            ),
            (
                (1, 3),
                (2, 1, 2, 1),
                [
                    ("sum", backend.sum, [[[[60.0], [92.0]]], [[[68.0], [100.0]]]]),
                    ("mean", backend.mean, [[[[15.0], [23.0]]], [[[17.0], [25.0]]]]),
                    ("max", backend.max, [[[[22.0], [30.0]]], [[[24.0], [32.0]]]]),
                    ("min", backend.min, [[[[8.0], [16.0]]], [[[10.0], [18.0]]]]),
                    ("std", backend.std, [[[[5.0], [5.0]]], [[[5.0], [5.0]]]]),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result_tensor = method(tensor, axis=axes, keepdims=True)
                        result = backend.to_python(result_tensor)
                        self.assertEqual(
                            backend.shape(result_tensor),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing with keepdims=True over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reduction_methods_remove_reduced_axes_when_keepdims_is_false(self):
        """
        This tests the behaviour of the reduction methods when keepdims=False.

        In these cases the reduced axes should be removed from the result,
        rather than being kept with size 1.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[10.0, 18.0], [16.0, 24.0]],
                [[16.0, 24.0], [22.0, 30.0]],
            ]
        )

        valid_cases = [
            (
                (1, 2),
                (2,),
                [
                    ("sum", backend.sum, [68.0, 92.0]),
                    ("mean", backend.mean, [17.0, 23.0]),
                    ("max", backend.max, [24.0, 30.0]),
                    ("min", backend.min, [10.0, 16.0]),
                    ("std", backend.std, [5.0, 5.0]),
                ],
            ),
            (
                (0, 2),
                (2,),
                [
                    ("sum", backend.sum, [68.0, 92.0]),
                    ("mean", backend.mean, [17.0, 23.0]),
                    ("max", backend.max, [24.0, 30.0]),
                    ("min", backend.min, [10.0, 16.0]),
                    ("std", backend.std, [5.0, 5.0]),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result_tensor = method(tensor, axis=axes, keepdims=False)
                        result = backend.to_python(result_tensor)
                        self.assertEqual(
                            backend.shape(result_tensor),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing with keepdims=False over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)


class BackendContractReductionInvalidAxisMixin(BackendContractBase):
    def test_reduction_methods_reject_duplicate_axes(self):
        """
        Confirms that the reduction methods reject axis tuples which refer
        to the same axis more than once.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        reduction_methods = [
            ("sum", backend.sum),
            ("mean", backend.mean),
            ("max", backend.max),
            ("min", backend.min),
            ("std", backend.std),
        ]
        invalid_axes = [
            (0, 0),
            (1, 1),
            (0, 2, 0),
        ]

        for method_name, method in reduction_methods:
            with self.subTest(method=method_name):
                for axis in invalid_axes:
                    with self.subTest(axis=axis):
                        with self.assertRaises(ValueError):
                            method(tensor, axis=axis)

    def test_reduction_methods_reject_axes_outside_the_valid_range(self):
        """
        Confirms that the reduction methods reject axis values which fall
        outside the valid range for the input tensor.

        Negative values for axes are treated like negative indicies when
        used with Python lists/tuples. If a negative value is too large
        then it will produce a value of less than 0 (the index of the first
        axis) when counted back from the end.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        reduction_methods = [
            ("sum", backend.sum),
            ("mean", backend.mean),
            ("max", backend.max),
            ("min", backend.min),
            ("std", backend.std),
        ]
        invalid_axes = [
            3,
            -4,
            (0, 3),
            (-4, 1),
        ]

        for method_name, method in reduction_methods:
            with self.subTest(method=method_name):
                for axis in invalid_axes:
                    with self.subTest(axis=axis):
                        with self.assertRaises(ValueError):
                            method(tensor, axis=axis)


class BackendContractReductionEmptyInputMixin(BackendContractBase):
    def test_sum_returns_zero_when_called_on_an_empty_tensor(self):
        backend = self.make_backend()
        empty_inputs = [
            [],
            [[], []],
            [[[], []], [[], []]],
        ]

        for data in empty_inputs:
            with self.subTest(data=data):
                tensor = backend.to_tensor(data)
                result = backend.sum(tensor)

                self.assertNotIsInstance(result, Sequence)
                assert_nested_close(result, 0.0, rel_tol=0, abs_tol=0)

    def test_mean_max_min_and_std_raise_when_called_on_an_empty_tensor(self):
        backend = self.make_backend()
        empty_inputs = [
            [],
            [[], []],
            [[[], []], [[], []]],
        ]

        reduction_methods = [
            ("mean", backend.mean),
            ("max", backend.max),
            ("min", backend.min),
            ("std", backend.std),
        ]

        for data in empty_inputs:
            with self.subTest(data=data):
                tensor = backend.to_tensor(data)
                for method_name, method in reduction_methods:
                    with self.subTest(method=method_name):
                        with self.assertRaises(ValueError):
                            method(tensor)
