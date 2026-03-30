# These are not the hardest methods to understand but explaining them
# turned out not to be easy (vs. e.g. matmul which was both hard to
# understand and explain). Come back to the method docstrings later.

"""
Tests for the five reduction methods: sum, mean, max, min, std

For the following tensor, with each operation run across *all*
of the axes:

[
    [1.0, 2.0],
    [3.0, 4.0]
]

sum adds every value in the tensor, in this case returning 10.0.

mean returns the arithmetic average of all the values in the
tensor. I.e. 10.0 / 4 = 2.5.

max returns the largest value anywhere in the tensor: 4.0.

min returns the smallest value anywhere in the tensor: 1.0.

std calculates the standard deviation of all the values in the
tensor. This means:
- calculate the mean: 2.5
- calculate the difference between each value and the mean: -1.5, -0.5, 0.5, 1.5
- square each of the difference values: 2.25, 0.25, 0.25, 2.25
- calculate the mean of those: 1.25
- calculate the square root of that mean: 1.118033988749895
"""

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close


class BackendContractReductionBehaviourMixin(BackendContractBase):
    """
    A class to pin down the most important behaviour of the reduction
    methods, encompassing both handling of different axis arguments
    and the accuracy of the underlying calculations.

    Note that the 'axis' argument for these methods is doing a different
    job than the 'axes' argument in transpose. The former is a selection
    of one or more axes to reduce/collapse. The second is a full list
    of axes in the desired output order.

    Passing different values for axis results in the operations
    reducing 'over' the given axes. Reducing over an axis means:

    - choose one axis to reduce
    - group together values whose positions are the same on every other axis
    - combine the values in each such group

    For:

    [
        [1.0, 2.0],
        [3.0, 5.0]
    ]

    the positions are:

    - 1.0 at (0, 0)
    - 2.0 at (0, 1)
    - 3.0 at (1, 0)
    - 5.0 at (1, 1)

    If we reduce over axis 0, we group together values whose positions are
    the same on every other axis, and whose positions differ only in axis 0.
    I.e. in a 2D array we take everything in the same column:

    - (0, 0) and (1, 0) -> 1.0 and 3.0
    - (0, 1) and (1, 1) -> 2.0 and 5.0

    If we reduce over axis 1, we group together values whose positions are
    the same on every other axis, and whose positions differ only in axis 1.
    I.e. in a 2D array we take everything in the same row:

    - (0, 0) and (0, 1) -> 1.0 and 2.0
    - (1, 0) and (1, 1) -> 3.0 and 5.0

    What can be slightly confusing is that if you reduce by axis 0 (rows)
    you are left with columns, and vice-versa.
    """

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

    def test_reduction_methods_reduce_over_all_axes_with_2D_array(
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

        We use assert_nested_close for testing the output of std for consistency
        with the other test modules but could have used math.to_close() directly,
        or self.assertIsClose(), to achieve the same effect.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])

        reduction_methods = [
            ("sum", backend.sum, 10.0),
            ("mean", backend.mean, 2.5),
            ("max", backend.max, 4.0),
            ("min", backend.min, 1.0),
            (
                "std",
                backend.std,
                1.118033988749895,
            ),  # Consider pruning the dec places here, given testing tolerances
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                calls = [
                    lambda: method(tensor),
                    lambda: method(tensor, axis=None),
                    lambda: method(tensor, axis=(0, 1)),
                ]

                for call in calls:
                    result = call()
                    if method_name == "std":
                        assert_nested_close(result, expected)
                    else:
                        self.assertEqual(result, expected)

    def test_reduction_methods_reduce_over_a_tuple_of_axes_with_2D_array(self):
        """
        This tests running the reduction methods across one of the (two) possible
        axes in the tensor. In each case the axis we want is passed as a singleton
        tuple.

        In the first case, axis=(0,) means we reduce over the first axis of
        the tensor and keep the second axis. For a 2D tensor, axis 0
        corresponds to the rows, so reducing over axis 0 means combining the
        values row by row within each column.

        For the input [[1.0, 2.0], [3.0, 5.0]], the first result value is
        calculated from the first column, i.e. from 1.0 and 3.0, and the
        second result value is calculated from the second column, i.e. from
        2.0 and 5.0. Because the columns remain, the result has shape
        (2,).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0], [3.0, 5.0]])

        valid_cases = [
            (
                (0,),
                [
                    ("sum", backend.sum, [4.0, 7.0]),
                    ("mean", backend.mean, [2.0, 3.5]),
                    ("max", backend.max, [3.0, 5.0]),
                    ("min", backend.min, [1.0, 2.0]),
                    ("std", backend.std, [1.0, 1.5]),
                ],
            ),
            (
                (1,),
                [
                    ("sum", backend.sum, [3.0, 8.0]),
                    ("mean", backend.mean, [1.5, 4.0]),
                    ("max", backend.max, [2.0, 5.0]),
                    ("min", backend.min, [1.0, 3.0]),
                    ("std", backend.std, [0.5, 1.0]),
                ],
            ),
        ]

        for axes, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result = method(tensor, axis=axes)
                        self.assertEqual(
                            backend.shape(result),
                            (2,),
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                "reducing over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected)

    def test_reduction_methods_keep_reduced_single_axis_when_keepdims_is_true_with_2D_array(
        self,
    ):
        """
        When True is passed as the keepdims argument, the reduction methods
        are run across each axis separately, with the result for each axis
        becoming that axis's only value.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0], [3.0, 5.0]])

        valid_cases = [
            ("sum", backend.sum, [[3.0], [8.0]]),
            ("mean", backend.mean, [[1.5], [4.0]]),
            ("max", backend.max, [[2.0], [5.0]]),
            ("min", backend.min, [[1.0], [3.0]]),
            ("std", backend.std, [[0.5], [1.0]]),
        ]

        for method_name, method, expected in valid_cases:
            with self.subTest(method=method_name):
                result = method(tensor, axis=(1,), keepdims=True)
                self.assertEqual(
                    backend.shape(result),
                    (2, 1),
                    msg=(
                        f"{method_name} did not return the expected shape when "
                        "reducing with keepdims=True over a single axis"
                    ),
                )
                assert_nested_close(result, expected)

    def test_reduction_methods_reduce_over_all_axes_with_3D_array(self):
        """
        Tests the application of the reduction operations when applied to all
        elements in a 3D tensor.

        The tests three call variants to the reduction methods, each of which
        performs the respective operation across all axes, with:
        - no axis argument
        - None as the axis argument
        - a tuple listing all three of the axes
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 9.0]],
            ]
        )

        reduction_methods = [
            ("sum", backend.sum, 37.0),
            ("mean", backend.mean, 4.625),
            ("max", backend.max, 9.0),
            ("min", backend.min, 1.0),
            ("std", backend.std, 2.496873044429772),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                calls = [
                    lambda: method(tensor),
                    lambda: method(tensor, axis=None),
                    lambda: method(tensor, axis=(0, 1, 2)),
                ]

                for call in calls:
                    result = call()
                    if method_name == "std":
                        assert_nested_close(result, expected)
                    else:
                        self.assertEqual(result, expected)

    def test_reduction_methods_reduce_over_a_tuple_of_axes_with_3D_array(self):
        """
        This tests reducing a 3D tensor over a tuple of axes.

        For the tensor:

        [
            [
                [1.0, 2.0],
                [3.0, 4.0]
            ],
            [
                [5.0, 6.0],
                [7.0, 9.0]
            ]
        ]

        the positions are:

        - 1.0 at (0, 0, 0)
        - 2.0 at (0, 0, 1)
        - 3.0 at (0, 1, 0)
        - 4.0 at (0, 1, 1)
        - 5.0 at (1, 0, 0)
        - 6.0 at (1, 0, 1)
        - 7.0 at (1, 1, 0)
        - 9.0 at (1, 1, 1)

        In the first case, axis=(0,) means we reduce over the first axis only.
        So we group together values whose positions are the same on axes 1 and 2.

        This leaves axes 1 and 2 in place, so the result has shape (2, 2).

        In the second case, axis=(1, 2) means we reduce over the second and
        third axes. So each value in the result comes from all the values with
        the same first position.

        The first result value comes from:

        - 1.0 at (0, 0, 0)
        - 2.0 at (0, 0, 1)
        - 3.0 at (0, 1, 0)
        - 4.0 at (0, 1, 1)

        The second result value comes from:

        - 5.0 at (1, 0, 0)
        - 6.0 at (1, 0, 1)
        - 7.0 at (1, 1, 0)
        - 9.0 at (1, 1, 1)

        This leaves only axis 0 in place, so the result has shape (2,).

        In the third case, axis=(0, 2) means we reduce over the first and
        third axes. So each value in the result comes from all the values with
        the same middle position.

        The first result value comes from:

        - 1.0 at (0, 0, 0)
        - 2.0 at (0, 0, 1)
        - 5.0 at (1, 0, 0)
        - 6.0 at (1, 0, 1)

        The second result value comes from:

        - 3.0 at (0, 1, 0)
        - 4.0 at (0, 1, 1)
        - 7.0 at (1, 1, 0)
        - 9.0 at (1, 1, 1)

        This again leaves one axis in place, so the result has shape (2,).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 9.0]],
            ]
        )

        valid_cases = [
            (
                (0,),
                (2, 2),
                [
                    ("sum", backend.sum, [[6.0, 8.0], [10.0, 13.0]]),
                    ("mean", backend.mean, [[3.0, 4.0], [5.0, 6.5]]),
                    ("max", backend.max, [[5.0, 6.0], [7.0, 9.0]]),
                    ("min", backend.min, [[1.0, 2.0], [3.0, 4.0]]),
                    ("std", backend.std, [[2.0, 2.0], [2.0, 2.5]]),
                ],
            ),
            (
                (1, 2),
                (2,),
                [
                    ("sum", backend.sum, [10.0, 27.0]),
                    ("mean", backend.mean, [2.5, 6.75]),
                    ("max", backend.max, [4.0, 9.0]),
                    ("min", backend.min, [1.0, 5.0]),
                    ("std", backend.std, [1.118033988749895, 1.479019945774904]),
                ],
            ),
            (
                (0, 2),
                (2,),
                [
                    ("sum", backend.sum, [14.0, 23.0]),
                    ("mean", backend.mean, [3.5, 5.75]),
                    ("max", backend.max, [6.0, 9.0]),
                    ("min", backend.min, [1.0, 3.0]),
                    ("std", backend.std, [2.0615528128088303, 2.384848003542364]),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result = method(tensor, axis=axes)
                        self.assertEqual(
                            backend.shape(result),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected)

    def test_reduction_methods_keep_reduced_axes_when_keepdims_is_true_with_3D_array(
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
                [1.0, 2.0],
                [3.0, 4.0]
            ],
            [
                [5.0, 6.0],
                [7.0, 9.0]
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
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 9.0]],
            ]
        )

        valid_cases = [
            (
                (1, 2),
                (2, 1, 1),
                [
                    ("sum", backend.sum, [[[10.0]], [[27.0]]]),
                    ("mean", backend.mean, [[[2.5]], [[6.75]]]),
                    ("max", backend.max, [[[4.0]], [[9.0]]]),
                    ("min", backend.min, [[[1.0]], [[5.0]]]),
                    (
                        "std",
                        backend.std,
                        [[[1.118033988749895]], [[1.479019945774904]]],
                    ),
                ],
            ),
            (
                (0, 2),
                (1, 2, 1),
                [
                    ("sum", backend.sum, [[[14.0], [23.0]]]),
                    ("mean", backend.mean, [[[3.5], [5.75]]]),
                    ("max", backend.max, [[[6.0], [9.0]]]),
                    ("min", backend.min, [[[1.0], [3.0]]]),
                    ("std", backend.std, [[[2.0615528128088303], [2.384848003542364]]]),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result = method(tensor, axis=axes, keepdims=True)
                        self.assertEqual(
                            backend.shape(result),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing with keepdims=True over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected)

    def test_reduction_methods_reduce_over_all_axes_with_4D_array(self):
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
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                [
                    [[2.0, 4.0], [6.0, 8.0]],
                    [[1.0, 3.0], [5.0, 9.0]],
                ],
            ]
        )

        reduction_methods = [
            ("sum", backend.sum, 74.0),
            ("mean", backend.mean, 4.625),
            ("max", backend.max, 9.0),
            ("min", backend.min, 1.0),
            ("std", backend.std, 2.4717149916606487),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                calls = [
                    lambda: method(tensor),
                    lambda: method(tensor, axis=None),
                    lambda: method(tensor, axis=(0, 1, 2, 3)),
                ]

                for call in calls:
                    result = call()
                    if method_name == "std":
                        assert_nested_close(result, expected)
                    else:
                        self.assertEqual(result, expected)

    def test_reduction_methods_reduce_over_a_tuple_of_axes_with_4D_array(self):
        # Some of the language here is taken from the NumPy docs on
        # ufunc.reduce. It's better than anything I could come up with
        # but still not great. It needs some work.
        """
        This tests reducing a 4D tensor over a tuple of axes.

        All cases in this test run across the same input tensor:

        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            [
                [[2.0, 4.0], [6.0, 8.0]],
                [[1.0, 3.0], [5.0, 9.0]],
            ],
        ]

        For the case axis=(1, 3), the second and fourth axes are reduced, and
        the first and third axes remain. So each value in the result comes from
        all the input values which share the same positions on axes 0 and 2.

        For example, the result value at position (0, 0) comes from:

        - 1.0 at (0, 0, 0, 0)
        - 2.0 at (0, 0, 0, 1)
        - 5.0 at (0, 1, 0, 0)
        - 6.0 at (0, 1, 0, 1)

        The result value at position (0, 1) comes from:

        - 3.0 at (0, 0, 1, 0)
        - 4.0 at (0, 0, 1, 1)
        - 7.0 at (0, 1, 1, 0)
        - 8.0 at (0, 1, 1, 1)

        The same pattern gives the result values at positions (1, 0) and
        (1, 1). Because axes 0 and 2 remain, the result has shape (2, 2).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                [
                    [[2.0, 4.0], [6.0, 8.0]],
                    [[1.0, 3.0], [5.0, 9.0]],
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
                        [[[3.0, 6.0], [9.0, 12.0]], [[6.0, 9.0], [12.0, 17.0]]],
                    ),
                    (
                        "mean",
                        backend.mean,
                        [[[1.5, 3.0], [4.5, 6.0]], [[3.0, 4.5], [6.0, 8.5]]],
                    ),
                    (
                        "max",
                        backend.max,
                        [[[2.0, 4.0], [6.0, 8.0]], [[5.0, 6.0], [7.0, 9.0]]],
                    ),
                    (
                        "min",
                        backend.min,
                        [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 3.0], [5.0, 8.0]]],
                    ),
                    (
                        "std",
                        backend.std,
                        [[[0.5, 1.0], [1.5, 2.0]], [[2.0, 1.5], [1.0, 0.5]]],
                    ),
                ],
            ),
            (
                (1, 2),
                (2, 2),
                [
                    ("sum", backend.sum, [[16.0, 20.0], [14.0, 24.0]]),
                    ("mean", backend.mean, [[4.0, 5.0], [3.5, 6.0]]),
                    ("max", backend.max, [[7.0, 8.0], [6.0, 9.0]]),
                    ("min", backend.min, [[1.0, 2.0], [1.0, 3.0]]),
                    (
                        "std",
                        backend.std,
                        [
                            [2.23606797749979, 2.23606797749979],
                            [2.0615528128088303, 2.5495097567963922],
                        ],
                    ),
                ],
            ),
            (
                (1, 3),
                (2, 2),
                [
                    ("sum", backend.sum, [[14.0, 22.0], [10.0, 28.0]]),
                    ("mean", backend.mean, [[3.5, 5.5], [2.5, 7.0]]),
                    ("max", backend.max, [[6.0, 8.0], [4.0, 9.0]]),
                    ("min", backend.min, [[1.0, 3.0], [1.0, 5.0]]),
                    (
                        "std",
                        backend.std,
                        [
                            [2.0615528128088303, 2.0615528128088303],
                            [1.118033988749895, 1.5811388300841898],
                        ],
                    ),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result = method(tensor, axis=axes)
                        self.assertEqual(
                            backend.shape(result),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected)

    def test_reduction_methods_keep_reduced_axes_when_keepdims_is_true_with_4D_array(
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
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                [
                    [[2.0, 4.0], [6.0, 8.0]],
                    [[1.0, 3.0], [5.0, 9.0]],
                ],
            ]
        )

        valid_cases = [
            (
                (1, 2),
                (2, 1, 1, 2),
                [
                    ("sum", backend.sum, [[[[16.0, 20.0]]], [[[14.0, 24.0]]]]),
                    ("mean", backend.mean, [[[[4.0, 5.0]]], [[[3.5, 6.0]]]]),
                    ("max", backend.max, [[[[7.0, 8.0]]], [[[6.0, 9.0]]]]),
                    ("min", backend.min, [[[[1.0, 2.0]]], [[[1.0, 3.0]]]]),
                    (
                        "std",
                        backend.std,
                        [
                            [[[2.23606797749979, 2.23606797749979]]],
                            [[[2.0615528128088303, 2.5495097567963922]]],
                        ],
                    ),
                ],
            ),
            (
                (1, 3),
                (2, 1, 2, 1),
                [
                    ("sum", backend.sum, [[[[14.0], [22.0]]], [[[10.0], [28.0]]]]),
                    ("mean", backend.mean, [[[[3.5], [5.5]]], [[[2.5], [7.0]]]]),
                    ("max", backend.max, [[[[6.0], [8.0]]], [[[4.0], [9.0]]]]),
                    ("min", backend.min, [[[[1.0], [3.0]]], [[[1.0], [5.0]]]]),
                    (
                        "std",
                        backend.std,
                        [
                            [[[2.0615528128088303], [2.0615528128088303]]],
                            [[[1.118033988749895], [1.5811388300841898]]],
                        ],
                    ),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result = method(tensor, axis=axes, keepdims=True)
                        self.assertEqual(
                            backend.shape(result),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing with keepdims=True over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected)

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
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                [
                    [[2.0, 4.0], [6.0, 8.0]],
                    [[1.0, 3.0], [5.0, 9.0]],
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
                        [[[6.0, 8.0], [10.0, 12.0]], [[3.0, 7.0], [11.0, 17.0]]],
                    ),
                    (
                        "mean",
                        backend.mean,
                        [[[3.0, 4.0], [5.0, 6.0]], [[1.5, 3.5], [5.5, 8.5]]],
                    ),
                    (
                        "max",
                        backend.max,
                        [[[5.0, 6.0], [7.0, 8.0]], [[2.0, 4.0], [6.0, 9.0]]],
                    ),
                    (
                        "min",
                        backend.min,
                        [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 3.0], [5.0, 8.0]]],
                    ),
                    (
                        "std",
                        backend.std,
                        [[[2.0, 2.0], [2.0, 2.0]], [[0.5, 0.5], [0.5, 0.5]]],
                    ),
                ],
            ),
            (
                3,
                [
                    (
                        "sum",
                        backend.sum,
                        [[[3.0, 7.0], [11.0, 15.0]], [[6.0, 14.0], [4.0, 14.0]]],
                    ),
                    (
                        "mean",
                        backend.mean,
                        [[[1.5, 3.5], [5.5, 7.5]], [[3.0, 7.0], [2.0, 7.0]]],
                    ),
                    (
                        "max",
                        backend.max,
                        [[[2.0, 4.0], [6.0, 8.0]], [[4.0, 8.0], [3.0, 9.0]]],
                    ),
                    (
                        "min",
                        backend.min,
                        [[[1.0, 3.0], [5.0, 7.0]], [[2.0, 6.0], [1.0, 5.0]]],
                    ),
                    (
                        "std",
                        backend.std,
                        [[[0.5, 0.5], [0.5, 0.5]], [[1.0, 1.0], [1.0, 2.0]]],
                    ),
                ],
            ),
        ]

        for axis, reduction_methods in valid_cases:
            with self.subTest(axis=axis):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result = method(tensor, axis=axis)
                        self.assertEqual(
                            backend.shape(result),
                            (2, 2, 2),
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing over axis {axis}"
                            ),
                        )
                        assert_nested_close(result, expected)

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
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 9.0]],
            ]
        )

        reduction_methods = [
            (
                "sum",
                backend.sum,
                [[4.0, 6.0], [12.0, 15.0]],
            ),
            (
                "mean",
                backend.mean,
                [[2.0, 3.0], [6.0, 7.5]],
            ),
            (
                "max",
                backend.max,
                [[3.0, 4.0], [7.0, 9.0]],
            ),
            (
                "min",
                backend.min,
                [[1.0, 2.0], [5.0, 6.0]],
            ),
            (
                "std",
                backend.std,
                [[1.0, 1.0], [1.0, 1.5]],
            ),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                integer_axis_result = method(tensor, axis=1)
                singleton_tuple_result = method(tensor, axis=(1,))
                self.assertEqual(
                    backend.shape(integer_axis_result),
                    backend.shape(singleton_tuple_result),
                    msg=(
                        f"{method_name} returned different shapes for axis=1 "
                        "and axis=(1,)"
                    ),
                )
                assert_nested_close(integer_axis_result, expected)
                assert_nested_close(singleton_tuple_result, expected)

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
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 9.0]],
            ]
        )

        valid_cases = [
            (
                -1,
                2,
                [
                    ("sum", backend.sum, [[3.0, 7.0], [11.0, 16.0]]),
                    ("mean", backend.mean, [[1.5, 3.5], [5.5, 8.0]]),
                    ("max", backend.max, [[2.0, 4.0], [6.0, 9.0]]),
                    ("min", backend.min, [[1.0, 3.0], [5.0, 7.0]]),
                    ("std", backend.std, [[0.5, 0.5], [0.5, 1.0]]),
                ],
            ),
            (
                (-3, -1),
                (0, 2),
                [
                    ("sum", backend.sum, [14.0, 23.0]),
                    ("mean", backend.mean, [3.5, 5.75]),
                    ("max", backend.max, [6.0, 9.0]),
                    ("min", backend.min, [1.0, 3.0]),
                    ("std", backend.std, [2.0615528128088303, 2.384848003542364]),
                ],
            ),
        ]

        for negative_axes, positive_axes, reduction_methods in valid_cases:
            with self.subTest(negative_axes=negative_axes, positive_axes=positive_axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        negative_result = method(tensor, axis=negative_axes)
                        positive_result = method(tensor, axis=positive_axes)
                        self.assertEqual(
                            backend.shape(negative_result),
                            backend.shape(positive_result),
                            msg=(
                                f"{method_name} returned different shapes for "
                                f"axis={negative_axes} and axis={positive_axes}"
                            ),
                        )
                        assert_nested_close(negative_result, expected)
                        assert_nested_close(positive_result, expected)

    def test_reduction_methods_remove_reduced_axes_when_keepdims_is_false(self):
        """
        This tests the behaviour of the reduction methods when keepdims=False.

        In these cases the reduced axes should be removed from the result,
        rather than being kept with size 1.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 9.0]],
            ]
        )

        valid_cases = [
            (
                (1, 2),
                (2,),
                [
                    ("sum", backend.sum, [10.0, 27.0]),
                    ("mean", backend.mean, [2.5, 6.75]),
                    ("max", backend.max, [4.0, 9.0]),
                    ("min", backend.min, [1.0, 5.0]),
                    ("std", backend.std, [1.118033988749895, 1.479019945774904]),
                ],
            ),
            (
                (0, 2),
                (2,),
                [
                    ("sum", backend.sum, [14.0, 23.0]),
                    ("mean", backend.mean, [3.5, 5.75]),
                    ("max", backend.max, [6.0, 9.0]),
                    ("min", backend.min, [1.0, 3.0]),
                    ("std", backend.std, [2.0615528128088303, 2.384848003542364]),
                ],
            ),
        ]

        for axes, expected_shape, reduction_methods in valid_cases:
            with self.subTest(axes=axes):
                for method_name, method, expected in reduction_methods:
                    with self.subTest(method=method_name):
                        result = method(tensor, axis=axes, keepdims=False)
                        self.assertEqual(
                            backend.shape(result),
                            expected_shape,
                            msg=(
                                f"{method_name} did not return the expected shape when "
                                f"reducing with keepdims=False over axis tuple {axes}"
                            ),
                        )
                        assert_nested_close(result, expected)


class BackendContractReductionInvalidAxisMixin(BackendContractBase):
    def test_reduction_methods_reject_duplicate_axes(self):
        pass

    def test_reduction_methods_reject_axes_outside_the_valid_range(self):
        pass


class BackendContractReductionEmptyInputMixin(BackendContractBase):
    def test_sum_returns_zero_when_called_on_an_empty_tensor(self):
        pass

    def test_mean_max_min_and_std_raise_when_called_on_an_empty_tensor(self):
        pass
