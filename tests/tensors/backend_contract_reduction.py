from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close


class BackendContractReductionBehaviourMixin(BackendContractBase):

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

        For example, sum adds every value in the tensor, so for
        [[1.0, 2.0], [3.0, 4.0]] the result is 10.0.

        std calculates the standard deviation of all the values in the
        tensor. It first finds the mean, which here is 2.5, then looks at
        how far each value lies from that mean: -1.5, -0.5, 0.5 and
        1.5. It squares those differences, takes their mean, and then
        takes the square root. For this tensor that about 1.118033988749895.

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

    def test_reduction_methods_reduce_over_a_single_integer_axis(self):
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

    def test_reduction_methods_reduce_over_a_tuple_of_axes_with_4D_array(self):
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

    def test_reduction_methods_treat_axis_1_and_axis_singleton_tuple_1_as_equivalent(
        self,
    ):
        pass

    def test_reduction_methods_accept_negative_axes(self):
        pass

    def test_reduction_methods_remove_reduced_axes_when_keepdims_is_false(self):
        pass


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
