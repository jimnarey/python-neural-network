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
        We assert_nested_close for testing the output of std for consistency with
        the other tests but could have used math.to_close() directly, or
        self.assertIsClose(), to achieve the same effect.
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
        tensor = backend.to_tensor([[1.0, 2.0], [3.0, 5.0]])

        valid_cases = [
            ("sum", backend.sum, [3.0, 8.0]),
            ("mean", backend.mean, [1.5, 4.0]),
            ("max", backend.max, [2.0, 5.0]),
            ("min", backend.min, [1.0, 3.0]),
            (
                "std",
                backend.std,
                [0.5, 1.0],
            ),
        ]

        for method_name, method, expected in valid_cases:
            with self.subTest(method=method_name):
                result = method(tensor, axis=1)
                self.assertEqual(
                    backend.shape(result),
                    (2,),
                    msg=(
                        f"{method_name} did not return the expected shape when "
                        "reducing over a single axis"
                    ),
                )
                if method_name == "std":
                    assert_nested_close(result, expected)
                else:
                    assert_nested_close(result, expected)

    def test_reduction_methods_reduce_over_a_tuple_of_axes(self):
        pass

    def test_reduction_methods_treat_axis_1_and_axis_singleton_tuple_1_as_equivalent(
        self,
    ):
        pass

    def test_reduction_methods_accept_negative_axes(self):
        pass

    def test_reduction_methods_keep_reduced_axes_with_length_1_when_keepdims_is_true(
        self,
    ):
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
