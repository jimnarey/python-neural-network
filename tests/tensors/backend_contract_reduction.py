from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close


class BackendContractReductionBehaviourMixin(BackendContractBase):
    def test_reduction_methods_reduce_over_all_axes_when_axis_is_omitted_or_none(
        self,
    ):
        backend = self.make_backend()
        # Values have been chosen to avoid the need to accomodate
        # rounding differences between backends.
        tensor = backend.to_tensor([[1.0, 2.0], [1.0, 2.0]])

        reduction_methods = [
            ("sum", backend.sum, 6.0),
            ("mean", backend.mean, 1.5),
            ("max", backend.max, 2.0),
            ("min", backend.min, 1.0),
            ("std", backend.std, 0.5),
        ]

        for method_name, method, expected in reduction_methods:
            with self.subTest(method=method_name):
                omitted_result = method(tensor)
                explicit_none_result = method(tensor, axis=None)
                self.assertEqual(
                    omitted_result,
                    explicit_none_result,
                    msg=(
                        f"{method_name} with axis=None did not match the result "
                        "when the axis argument was omitted"
                    ),
                )
                self.assertEqual(
                    explicit_none_result,
                    expected,
                    msg=(
                        f"{method_name} with axis=None did not return the "
                        "expected scalar value"
                    ),
                )
                self.assertIsInstance(
                    explicit_none_result,
                    float,
                    msg=f"{method_name} returned {explicit_none_result!r} instead of a float",
                )

    def test_reduction_methods_reduce_over_a_single_integer_axis(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0], [1.0, 2.0]])

        valid_cases = [
            ("sum", backend.sum, [3.0, 3.0]),
            ("mean", backend.mean, [1.5, 1.5]),
            ("max", backend.max, [2.0, 2.0]),
            ("min", backend.min, [1.0, 1.0]),
            (
                "std",
                backend.std,
                [0.5, 0.5],
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
