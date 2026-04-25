from tests.tensors.backend_contract_shared import BackendContractBase


class BackendReferenceElementwiseFloatValueMixin(BackendContractBase):
    def test_elementwise_methods_return_float_valued_tensors_for_same_shape_inputs(
        self,
    ):
        pass

    def test_elementwise_methods_return_float_valued_tensors_when_broadcasting(self):
        pass

    def test_elementwise_methods_return_float_valued_tensors_when_given_int_scalar_rhs(
        self,
    ):
        pass


class BackendReferenceElementwiseArithmeticMixin(BackendContractBase):
    def test_add_and_subtract_return_expected_results_with_non_integer_values(self):
        pass

    def test_multiply_returns_expected_results_with_non_integer_values(self):
        pass

    def test_divide_returns_expected_results_with_non_integer_values(self):
        pass

    def test_maximum_and_minimum_return_expected_results_with_non_integer_values(
        self,
    ):
        pass
