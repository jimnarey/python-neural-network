from tests.tensors.backend_contract_shared import BackendContractBase


def _all_values_are_floats(value) -> bool:
    if isinstance(value, list):
        return all(_all_values_are_floats(item) for item in value)
    return isinstance(value, float)


# TODO - Add stubs and consider whether any native representation tests
# are needed for empty, empty-like.


class BackendReferenceCreationValueTypeMixin(BackendContractBase):
    def test_zeros_returns_float_values(self):
        backend = self.make_backend()
        tensor = backend.zeros((2, 2))
        result = backend.to_python(tensor)
        self.assertTrue(_all_values_are_floats(result))

    def test_ones_returns_float_values(self):
        backend = self.make_backend()
        tensor = backend.ones((2, 2))
        result = backend.to_python(tensor)
        self.assertTrue(_all_values_are_floats(result))

    def test_full_returns_float_values_when_given_an_int_fill_value(self):
        backend = self.make_backend()
        tensor = backend.full((2, 2), 1)
        result = backend.to_python(tensor)
        self.assertTrue(_all_values_are_floats(result))

    def test_eye_returns_float_values(self):
        backend = self.make_backend()
        tensor = backend.eye(3)
        result = backend.to_python(tensor)
        self.assertTrue(_all_values_are_floats(result))
