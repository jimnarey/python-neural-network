from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_helpers import all_values_are_floats


class BackendReferenceCreationValueTypeMixin(BackendContractBase):
    def test_zeros_returns_float_values(self):
        backend = self.make_backend()
        tensor = backend.zeros((2, 2))
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_ones_returns_float_values(self):
        backend = self.make_backend()
        tensor = backend.ones((2, 2))
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_full_returns_float_values_when_given_float_fill_value(self):
        backend = self.make_backend()
        tensor = backend.full((2, 2), 1.0)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_full_returns_float_values_when_given_an_int_fill_value(self):
        backend = self.make_backend()
        tensor = backend.full((2, 2), 1)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_eye_returns_float_values(self):
        backend = self.make_backend()
        tensor = backend.eye(3)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))


class BackendReferenceCreationLikeValueTypeMixin(BackendContractBase):
    def test_zeros_like_returns_float_values(self):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor = backend.zeros_like(source_tensor)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_ones_like_returns_float_values(self):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor = backend.ones_like(source_tensor)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_full_like_returns_float_values_when_given_float_fill_value(self):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor = backend.full_like(source_tensor, 1.0)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))

    def test_full_like_returns_float_values_when_given_an_int_fill_value(self):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor = backend.full_like(source_tensor, 1)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))


class BackendReferenceCopyMixin(BackendContractBase):
    def test_copy_returns_float_values(self):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor = backend.copy(source_tensor)
        result = backend.to_python(tensor)
        self.assertTrue(all_values_are_floats(result))
