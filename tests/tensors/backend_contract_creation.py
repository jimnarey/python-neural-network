"""Test classes for tensor creation methods

These test classes enforce the contract for those methods which create
new tensors according to a specification (as distinct from converting
an existing structure into a tensor as to_tensor does).

The main things we care about is that new tensors are float-valued,
and that rank 0 tensors cannot be created (which involves passing
an empty shape to the method).
"""

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import to_python


def _all_values_are_floats(value) -> bool:
    if isinstance(value, list):
        return all(_all_values_are_floats(item) for item in value)
    return isinstance(value, float)


class BackendContractCreationMixin(BackendContractBase):
    def test_creation_methods_reject_empty_shape(self):
        backend = self.make_backend()

        creation_methods = [
            ("randn", lambda: backend.randn(())),
            ("zeros", lambda: backend.zeros(())),
            ("ones", lambda: backend.ones(())),
            ("full", lambda: backend.full((), 7.0)),
            ("empty", lambda: backend.empty(())),
        ]

        for method_name, call in creation_methods:
            with self.subTest(method=method_name):
                with self.assertRaises(
                    ValueError,
                    msg=f"{method_name} accepted an empty shape when it should reject it",
                ):
                    call()


class BackendContractFloatCreationMixin(BackendContractBase):
    def test_zeros_returns_float_values(self):
        backend = self.make_backend()
        result = to_python(backend.zeros((2, 2)))
        self.assertTrue(_all_values_are_floats(result))

    def test_ones_returns_float_values(self):
        backend = self.make_backend()
        result = to_python(backend.ones((2, 2)))
        self.assertTrue(_all_values_are_floats(result))

    def test_full_returns_float_values_when_given_an_int_fill_value(self):
        backend = self.make_backend()
        result = to_python(backend.full((2, 2), 1))
        self.assertTrue(_all_values_are_floats(result))

    def test_eye_returns_float_values(self):
        backend = self.make_backend()
        result = to_python(backend.eye(3))
        self.assertTrue(_all_values_are_floats(result))
