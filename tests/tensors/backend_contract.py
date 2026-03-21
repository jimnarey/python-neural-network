"""Base class for backend contract tests

Together with the TensorBackend protocol class, the classes which inherit
from BackendContractBase define the contract between the network and the
tensor backends. It was built against the NumPy backend. Because that backend
is a very, very thin wrapper around NumPy's functions it means we can have a
high level of certainty about other backends which also pass these tests.

One consequence of this approach is that it was decided to mirror some of
NumPy's specific behaviour across all backends, in particular broadcasting
and trailing axes matmul. Otherwise, we start to lose some of the guarantees
provided by using NumPy as the reference implementation.

How inheritance for backend test classes works:
- Each test class covering a specific backend method (e.g. BackendContractMatmulMixin)
  must inherit from BackendContractBase. This enforces implementation of make_backend.
- BackendConstructionContractMixin also inherits from BackendContractBase (see below).
- The test classes which are responsible for running the tests against a specific
  backend implementation (e.g. TestNumpyBackend) inherit from BackendConstructionContractMixin
  and *all of* the method mixins (BackendContractMatmulMixin, BackendContractRandnMixin etc).
- This is a bit more complex than is ideal but it separates concerns while ensuring the
  requirement to implement make_backend is enforced even in every mixin that uses it.
- Having the mixins NOT inherit from TestCase means that they are not picked up during test
  discovery, which would cause an exception because they don't have a concrete make_backend

One consequence of this approach is that have test methods in classes (the mixins) which
do not inherit from TestCase. The 'if TYPE_CHECKING' block enables the type checker to
recognise when we are calling methods inherited from TestCase and treat them properly.
It also has the highly desirable effect of making IntelliSense work properly in VSCode.
"""

# The NumPy backend tests currently check that methods which accept an existing tensor
# reject NumPy rank 0 arrays and NumPy scalar values. This is reasonably safe for now,
# because rank 0 values are, so far, a NumPy concern in this project and we will not be
# mixing backends within the same network instance. It is also unclear how shared tests
# for this rule could be written without importing NumPy. There is still an argument that
# these tests should eventually move here if rejecting rank 0 tensors becomes part of the
# shared backend contract. This is not the case with the other tests designed to stop
# rank 0 values leaking into the application, since those tests are specifically about
# values produced by the NumPy backend and therefore properly belong in the NumPy-specific
# tests.

from typing import TYPE_CHECKING

from src.tensors.backend import TensorBackend
from tests.helpers.tensor_assertions import to_python

if TYPE_CHECKING:
    import unittest

    class _BackendTestCase(unittest.TestCase):
        def make_backend(self, seed: int | None = None) -> TensorBackend:
            raise NotImplementedError

else:

    class _BackendTestCase:
        pass


class BackendContractBase(_BackendTestCase):
    def make_backend(self, seed: int | None = None) -> TensorBackend:
        raise NotImplementedError


class BackendContractConstructionMixin(BackendContractBase):
    def _construct_backend(
        self,
        message: str,
        seed: int | None = None,
    ) -> TensorBackend:
        try:
            return self.make_backend(seed=seed)
        except Exception as exc:
            # The following line is an example of why the 'if TYPE_CHECKING' block is
            # helpful. Without the latter this line will fail type checking.
            self.fail(f"{message}: {exc}")

    def test_backend_can_be_constructed_without_seed(self):
        backend = self._construct_backend(
            "Backend construction without a seed raised an exception"
        )
        self.assertIsNotNone(backend)

    def test_backend_can_be_constructed_with_seed(self):
        backend = self._construct_backend(
            "Backend construction with a seed raised an exception",
            seed=0,
        )
        self.assertIsNotNone(backend)

    def test_backend_can_be_constructed_with_explicit_none_seed(self):
        backend = self._construct_backend(
            "Backend construction with an explicit None seed raised an exception",
            seed=None,
        )
        self.assertIsNotNone(backend)

    def test_constructed_backend_implements_tensor_backend_protocol(self):
        backend = self._construct_backend(
            "Constructing a backend to check protocol conformance raised an exception"
        )
        self.assertIsInstance(backend, TensorBackend)

    def test_make_backend_returns_distinct_instances(self):
        first_backend = self.make_backend()
        second_backend = self.make_backend()
        self.assertIsNot(first_backend, second_backend)


class BackendContractCreationMixin(BackendContractBase):
    def test_creation_methods_reject_empty_shape(self):
        backend = self.make_backend()

        creation_methods = [
            ("randn", lambda: backend.randn(())),
            ("zeros", lambda: backend.zeros(())),
            ("ones", lambda: backend.ones(())),
            ("full", lambda: backend.full((), 7)),
            ("empty", lambda: backend.empty(())),
        ]

        for method_name, call in creation_methods:
            with self.subTest(method=method_name):
                with self.assertRaises(
                    ValueError,
                    msg=f"{method_name} accepted an empty shape when it should reject it",
                ):
                    call()

    def test_reshape_rejects_empty_shape(self):
        backend = self.make_backend()

        with self.assertRaises(
            ValueError,
            msg="reshape accepted an empty shape when it should reject it",
        ):
            backend.reshape([1.0], ())


class BackendContractToTensorInputMixin(BackendContractBase):
    def test_to_tensor_accepts_list_input(self):
        backend = self.make_backend()
        backend.to_tensor([1, 2, 3])

    def test_to_tensor_accepts_tuple_input(self):
        backend = self.make_backend()
        backend.to_tensor((1, 2, 3))

    def test_to_tensor_rejects_plain_scalar_values(self):
        backend = self.make_backend()
        for data in (1, 1.5):
            with self.subTest(data=data):
                with self.assertRaises(
                    ValueError,
                    msg="to_tensor accepted a plain scalar value when it should reject it",
                ):
                    backend.to_tensor(data)

    def test_to_tensor_rejects_ragged_input(self):
        backend = self.make_backend()
        with self.assertRaises(
            ValueError,
            msg="to_tensor accepted ragged input when it should reject it",
        ):
            backend.to_tensor([[1, 2], [3]])


class BackendContractToTensorShapeMixin(BackendContractBase):
    def test_to_tensor_converts_1D_input_to_tensor(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor([1, 2, 3]))
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_to_tensor_converts_2D_input_to_tensor(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor(((1, 2), (3, 4))))
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_to_tensor_converts_3D_input_to_tensor(self):
        backend = self.make_backend()
        result = to_python(
            backend.to_tensor(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                ]
            )
        )
        self.assertEqual(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        )


class BackendContractToTensorValueMixin(BackendContractBase):
    def test_to_tensor_converts_integer_values_to_float(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor([1, 2, 3]))
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_to_tensor_preserves_float_values(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor([1.5, 2.5, 3.5]))
        self.assertEqual(result, [1.5, 2.5, 3.5])

    def test_to_tensor_normalises_mixed_numeric_input_to_float(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor([1, 2.5, 3]))
        self.assertEqual(result, [1.0, 2.5, 3.0])
