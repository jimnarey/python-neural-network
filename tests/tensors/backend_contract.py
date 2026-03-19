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
"""

from src.tensors.backend import TensorBackend


class BackendContractBase:
    def make_backend(self, seed: int | None = None):
        raise NotImplementedError


class BackendConstructionContractMixin(BackendContractBase):
    def _construct_backend(self, message: str, seed: int | None = None):
        try:
            return self.make_backend(seed=seed)
        except Exception as exc:
            # This mixin only makes any sense if it is inherited by a class
            # which also inherits from unittest.TestCase
            self.fail(f"{message}: {exc}")  # type: ignore

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
