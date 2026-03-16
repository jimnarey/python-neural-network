"""Base class for backend contract tests

Together with the TensorBackend protocol class, the classes which inherit
from BackendContractBase define the contract between the network and the
tensor backends. It was built againstthe NumPy backend. Because that backend
is a very, very thin wrapper around NumPy's functions it means we can have a
high level of certaintly about other backends which also pass these tests.

One consequence of this approach is that it was decided to mirror some of
NumPy's specific behaviour across all backends, in particular broadcasting
and trailing axes matmul. Otherwise, we start to lose some of the guarantees
provided by using NumPy as the reference implementation.
"""


class BackendContractBase:
    def make_backend(self):
        raise NotImplementedError
