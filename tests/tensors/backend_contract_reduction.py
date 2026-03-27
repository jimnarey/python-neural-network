from tests.tensors.backend_contract_shared import BackendContractBase


class BackendContractScalarReturnTypeMixin(BackendContractBase):
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
