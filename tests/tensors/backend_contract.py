from tests.helpers.tensor_assertions import assert_nested_close


class BackendContractMixin:
    def make_backend(self):
        raise NotImplementedError

    def test_matmul_multiplies_2d_matrices(self):
        backend = self.make_backend()

        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]

        result = backend.matmul(a, b)

        expected = [
            [21.0, 24.0, 27.0],
            [47.0, 54.0, 61.0],
        ]
        assert_nested_close(result, expected)
