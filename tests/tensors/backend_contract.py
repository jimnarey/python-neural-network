import math

# TODO - come back to type checking in this module (for the helpers), once we
# have pinned down the Tensor type declared in backend_contract.py


class BackendContractMixin:
    def make_backend(self):
        raise NotImplementedError

    def to_python(self, tensor):
        """
        Convert a backend-native tensor into plain Python values so the same
        assertions work for list-based and NumPy-backed implementations.
        """
        if hasattr(tensor, "tolist"):
            return tensor.tolist()
        return tensor

    def assert_nested_close(self, actual, expected, dec_places=7):
        """
        Assert that two tensors/matrices with an arbitary number of dimensions
        are the same.
        """
        # Expected is always a native Python type as it is declared in our tests
        actual_value = self.to_python(actual)
        self._assert_nested_close(actual_value, expected, dec_places=dec_places)

    def _assert_nested_close(self, actual, expected, dec_places=7):
        # If 'expected' is a list or tuple check that 'actual' is the same
        # type and then compare them
        if isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, (list, tuple))
            self.assertEqual(len(actual), len(expected))
            # Iterate over the items in expected and actual in parallel and call
            # this method recursively
            for actual_item, expected_item in zip(actual, expected, strict=True):
                self._assert_nested_close(
                    actual_item, expected_item, dec_places=dec_places
                )
            return

        # If expected/actual are not lists/tuples we assume they are scalar values
        # and compare them. Because we are dealing with floats, which might be
        # rounded differently by different backends, we limit the number of decimal
        # places used for the comparison
        self.assertTrue(
            math.isclose(
                actual,
                expected,
                rel_tol=1e-7,  # Max allowable difference, relative to the larges of a and b
                abs_tol=10
                ** (
                    -dec_places - 1
                ),  # Max absolute difference, needed for values close to zero
            ),
            msg=f"{actual!r} != {expected!r}",
        )

    def test_matmul_multiplies_2d_matrices(self):
        backend = self.make_backend()

        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]

        result = backend.matmul(a, b)

        expected = [
            [21.0, 24.0, 27.0],
            [47.0, 54.0, 61.0],
        ]
        self.assert_nested_close(result, expected)
