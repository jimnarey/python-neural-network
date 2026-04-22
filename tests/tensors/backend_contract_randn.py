from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures

# TODO - consider whether to use a broader range of shapes in these tests

# The randn reference tests include tests to ensure that we get the same
# consecutive values from randn when using the same seed. This is
# universally-desirable behaviour for randn methods but the reference
# implementation backend is float-based so those tests cannot be run
# against all backends and this module is therefore not the right place
# for them. This means that additional tests may be needed for any future,
# int-based backends.


@EnforceSharedNumericFixtures()
class BackendContractRandnMixin(BackendContractBase):
    """
    This class tests the important behaviour of the backend randn methods
    but *does not* test the randomness of the values produced. It is assumed
    that in each case randomness will be provided by built-in features of the
    the language used.
    """

    def test_randn_returns_1D_array_with_requested_shape(self):
        """
        Example:
             [0.1, -0.2, 0.3]
        """
        backend = self.make_backend(seed=0)
        tensor = backend.randn((3,))
        result = backend.to_python(tensor)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_randn_returns_2D_array_with_requested_shape(self):
        """
        Example: [
                      [0.1, -0.2, 0.3],
                      [0.4, -0.5, 0.6]
                    ]
        """
        backend = self.make_backend(seed=0)
        tensor = backend.randn((2, 3))
        result = backend.to_python(tensor)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), 3)

    def test_randn_returns_3D_array_with_requested_shape(self):
        """
        Example: [
                      [
                        [0.1, -0.2],
                        [0.3, 0.4]
                      ],
                      [
                        [-0.5, 0.6],
                        [0.7, -0.8]
                      ]
                    ]
        """
        backend = self.make_backend(seed=0)
        tensor = backend.randn((2, 3, 4))
        result = backend.to_python(tensor)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for matrix in result:
            self.assertIsInstance(matrix, list)
            self.assertEqual(len(matrix), 3)
            for row in matrix:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 4)

    def test_randn_supports_zero_length_dimensions(self):
        """
        Example: []

        This demonstrates that we can use randn to produce an
        array with one or more empty dimensions.
        """
        backend = self.make_backend(seed=0)
        tensor = backend.randn((0,))
        result = backend.to_python(tensor)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_successive_randn_calls_return_different_values(self):
        """
        Demonstrates that for the same backend instance, the subsequent draws
        differ
        """
        backend = self.make_backend(seed=0)
        first_draw_tensor = backend.randn((2, 3))
        first_draw = backend.to_python(first_draw_tensor)
        second_draw_tensor = backend.randn((2, 3))
        second_draw = backend.to_python(second_draw_tensor)
        self.assertNotEqual(first_draw, second_draw)
