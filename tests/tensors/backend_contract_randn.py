from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close

# TODO - consider whether to use a broader range of shapes in these tests


class BackendContractRandnMixin(BackendContractBase):
    """
    This class tests the important behaviour of the backend randn methods
    but *does not* test the randomness of the values produced. It is assumed
    that in each case randomness will be provided by built-in features of the
    the language used. What we do need to test - in order to write new
    backends with confidence - is how passing a seed affects the values returned
    and that we get those values in arrays with the expected dimensions.
    """

    def test_randn_returns_float_values(self):
        backend = self.make_backend(seed=0)
        tensor = backend.randn((10,))
        result = backend.to_python(tensor)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 10)
        for value in result:
            self.assertIsInstance(value, float)

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

    def test_backends_constructed_with_same_seed_produce_same_first_draw(self):
        first_backend = self.make_backend(seed=0)
        second_backend = self.make_backend(seed=0)
        first_draw_tensor = first_backend.randn((2, 3))
        second_draw_tensor = second_backend.randn((2, 3))
        first_draw = first_backend.to_python(first_draw_tensor)
        second_draw = second_backend.to_python(second_draw_tensor)
        assert_nested_close(first_draw, second_draw)

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

    def test_backends_constructed_with_same_seed_produce_same_sequence(self):
        """
        Example: first and second backends both return:
                 [
                   [0.1, -0.2, 0.3],
                   [0.4, -0.5, 0.6]
                 ]
                 then both return:
                 [
                   [-0.7, 0.8, -0.9],
                   [1.0, -1.1, 1.2]
                 ]

        Note that these are *examples*. This test doesn't prove that
        a given backend instance produces different return values
        on each corresponding draw, just that two backend instances with
        the same seed produce the same values for each draw (first, second, etc)
        """
        first_backend = self.make_backend(seed=0)
        second_backend = self.make_backend(seed=0)
        first_backend_draw_1_tensor = first_backend.randn((2, 3))
        second_backend_draw_1_tensor = second_backend.randn((2, 3))
        first_backend_draw_1 = first_backend.to_python(first_backend_draw_1_tensor)
        second_backend_draw_1 = second_backend.to_python(second_backend_draw_1_tensor)
        assert_nested_close(first_backend_draw_1, second_backend_draw_1)
        first_backend_draw_2_tensor = first_backend.randn((2, 3))
        second_backend_draw_2_tensor = second_backend.randn((2, 3))
        first_backend_draw_2 = first_backend.to_python(first_backend_draw_2_tensor)
        second_backend_draw_2 = second_backend.to_python(second_backend_draw_2_tensor)
        assert_nested_close(first_backend_draw_2, second_backend_draw_2)
