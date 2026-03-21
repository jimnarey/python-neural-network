from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close, to_python


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
        result = to_python(backend.randn((10,)))
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
        result = to_python(backend.randn((3,)))
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
        result = to_python(backend.randn((2, 3)))
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
        result = to_python(backend.randn((2, 3, 4)))
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
        result = to_python(backend.randn((0,)))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_backends_constructed_with_same_seed_produce_same_first_draw(self):
        first_backend = self.make_backend(seed=0)
        second_backend = self.make_backend(seed=0)
        first_draw = first_backend.randn((2, 3))
        second_draw = second_backend.randn((2, 3))
        assert_nested_close(first_draw, to_python(second_draw))

    def test_successive_randn_calls_return_different_values(self):
        """
        Demonstrates that for the same backend instance, the subsequent draws
        differ
        """
        backend = self.make_backend(seed=0)
        first_draw = to_python(backend.randn((2, 3)))
        second_draw = to_python(backend.randn((2, 3)))
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
        first_backend_draw_1 = first_backend.randn((2, 3))
        second_backend_draw_1 = second_backend.randn((2, 3))
        assert_nested_close(first_backend_draw_1, to_python(second_backend_draw_1))
        first_backend_draw_2 = first_backend.randn((2, 3))
        second_backend_draw_2 = second_backend.randn((2, 3))
        assert_nested_close(first_backend_draw_2, to_python(second_backend_draw_2))
