from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close


class BackendReferenceRandnMixin(BackendContractBase):
    """ "
    Tests for behaviour of randn methods which falls outside of the contract
    but is part of the reference design. We do not test the randomness of the
    values returned by the methods (we assume the language used handles this).
    What we do need to test - in order to write new backends with confidence -
    is how passing a seed affects the values returned and that we get those
    values in arrays with the expected dimensions.

    We also test that the values returned by randn are floats.
    """

    def test_randn_returns_float_values(self):
        backend = self.make_backend(seed=0)
        tensor = backend.randn((10,))
        result = backend.to_python(tensor)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 10)
        for value in result:
            self.assertIsInstance(value, float)

    def test_backends_constructed_with_same_seed_produce_same_first_draw(self):
        first_backend = self.make_backend(seed=0)
        second_backend = self.make_backend(seed=0)
        first_draw_tensor = first_backend.randn((2, 3))
        second_draw_tensor = second_backend.randn((2, 3))
        first_draw = first_backend.to_python(first_draw_tensor)
        second_draw = second_backend.to_python(second_draw_tensor)
        assert_nested_close(first_draw, second_draw)

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
