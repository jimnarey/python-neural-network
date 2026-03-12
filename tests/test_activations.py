from unittest import TestCase
import numpy as np

from src import activations


class TestSoftmax(TestCase):

    def setUp(self):
        self.neuron_outputs_simple = [4.8, 1.21, 2.385]
        # It's slightly curious that these exact values match in both implementations,
        # perhaps because we use numpy as an intermediary in the 'pure' Python implementation
        # The need to do some rounding was expected...
        self.expected_values_simple = [0.89528266, 0.02470831, 0.08000903]

    def test_softmax_np(self):
        activation_outputs = activations.softmax_np(
            np.array([self.neuron_outputs_simple])
        )
        self.assertTrue(
            np.allclose(
                activation_outputs,
                np.array([self.expected_values_simple]),
                rtol=1e-6,
                atol=1e-8,
            )
        )

    def test_softmax_py(self):
        activation_outputs = activations.softmax_py(
            np.array([self.neuron_outputs_simple])
        )
        import logging

        logging.error(activation_outputs)
        self.assertTrue(
            np.allclose(
                activation_outputs,
                np.array([self.expected_values_simple]),
                rtol=1e-6,
                atol=1e-8,
            )
        )
