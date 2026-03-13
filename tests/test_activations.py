import importlib.util
from unittest import TestCase, skipUnless

from src import activations

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    import numpy as np


@skipUnless(NUMPY_AVAILABLE, "numpy is not installed")
class TestSoftmax(TestCase):

    def setUp(self):
        self.neuron_outputs_simple = [4.8, 1.21, 2.385]
        self.expected_values_simple = [0.89528266, 0.02470831, 0.08000903]
        from src.tensors import NumpyBackend

        self.backend = NumpyBackend()

    def test_softmax(self):
        activation_outputs = activations.Softmax().forward(
            self.backend, np.array([self.neuron_outputs_simple])
        )
        self.assertTrue(
            np.allclose(
                activation_outputs,
                np.array([self.expected_values_simple]),
                rtol=1e-6,
                atol=1e-8,
            )
        )
