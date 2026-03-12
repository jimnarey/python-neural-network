from typing import Optional, Callable
import numpy as np

from src import activations

# Check the scope of this
np.random.seed(0)


class DenseLayer:
    """
    Models a layer of neurons in which each neuron receives all of the outputs from
    all neurons in the previous layer.
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        activation: Optional[Callable] = None,
        weight_mod: float = 0.1,
    ):
        # weight_mod constrains the initial weights to small numbers. As values pass
        # through the network they are multiplied repeatedly, as are the resources
        # required to manage them.
        # Passing the arguments in this order avoids a transpose later
        self.weights = weight_mod * np.random.randn(num_inputs, num_neurons)
        # array([[0., 0., 0., 0.]]) where num_neurons is 4
        self.biases: np.ndarray = np.zeros((1, num_neurons))
        self.output = np.ndarray(1)
        self.activation = activation if activation is not None else activations.relu

    def forward(self, inputs):
        pre_act = np.dot(inputs, self.weights) + self.biases
        # We check we're not getting a dead network by monitoring the output
        # for too many zeros. We can mitigate this by setting the biases to non-zero
        self.output = self.activation(pre_act)
