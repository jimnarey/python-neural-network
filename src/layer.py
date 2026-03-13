from typing import Optional

from src import activations
from src.activations import Activation
from src.tensors import Tensor, TensorBackend


class DenseLayer:
    """
    Models a layer of neurons in which each neuron receives all of the outputs from
    all neurons in the previous layer.
    """

    def __init__(
        self,
        backend: TensorBackend,
        num_inputs: int,
        num_neurons: int,
        activation: Optional[Activation] = None,
        weight_mod: float = 0.1,
    ):
        self.backend = backend
        # weight_mod constrains the initial weights to small numbers. As values pass
        # through the network they are multiplied repeatedly, as are the resources
        # required to manage them.
        # Passing the arguments in this order avoids a transpose later
        self.weights = self.backend.multiply(
            self.backend.randn((num_inputs, num_neurons)), weight_mod
        )
        # array([[0., 0., 0., 0.]]) where num_neurons is 4
        self.biases = self.backend.zeros((1, num_neurons))
        self.output: Tensor | None = None
        self.activation = activation if activation is not None else activations.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        pre_act = self.backend.add(
            self.backend.matmul(inputs, self.weights), self.biases
        )
        # We check we're not getting a dead network by monitoring the output
        # for too many zeros. We can mitigate this by setting the biases to non-zero
        self.output = self.activation.forward(self.backend, pre_act)
        return self.output
