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
        # weight_mod constrains the initial weights to small numbers. As values pass
        # through the network they are multiplied repeatedly, as are the resources
        # required to manage them.
        weight_mod: float = 0.1,
    ):
        self.backend = backend

        # We store weights as (inputs, neurons) so inputs @ weights produces
        # one output value per neuron.
        self.weights = self.backend.multiply(
            self.backend.randn((num_inputs, num_neurons)), weight_mod
        )
        # One bias per neuron
        self.biases = self.backend.zeros((num_neurons,))
        self.output: Tensor | None = None
        self.activation = activation if activation is not None else activations.ReLU()
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

    def forward(self, inputs: Tensor) -> Tensor:
        # Consider whether we can achieve a slight optimisation with :=
        input_shape = self.backend.shape(inputs)
        if input_shape[-1] != self.num_inputs:
            raise ValueError(
                f"DenseLayer expected the last input dimension to be {self.num_inputs}, "
                f"got {input_shape[-1]}."
            )

        pre_act = self.backend.add(
            # Multiply the input tensor by the layer weights.
            self.backend.matmul(inputs, self.weights),
            self.biases,
        )
        # We check we're not getting a dead network by monitoring the output
        # for too many zeros. We can mitigate this by setting the biases to non-zero
        self.output = self.activation.forward(self.backend, pre_act)
        return self.output
