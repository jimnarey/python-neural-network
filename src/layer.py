import numpy as np

# Check the scope of this
np.random.seed(0)


class DenseLayer:

    def __init__(self, num_inputs: int, num_neurons: int, weight_mod: float = 0.1):
        # weight_mod constrains the initial weights to small numbers. As values pass
        # through the network they are multiplied repeatedly, as are the resources
        # required to manage them.
        # Passing the arguments in this order avoids a transpose later
        self.weights = weight_mod * np.random.randn(num_inputs, num_neurons)
        # array([[0., 0., 0., 0.]]) where num_neurons is 4
        self.biases: np.ndarray = np.zeros((1, num_neurons))
        self.output: np.ndarray | None = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
