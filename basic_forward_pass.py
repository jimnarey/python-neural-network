# This represents the first run with two layers. It passes
# the whole dataset in as a single batch. Each of the rows
# in the reproduced output at the end represents a single
# sample. Because the samples were randomly generated these
# output probabilities are close to 1/3 each

from src import layer, activations, spiral

X, y = spiral.spiral_data(points=100, classes=3)

layer_1 = layer.DenseLayer(2, 3)
layer_2 = layer.DenseLayer(3, 3, activation=activations.softmax_np)

layer_1.forward(X)
layer_2.forward(layer_1.output)

print(layer_2.output[:5])

# [[0.33333333 0.33333333 0.33333333]
#  [0.33331734 0.33331832 0.33336434]
#  [0.3332888  0.33329153 0.33341967]
#  [0.33325941 0.33326395 0.33347665]
#  [0.33323311 0.33323926 0.33352763]]
