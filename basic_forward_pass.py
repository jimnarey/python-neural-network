# This represents the first run with two layers. It passes
# the whole dataset in as a single batch. Each of the rows
# in the reproduced output at the end represents a single
# sample. Because the samples were randomly generated these
# output probabilities are close to 1/3 each

from src import layer, activations

try:
    from src.sample_data import spiral
    from src.tensors import NumpyBackend
except (ModuleNotFoundError, RuntimeError) as exc:
    if isinstance(exc, ModuleNotFoundError) and exc.name != "numpy":
        raise
    raise SystemExit(
        "basic_forward_pass.py requires numpy for the spiral dataset "
        "generator and the NumPy tensor backend."
    ) from exc

X, y = spiral.generate_numpy(points=100, classes=3)
backend = NumpyBackend(seed=0)

layer_1 = layer.DenseLayer(backend, 2, 3)
layer_2 = layer.DenseLayer(backend, 3, 3, activation=activations.Softmax())

layer_1.forward(X)
layer_2.forward(layer_1.output)

print(layer_2.output[:5])  # type: ignore

# [[0.33333333 0.33333333 0.33333333]
#  [0.33331785 0.33335415 0.333328  ]
#  [0.33329276 0.33338632 0.33332091]
#  [0.33327083 0.33341186 0.33331731]
#  [0.33324785 0.33344125 0.3333109 ]]
