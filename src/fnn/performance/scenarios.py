from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fnn.tensors.numpy_backend import NumpyBackend
from fnn.tensors.python_backend.python_backend import PythonBackend

type BenchmarkCallable = Callable[[], object]


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    callable_name: str
    description: str
    argument_description: str
    make_callable: Callable[[], BenchmarkCallable]


BATCH_SIZE = 32
INPUT_SIZE = 64
OUTPUT_SIZE = 32


def _matrix(rows: int, columns: int) -> list[list[float]]:
    return [
        [float(((row * columns + column) % 17) - 8) / 8.0 for column in range(columns)]
        for row in range(rows)
    ]


def _vector(size: int) -> list[float]:
    return [float((index % 7) - 3) / 7.0 for index in range(size)]


def _make_dense_forward_callable(backend: Any) -> BenchmarkCallable:
    inputs = backend.to_tensor(_matrix(BATCH_SIZE, INPUT_SIZE))
    weights = backend.to_tensor(_matrix(INPUT_SIZE, OUTPUT_SIZE))
    biases = backend.to_tensor(_vector(OUTPUT_SIZE))

    def run_dense_forward() -> object:
        matmul_result = backend.matmul(inputs, weights)
        biased = backend.add(matmul_result, biases)
        return backend.maximum(biased, 0.0)

    return run_dense_forward


BENCHMARK_SCENARIOS = {
    "dense_forward_python": BenchmarkScenario(
        name="dense_forward_python",
        callable_name="PythonBackend dense-like forward pass",
        description=(
            "Python backend matmul followed by bias addition and ReLU-style maximum."
        ),
        argument_description=(
            f"inputs=({BATCH_SIZE}, {INPUT_SIZE}), "
            f"weights=({INPUT_SIZE}, {OUTPUT_SIZE}), "
            f"biases=({OUTPUT_SIZE},), deterministic generated float literals"
        ),
        make_callable=lambda: _make_dense_forward_callable(PythonBackend(seed=0)),
    ),
    "dense_forward_numpy": BenchmarkScenario(
        name="dense_forward_numpy",
        callable_name="NumpyBackend dense-like forward pass",
        description=(
            "NumPy backend matmul followed by bias addition and ReLU-style maximum."
        ),
        argument_description=(
            f"inputs=({BATCH_SIZE}, {INPUT_SIZE}), "
            f"weights=({INPUT_SIZE}, {OUTPUT_SIZE}), "
            f"biases=({OUTPUT_SIZE},), deterministic generated float literals"
        ),
        make_callable=lambda: _make_dense_forward_callable(NumpyBackend(seed=0)),
    ),
}
