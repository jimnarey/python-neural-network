"""Interface for tensor backends.

This allows us to swap out different backends without altering the code
for layers and the wider model. The NumPy backend is the reference version.

Protocol classes were chosen over ABCs because the type and behaviour of
the tensors used in each backend are too different. NumPy uses ndarray,
where the Python implementation uses nested lists. These are not
interchangeable. What matters is that the tensor implementation is always
hidden from the rest of the program; we stick to one backend for each
network; and each backend returns and expects its own tensor type.

The backend contract is intended to support tensors with an arbitrary
number of dimensions, where possible.

The price for the flexibility this approach provides is that we need
additional testing to pin down the interface's contract, since we can't
rely on static, nominal typing to do this for us.
"""

from typing import Any, Protocol, runtime_checkable

# A placeholder for specific tensor implementations, which will be added later
type Tensor = Any


@runtime_checkable
class TensorBackend(Protocol):
    def randn(self, shape: tuple[int, ...]) -> Tensor:
        """
        Generate a tensor with the given shape, filled with random numbers
        drawn from a standard normal distribution.

        Example:
        randn((2, 3)) -> [[-0.5, 1.2, 0.3],
                          [ 0.8, -1.1, 0.0]]
        """

    def zeros(self, shape: tuple[int, ...]) -> Tensor:
        """
        Create a tensor with the given shape, filled with zeros.

        Example:
        zeros((2, 3)) -> [[0, 0, 0],
                          [0, 0, 0]]
        """

    def shape(self, x: Tensor) -> tuple[int, ...]:
        """
        Return the shape of a tensor.

        Example:
        shape([[1, 2], [3, 4]]) -> (2, 2)
        """

    def reshape(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        """
        Return a tensor with the same values as ``x`` but with a new shape.

        Example:
        reshape([[1, 2], [3, 4]], (4,)) -> [1, 2, 3, 4]
        """

    def transpose(self, x: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
        """
        Permute tensor axes.

        If ``axes`` is ``None``, reverse the axes.
        """

    def add(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Add two tensors element-wise, or add a scalar to all elements of a
        tensor. Compatible shapes should follow the backend's broadcasting
        rules.

        Example:
        add([[1, 2], [3, 4]], 5) -> [[6, 7],
                                     [8, 9]]
        """

    def subtract(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Subtract one tensor from another element-wise, or subtract a scalar
        from all elements of a tensor.

        Example:
        subtract([[5, 6], [7, 8]], 3) -> [[2, 3],
                                          [4, 5]]
        """

    def multiply(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Multiply two tensors element-wise, or multiply all elements of a
        tensor by a scalar.

        Example:
        multiply([[1, 2], [3, 4]], 2) -> [[2, 4],
                                          [6, 8]]
        """

    def divide(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Divide one tensor by another element-wise, or divide all elements of
        a tensor by a scalar.

        Example:
        divide([[4, 6], [8, 10]], 2) -> [[2, 3],
                                         [4, 5]]
        """

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        # These docstrings were added early to ensure each of the tensor
        # operations were completely understood. This was the hardest
        # operation to understand by far. Specifically, the need to match
        # NumPy's 'trailing axis matmul' feature which needs to be replicated
        # across all implementations for consistency.
        # This is the only non-trivial docstring in the project which was
        # substantially written using an LLM (GPT-5.4). It has been improved
        # and simplified a lot but still needs more work
        """
        Perform matrix multiplication between two tensors.

        Matmul implementations must follow NumPy-style semantics
        in relation to trailing axes. This is expressed by the
        following notation:

        (..., m, k) @ (..., k, n) -> (..., m, n)

        - m is the number of rows in the left-hand matrix
        - k is the shared inner dimension
        - n is the number of columns in the right-hand matrix
        - ... means an arbitary number of dimensions

        The last two dimensions of each tensor are treated as the matrix
        dimensions. The last dimension of the left-hand tensor and the
        penultimate dimension of the right-hand tensor must match (k)

        The leading dimensions are not multiplied together as part of the
        matrix operation. Instead, matmul performs the same matrix multiplication
        for each combination of leading dimensions and preserves them in the result.

        For ordinary 2D matrices, the ... part is empty:

        - (m, k) @ (k, n) -> (m, n)

        For higher-dimensional tensors, only the trailing matrix dimensions
        are consumed by the multiplication. For example:

        - (batch, m, k) @ (k, n) -> (batch, m, n)
        - (batch, time, m, k) @ (k, n) -> (batch, time, m, n)

        This is important for layers such as dense layers. If an input has
        shape (..., num_inputs) and the weights have shape
        (num_inputs, num_neurons), the result has shape
        (..., num_neurons). That lets the same layer work with a simple
        batch of samples, or with inputs that have extra leading dimensions
        such as time steps.

        Example:
        matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) -> [[19, 22],
                                                       [43, 50]]
        """

    def maximum(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Compute the element-wise maximum of two tensors, or the maximum
        between each element of a tensor and a scalar.

        Example:
        maximum([[1, 5], [3, 2]], 4) -> [[4, 5],
                                         [4, 4]]
        """

    def exp(self, x: Tensor) -> Tensor:
        """
        Compute the exponential (e^x) of each element in the tensor.

        Example:
        exp([[0, 1], [2, 3]]) -> [[1, 2.718],  # e^0 = 1, e^1 ≈ 2.718
                                  [7.389, 20.085]]  # e^2 ≈ 7.389, e^3 ≈ 20.085
        """

    def sum(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """
        Compute the sum of all elements in the tensor, or along one or more
        specific axes.
        If `keepdims` is True, the reduced dimensions are kept with size 1.

        Example:
        sum([[1, 2], [3, 4]]) -> 10  # Total sum
        sum([[1, 2], [3, 4]], axis=0) -> [4, 6]  # Column-wise sum
        """

    def max(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """
        Compute the maximum value of all elements in the tensor, or along one
        or more specific axes.
        If `keepdims` is True, the reduced dimensions are kept with size 1.

        Example:
        max([[1, 2], [3, 4]]) -> 4  # Maximum value
        max([[1, 2], [3, 4]], axis=1) -> [2, 4]  # Row-wise maximum
        """
