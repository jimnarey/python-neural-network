from typing import Any, Protocol, runtime_checkable

# A placeholder for specific tensor implementations, which will be added later
type Tensor = Any


@runtime_checkable
class TensorBackend(Protocol):
    def randn(self, shape: tuple[int, ...]) -> Tensor:
        # What are we doing about mean and SD here?
        """
        Generates a tensor with the given shape, filled with random numbers
        drawn from a standard normal distribution.

        Example:
        randn((2, 3)) -> [[-0.5, 1.2, 0.3],
                          [ 0.8, -1.1, 0.0]]
        """

    def zeros(self, shape: tuple[int, ...]) -> Tensor:
        """
        Creates a tensor with the given shape, filled with zeros.

        Example:
        zeros((2, 3)) -> [[0, 0, 0],
                          [0, 0, 0]]
        """

    def add(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Adds two tensors element-wise, or adds a scalar (float or int) to all elements of a tensor.

        Example:
        add([[1, 2], [3, 4]], 5) -> [[6, 7],
                                     [8, 9]]
        """

    def subtract(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Subtracts one tensor from another element-wise, or subtracts a scalar from all elements of a tensor.

        Example:
        subtract([[5, 6], [7, 8]], 3) -> [[2, 3],
                                          [4, 5]]
        """

    def multiply(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Multiplies two tensors element-wise, or multiplies all elements of a tensor by a scalar.

        Example:
        multiply([[1, 2], [3, 4]], 2) -> [[2, 4],
                                          [6, 8]]
        """

    def divide(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Divides one tensor by another element-wise, or divides all elements of a tensor by a scalar.

        Example:
        divide([[4, 6], [8, 10]], 2) -> [[2, 3],
                                         [4, 5]]
        """

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        # Matmul behaves the same as a numpy dot product on 2D arrays but not if one
        # is a 1D array so some care needs to be taken with the interface
        """
        Performs matrix multiplication between two tensors.
        The number of columns in the first tensor must match the number of rows in the second tensor.

        Example:
        matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) -> [[19, 22],
                                                       [43, 50]]
        """

    def maximum(self, a: Tensor, b: Tensor | float | int) -> Tensor:
        """
        Computes the element-wise maximum of two tensors, or the maximum between each element of a tensor and a scalar.

        Example:
        maximum([[1, 5], [3, 2]], 4) -> [[4, 5],
                                         [4, 4]]
        """

    def exp(self, x: Tensor) -> Tensor:
        """
        Computes the exponential (e^x) of each element in the tensor.

        Example:
        exp([[0, 1], [2, 3]]) -> [[1, 2.718],  # e^0 = 1, e^1 ≈ 2.718
                                  [7.389, 20.085]]  # e^2 ≈ 7.389, e^3 ≈ 20.085
        """

    def sum(self, x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """
        Computes the sum of all elements in the tensor, or along a specific axis.
        If `keepdims` is True, the reduced dimensions are kept with size 1.

        Example:
        sum([[1, 2], [3, 4]]) -> 10  # Total sum
        sum([[1, 2], [3, 4]], axis=0) -> [4, 6]  # Column-wise sum
        """

    def max(self, x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """
        Computes the maximum value of all elements in the tensor, or along a specific axis.
        If `keepdims` is True, the reduced dimensions are kept with size 1.

        Example:
        max([[1, 2], [3, 4]]) -> 4  # Maximum value
        max([[1, 2], [3, 4]], axis=1) -> [2, 4]  # Row-wise maximum
        """
