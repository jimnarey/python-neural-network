import math
from typing import Optional
from array import array


class PythonTensor:

    @staticmethod
    def _validated_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        # TODO - consider whether these guards should be at the backend class
        # level (for all backends)
        if not shape:
            raise ValueError("shape must be non-empty")
        for dimension in shape:
            if dimension < 0:
                raise ValueError("shape dimensions must be non-negative")
        return shape

    @staticmethod
    def _default_data(shape: tuple[int, ...]) -> array:
        """
        Create a float-valued buffer with the correct size for shape.
        """
        return array("d", [0.0]) * math.prod(shape)

    @staticmethod
    def _validated_data(data: array) -> array:
        """
        Ensure that a caller-supplied buffer is float valued.
        """
        if data.typecode != "d":
            raise TypeError("data must be a float array")
        return data

    @staticmethod
    def _default_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate the strides for a given shape

        Works through the shape tuple. For each value of i in the shape
        tuple, i in the strides tuple is the product of the lengths of
        all dimensions to the right.

        The last value in a strides tuple is always 1. This calculation
        works because math.prod on a empty tuple always returns 1.
        """
        return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))

    @staticmethod
    def _validate_strides_arg(strides: tuple[int, ...], shape: tuple[int, ...]) -> None:
        """
        Validate a caller-supplied strides tuple.

        This catches two basic problems with a caller-supplied strides value.
        It doesn't confirm that the shape and strides, in combination, describe
        a valid tensor.
        """
        if len(strides) != len(shape):
            raise ValueError("strides must have the same length as shape")
        if any(stride < 0 for stride in strides):
            raise ValueError("strides values must be positive")

    @staticmethod
    def _validate_buffer_bounds(
        strides: tuple[int, ...], shape: tuple[int, ...], data: array, offset: int
    ) -> None:
        """
        Check that the data buffer is large enough for the tensor layout
        described by shape, strides, and offset.

        For non-empty tensors, the highest buffer index reachable via the
        layout must be strictly within the buffer (i.e. a valid access index).

        For empty tensors no element is ever accessed, so offset is never
        dereferenced. However, offset may still equal len(data) to support
        views that produce an empty tensor at the end of a parent buffer.

        Example: a (3, 4) tensor occupies buffer indices 0-11:

            row 0: [ 0,  1,  2,  3]
            row 1: [ 4,  5,  6,  7]
            row 2: [ 8,  9, 10, 11]

        The slice [3:3, :] selects zero rows starting at row 3, producing
        shape (0, 4) with offset 12 — one past the end of the buffer.
        The offset is never dereferenced because the tensor has no elements,
        but it records where row 3 would begin in the parent buffer.

        Allowing offset == len(data) means view construction (slicing,
        broadcasting, etc.) can follow a single uniform code path without
        special-casing empty results that land at the end of the buffer.
        """
        if math.prod(shape) == 0:
            if offset > len(data):
                raise ValueError("offset must be within one-past-end of data buffer")
            return
        max_index = offset + sum(
            (dim - 1) * stride for dim, stride in zip(shape, strides)
        )
        if max_index >= len(data):
            raise ValueError("data buffer is too small")

    @staticmethod
    def _validated_layout(
        strides: tuple[int, ...] | None,
        shape: tuple[int, ...],
        data: array,
        offset: int,
    ) -> tuple[tuple[int, ...], int]:
        """
        Validate offset and strides, returning (strides, offset).

        If strides is None, default C-contiguous (row-major) strides are
        used. Buffer bounds are always checked.
        """
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if strides is None:
            validated_strides = PythonTensor._default_strides(shape)
        else:
            PythonTensor._validate_strides_arg(strides, shape)
            validated_strides = strides
        PythonTensor._validate_buffer_bounds(validated_strides, shape, data, offset)
        return validated_strides, offset

    @staticmethod
    def _flat_index(
        indices: tuple[int, ...],
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
    ) -> int:
        """
        Return the index within the flat buffer for a tensor element given its
        indices, shape, strides and offset.
        """
        if len(indices) != len(shape):
            raise IndexError("wrong number of indices")
        flat_index = offset
        for index, dim, stride in zip(indices, shape, strides):
            if not -dim <= index < dim:
                raise IndexError("tensor index out of range")
            if index < 0:
                index += dim
            flat_index += index * stride
        return flat_index

    def __init__(
        self,
        shape: tuple[int, ...],
        data: Optional[array] = None,
        offset: int = 0,
        strides: Optional[tuple[int, ...]] = None,
        writable: bool = True,
    ) -> None:
        self.shape = PythonTensor._validated_shape(shape)
        if data is None:
            self.data = PythonTensor._default_data(self.shape)
        else:
            self.data = PythonTensor._validated_data(data)
        self.strides, self.offset = PythonTensor._validated_layout(
            strides, self.shape, self.data, offset
        )
        self.writable = writable

    def get_scalar(self, indices: tuple[int, ...]) -> float:
        flat_index = PythonTensor._flat_index(
            indices, self.shape, self.strides, self.offset
        )
        return self.data[flat_index]

    def set_scalar(self, indices: tuple[int, ...], value: float) -> None:
        if not self.writable:
            raise ValueError("tensor is not writable")
        flat_index = PythonTensor._flat_index(
            indices, self.shape, self.strides, self.offset
        )
        self.data[flat_index] = value
