from __future__ import annotations
import math
from typing import Iterator, Optional
from array import array
from itertools import product

from src.tensors.tensor_backend import Scalar


class PythonTensor:
    FLOAT = "d"
    INT = "q"
    SUPPORTED_TYPECODES = {FLOAT, INT}

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
    def _default_data(
        shape: tuple[int, ...],
        typecode: str = FLOAT,
    ) -> array:
        """
        Create a buffer with the correct size and type for shape.
        """
        if typecode == PythonTensor.FLOAT:
            return array(typecode, [0.0]) * math.prod(shape)
        if typecode == PythonTensor.INT:
            return array(typecode, [0]) * math.prod(shape)
        raise TypeError("data must be a supported tensor array")

    @staticmethod
    def _validated_data(data: array) -> array:
        """
        Ensure that a caller-supplied buffer uses a supported type.
        """
        if data.typecode not in PythonTensor.SUPPORTED_TYPECODES:
            raise TypeError("data must be a supported tensor array")
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
            raise ValueError("strides values must be non-negative")

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
        typecode: str = FLOAT,
    ) -> None:
        self.shape = PythonTensor._validated_shape(shape)
        if data is None:
            self.data = PythonTensor._default_data(self.shape, typecode)
        else:
            self.data = PythonTensor._validated_data(data)
        self.strides, self.offset = PythonTensor._validated_layout(
            strides, self.shape, self.data, offset
        )
        self.writable = writable

    def get_scalar(self, indices: tuple[int, ...]) -> Scalar:
        flat_index = PythonTensor._flat_index(
            indices, self.shape, self.strides, self.offset
        )
        return self.data[flat_index]

    def set_scalar(self, indices: tuple[int, ...], value: Scalar) -> None:
        if not self.writable:
            raise ValueError("tensor is not writable")
        if type(value) is bool:
            raise TypeError("tensor values must not be bool")
        flat_index = PythonTensor._flat_index(
            indices, self.shape, self.strides, self.offset
        )
        self.data[flat_index] = value

    def ndim(self) -> int:
        return len(self.shape)

    def size(self) -> int:
        return math.prod(self.shape)

    def indices(self) -> Iterator[tuple[int, ...]]:
        return product(*(range(dim) for dim in self.shape))

    # TODO - For tensors with default strides and offset zero we can probably
    # just bind .items to something which iterates through the buffer for a
    # possible performance improvement.
    def items(self) -> Iterator[tuple[tuple[int, ...], Scalar]]:
        for indices in self.indices():
            yield indices, self.get_scalar(indices)

    def to_list(self) -> list:
        def build(indices):
            if len(indices) == len(self.shape):
                return self.get_scalar(indices)
            dim = self.shape[len(indices)]
            return [build(indices + (i,)) for i in range(dim)]

        return build(())

    def view(
        self,
        shape: tuple[int, ...],
        offset: Optional[int] = None,
        strides: Optional[tuple[int, ...]] = None,
        writable: Optional[bool] = None,
    ) -> PythonTensor:
        """
        Create a 'cheap' tensor using the same data buffer as the instance
        on which the method is called.

        The offset chooses where the view starts in that buffer, and the
        strides describe how to move through the buffer for each axis. Together
        they let us represent slices, transposes and other layout changes
        without copying the underlying values.
        """
        return PythonTensor(
            shape,
            self.data,
            self.offset if offset is None else offset,
            strides,
            self.writable if writable is None else writable,
        )

    def copy(self, writable: bool = True) -> PythonTensor:
        copied_data = array(
            self.data.typecode,
            (self.get_scalar(indices) for indices in self.indices()),
        )
        return PythonTensor(self.shape, copied_data, writable=writable)
