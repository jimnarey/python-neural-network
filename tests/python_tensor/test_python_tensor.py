import unittest
from array import array
from src.tensors.python_backend.tensor import PythonTensor


class TestValidatedShape(unittest.TestCase):

    def test_returns_valid_shapes_unchanged(self):
        shapes = (
            (1,),
            (0, 1),
            (1, 2),
            (1, 0, 3),
            (1000, 1001),
            (1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
        for shape in shapes:
            with self.subTest():
                result = PythonTensor._validated_shape(shape)
                self.assertEqual(shape, result)

    def test_raises_on_empty_shape(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            PythonTensor._validated_shape(())

    def test_raises_on_negative_dimension(self):
        shapes = ((-1,), (2, -20))
        for shape in shapes:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    PythonTensor._validated_shape(shape)


class TestDefaultData(unittest.TestCase):

    def test_returns_correct_sized_array_for_given_shape_when_data_is_none(self):
        shapes_data_sizes = (
            ((1,), 1),
            ((2, 3), 6),
            ((33, 4, 5), 660),
            ((1000, 21, 45, 8), 7560000),
        )
        for shape, data_size in shapes_data_sizes:
            with self.subTest():
                result = PythonTensor._default_data(shape)
                self.assertEqual(data_size, len(result))

    def test_returns_float_valued_array(self):
        result = PythonTensor._default_data((1, 2))
        self.assertEqual(result.typecode, "d")


class TestValidatedData(unittest.TestCase):

    def test_returns_same_array_if_float_valued(self):
        input = array("d", [0.0] * 12)
        result = PythonTensor._validated_data(input)
        self.assertIs(input, result)

    def test_raises_if_passed_non_float_valued_array(self):
        invalid_typecodes = "bBuwhHiIlLqQf"
        for code in invalid_typecodes:
            with self.subTest():
                with self.assertRaisesRegex(TypeError, "float array"):
                    PythonTensor._validated_data(array(code))


class TestDefaultStrides(unittest.TestCase):

    def test_default_strides_returned_based_on_shape(self):
        shapes_strides = (
            ((1,), (1,)),
            ((2, 3), (3, 1)),
            ((6, 4, 8), (32, 8, 1)),
            ((1, 2, 9, 1), (18, 9, 1, 1)),
        )
        for shape, strides in shapes_strides:
            with self.subTest():
                result = PythonTensor._default_strides(shape)
                self.assertEqual(strides, result)


class TestValidateStridesArg(unittest.TestCase):

    def test_returns_none_if_strides_and_shape_same_length_and_strides_values_not_negative(
        self,
    ):
        strides_shapes = (
            ((0,), (2,)),
            ((8, 100), (4, 8)),
            ((0, 1, 2), (1, 2, 3)),
        )
        for strides, shape in strides_shapes:
            with self.subTest():
                self.assertIsNone(PythonTensor._validate_strides_arg(strides, shape))

    def test_raises_if_strides_and_shape_have_different_lengths(self):
        strides_shapes = (
            ((0, 1), (2,)),
            ((8,), (4, 8)),
            ((0, 1, 2, 3, 4, 5), (1, 2)),
            ((1, 2), (1, 2, 3, 4, 5, 6)),
        )
        for strides, shape in strides_shapes:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "same length"):
                    PythonTensor._validate_strides_arg(strides, shape)

    def test_raises_if_any_stride_value_is_negative(self):
        strides_shapes = (
            ((4, -8), (2, 2)),
            ((-1, 2), (2, 2)),
            ((2, 4, 6, -8), (2, 2, 2, 2)),
        )
        for strides, shape in strides_shapes:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "be positive"):
                    PythonTensor._validate_strides_arg(strides, shape)


class TestValidateBufferBounds(unittest.TestCase):

    def test_raises_if_offset_not_within_buffer_one_past_end_when_passed_empty_tensor(
        self,
    ):
        """
        For an empty tensor, no element is ever accessed. The offset is allowed
        to equal len(data) as a one-past-end position.

        So in the invalid cases here, the tensor shape contains a zero-length
        dimension and therefore accesses no values, but the offset lies further
        than one past the end of the buffer and must be rejected.
        """
        datas_offsets = (
            (array("d", [0.0]), 2),
            (array("d", [0.0] * 10), 11),
            (array("d", [0.0] * 10), 100),
        )
        for data, offset in datas_offsets:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "within one-past-end"):
                    PythonTensor._validate_buffer_bounds(
                        (2, 4), (2, 0, 2), data, offset
                    )

    def test_returns_none_if_offset_within_buffer_one_past_end_when_passed_empty_tensor(
        self,
    ):
        datas_offsets = (
            (array("d", []), 0),
            (array("d", [0.0]), 1),
            (array("d", [0.0] * 10), 9),
            (array("d", [0.0] * 10), 10),
            (array("d", [0.0] * 100), 99),
            (array("d", [0.0] * 100), 100),
        )
        for data, offset in datas_offsets:
            with self.subTest():
                self.assertIsNone(
                    PythonTensor._validate_buffer_bounds(
                        (2, 4), (2, 0, 2), data, offset
                    )
                )

    def test_raises_if_max_index_out_of_bounds_when_passed_non_empty_tensor(self):
        """
        For a non-empty tensor, the largest buffer index reachable via the shape,
        strides and offset must still lie within the data buffer.

        In the first case, shape (2, 3) with strides (3, 1) and offset 1 reaches
        a maximum index of 1 + (2 - 1) * 3 + (3 - 1) * 1 = 6. But a buffer of
        length 6 has valid indices only up to 5, so this layout is out of bounds
        and must be rejected.
        """
        invalid_cases = (
            ((3, 1), (2, 3), array("d", [0.0] * 6), 1),
            ((4, 1), (2, 3), array("d", [0.0] * 6), 0),
            ((12, 6, 2, 1), (2, 2, 2, 3), array("d", [0.0] * 24), 2),
        )
        for strides, shape, data, offset in invalid_cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "buffer is too small"):
                    PythonTensor._validate_buffer_bounds(strides, shape, data, offset)

    def test_returns_none_if_max_index_in_bounds_when_passed_non_empty_tensor(self):
        cases = (
            ((3, 1), (2, 3), array("d", [0.0] * 7), 1),
            ((4, 1), (2, 3), array("d", [0.0] * 7), 0),
            ((12, 6, 2, 1), (2, 2, 2, 3), array("d", [0.0] * 40), 2),
        )
        for strides, shape, data, offset in cases:
            with self.subTest():
                self.assertIsNone(
                    PythonTensor._validate_buffer_bounds(strides, shape, data, offset)
                )


class TestValidatedLayout(unittest.TestCase):

    def test_raises_if_offset_is_negative(self):
        with self.assertRaisesRegex(ValueError, "offset must be >= 0"):
            PythonTensor._validated_layout(None, (2, 3), array("d", [0.0] * 6), -1)

    def test_returns_default_strides_and_offset_if_strides_arg_is_none_and_layout_valid(
        self,
    ):
        """
        A valid layout here means a non-negative offset and a buffer large
        enough for the shape. The default strides are the standard strides
        implied by the shape: for shape (2, 3), (3, 1) means move forward
        3 buffer positions to go from one row to the next and 1 position to
        move from one value to the next within a row.
        """
        result = PythonTensor._validated_layout(None, (2, 3), array("d", [0.0] * 6), 0)
        self.assertEqual(result, ((3, 1), 0))

    def test_returns_caller_supplied_strides_and_offset_if_layout_valid(self):
        result = PythonTensor._validated_layout(
            (4, 1), (2, 3), array("d", [0.0] * 7), 0
        )
        self.assertEqual(result, ((4, 1), 0))

    def test_raises_if_caller_supplied_strides_invalid(self):
        """
        See the tests for _valdiate_strides_arg for what constitutes
        an invalid strides arg. This is just a sample case (the tuple
        is too long).
        """
        with self.assertRaisesRegex(ValueError, "same length"):
            PythonTensor._validated_layout((3,), (2, 3), array("d", [0.0] * 6), 0)

    def test_raises_if_buffer_bounds_invalid(self):
        with self.assertRaisesRegex(ValueError, "buffer is too small"):
            PythonTensor._validated_layout((4, 1), (2, 3), array("d", [0.0] * 6), 0)

    def test_returns_valid_layout_for_empty_tensor_with_offset_one_past_end(self):
        result = PythonTensor._validated_layout(
            None, (2, 0, 2), array("d", [0.0] * 8), 8
        )
        self.assertEqual(result, ((0, 2, 1), 8))


class TestFlatIndex(unittest.TestCase):

    def test_returns_expected_index_for_1D_tensor(self):
        """
        For a 1D tensor, the flat index is just offset + index * stride.

        The arguments are indices (2,), shape (4,), strides (1,) and
        offset 0, so the result is 0 + 2 * 1 = 2.
        """
        result = PythonTensor._flat_index((2,), (4,), (1,), 0)
        self.assertEqual(result, 2)

    def test_returns_expected_index_for_2D_tensor(self):
        """
        For a 2D tensor, the flat buffer position is found by starting from
        the offset and then adding index * stride for each of the two axes.

        The arguments are indices (1, 2), shape (2, 3), strides (3, 1)
        and offset 0, so the result is 0 + 1 * 3 + 2 * 1 = 5.
        """
        result = PythonTensor._flat_index((1, 2), (2, 3), (3, 1), 0)
        self.assertEqual(result, 5)

    def test_returns_expected_index_for_3D_tensor(self):
        """
        For a 3D tensor, the flat buffer position is found by starting from
        the offset and then adding index * stride for each of the three axes.

        The arguments are indices (1, 2, 3), shape (2, 3, 4), strides
        (12, 4, 1) and offset 0, so the result is
        0 + 1 * 12 + 2 * 4 + 3 * 1 = 23.
        """
        result = PythonTensor._flat_index((1, 2, 3), (2, 3, 4), (12, 4, 1), 0)
        self.assertEqual(result, 23)

    def test_returns_expected_index_when_offset_is_non_zero(self):
        """
        Offset shifts the whole tensor view forward within the underlying
        buffer.

        The same indices and strides as the previous test but with
        offset 5, the result is 5 + 1 * 12 + 2 * 4 + 3 * 1 = 28.
        """
        result = PythonTensor._flat_index((1, 2, 3), (2, 3, 4), (12, 4, 1), 5)
        self.assertEqual(result, 28)

    def test_accepts_negative_indices_within_bounds(self):
        cases = (
            ((-1,), (4,), (1,), 0, 3),
            ((-1, -1), (2, 3), (3, 1), 0, 5),
            ((-2, -3), (2, 3), (3, 1), 0, 0),
        )
        for indices, shape, strides, offset, expected in cases:
            with self.subTest():
                result = PythonTensor._flat_index(indices, shape, strides, offset)
                self.assertEqual(result, expected)

    def test_raises_if_wrong_number_of_indices_passed(self):
        cases = (
            ((1,), (2, 3), (3, 1), 0),
            ((1, 2, 3), (2, 3), (3, 1), 0),
        )
        for indices, shape, strides, offset in cases:
            with self.subTest():
                with self.assertRaisesRegex(IndexError, "wrong number of indices"):
                    PythonTensor._flat_index(indices, shape, strides, offset)

    def test_raises_if_index_out_of_range(self):
        cases = (
            ((4,), (4,), (1,), 0),
            ((-5,), (4,), (1,), 0),
            ((2, 0), (2, 3), (3, 1), 0),
            ((0, 3), (2, 3), (3, 1), 0),
        )
        for indices, shape, strides, offset in cases:
            with self.subTest():
                with self.assertRaisesRegex(IndexError, "tensor index out of range"):
                    PythonTensor._flat_index(indices, shape, strides, offset)


class TestGetScalar(unittest.TestCase):
    """
    A thin test class, given that this is just a thin wrapper
    around _flat_index.
    """

    def test_returns_value_at_indices(self):
        tensor = PythonTensor((2, 3), array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        result = tensor.get_scalar((1, 2))
        self.assertEqual(result, 6.0)

    def test_uses_tensor_layout_when_reading_value(self):
        tensor = PythonTensor(
            (2, 3),
            array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            offset=1,
            strides=(3, 1),
        )
        result = tensor.get_scalar((1, 2))
        self.assertEqual(result, 7.0)


class TestSetScalar(unittest.TestCase):
    """
    Also a thin test class but in this case we also need to
    ensure that non-writable behaviour is respected.
    """

    def test_updates_value_at_indices(self):
        data = array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        tensor = PythonTensor((2, 3), data)
        tensor.set_scalar((1, 1), 100.0)
        self.assertEqual(data[4], 100.0)

    def test_uses_tensor_layout_when_writing_value(self):
        data = array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        tensor = PythonTensor((2, 3), data, offset=1, strides=(3, 1))
        tensor.set_scalar((0, 2), 100.0)
        self.assertEqual(data[3], 100.0)

    def test_raises_if_tensor_not_writable(self):
        tensor = PythonTensor(
            (2, 3),
            array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            writable=False,
        )
        with self.assertRaisesRegex(ValueError, "tensor is not writable"):
            tensor.set_scalar((1, 2), 100.0)
