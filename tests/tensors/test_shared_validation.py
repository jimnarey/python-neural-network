import unittest
from src.tensors.validation import parse_tensor_data


class TestParseTensorData(unittest.TestCase):

    def test_returns_rank_0_shape_and_data_in_list_when_data_is_float(self):
        cases = (1.0, 5.7, 100.6)
        for data in cases:
            with self.subTest():
                shape, values = parse_tensor_data(data)
                self.assertEqual(values, [data])
                self.assertIs(type(values[0]), float)
                self.assertEqual(shape, ())

    def test_returns_rank_0_shape_and_float_in_list_when_data_is_int(self):
        cases = (1, 5, 100)
        for data in cases:
            with self.subTest():
                shape, values = parse_tensor_data(data)
                self.assertEqual(values, [float(data)])
                self.assertIs(type(values[0]), float)
                self.assertEqual(shape, ())

    def test_raises_when_data_is_or_contains_invalid_type(self):
        cases = (
            "data",
            True,
            None,
            dict(),
            set(),
            bytes(),
            bytearray(),
            [1.0, "data"],
            [[1.0, True]],
            [[[1.0], [None]]],
        )
        for data in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "requires numeric values"):
                    parse_tensor_data(data)

    def test_returns_rank_1_shape_with_zero_length_dimension_and_no_values_when_data_is_empty_list_tuple(
        self,
    ):
        cases = (
            [],
            (),
        )
        for data in cases:
            with self.subTest():
                shape, values = parse_tensor_data(data)
                self.assertEqual(shape, (0,))
                self.assertEqual(values, [])

    def test_returns_expected_shape_and_no_values_when_data_has_nested_empty_sequences(
        self,
    ):
        cases = (
            ([[]], (1, 0)),
            ([[], []], (2, 0)),
            ([[[]], [[]]], (2, 1, 0)),
        )
        for data, expected_shape in cases:
            with self.subTest():
                shape, values = parse_tensor_data(data)
                self.assertEqual(shape, expected_shape)
                self.assertEqual(values, [])

    def test_raises_when_data_is_not_rectangular(self):
        cases = (
            [[1, 2], [3]],
            [[[1, 2], [3, 4]], [[5, 6]]],
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]]],
            ],
        )
        for data in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "requires rectangular input"):
                    parse_tensor_data(data)

    def test_raises_when_data_mixes_empty_and_non_empty_sequences_at_same_level(self):
        cases = (
            [[], [1.0]],
            [[[]], [[1.0]]],
            [[], [], [1.0]],
        )
        for data in cases:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "requires rectangular input"):
                    parse_tensor_data(data)

    def test_returns_expected_shape_and_flat_list_when_data_is_1D(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (5,))
        self.assertEqual(values, data)
        self.assertTrue(all(type(value) is float for value in values))

    def test_returns_expected_shape_and_flat_list_when_data_is_2D_and_rectangular(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (2, 3))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertTrue(all(type(value) is float for value in values))

    def test_returns_expected_shape_and_flat_list_when_data_is_3D_and_rectangular(self):
        data = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (2, 2, 2))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertTrue(all(type(value) is float for value in values))

    def test_returns_expected_shape_and_flat_list_when_data_is_4D_and_rectangular(self):
        data = [
            [
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            [
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        ]
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (2, 1, 2, 2))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertTrue(all(type(value) is float for value in values))

    def test_returns_expected_shape_and_flat_list_when_data_mixes_lists_and_tuples(
        self,
    ):
        """
        Test that we get valid return values when lists and tuples are mixed
        in the same input data.

        This is to pin down the behaviour. Mixing lists and tuples is not
        especially desirable but it isn't forbidden so we should test for it.
        """
        data = ([1.0, 2.0], (3.0, 4.0))
        shape, values = parse_tensor_data(data)
        self.assertEqual(shape, (2, 2))
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(all(type(value) is float for value in values))
