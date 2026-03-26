from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close


class BackendContractReshapeMixin(BackendContractBase):
    """
    This class provides good coverage of reshape's handling of 2D > 1D
    (and vice-versa) operations, with enough coverage of 3D and 4D inputs
    to ensure that implementations are sufficiently generalised and not
    written with 1D/2D tensors as special cases.

    When run against the expected tensor/values, assert_nested_close
    confims that reshape preserves both the element order and the values
    of the elements. There's no rounding of floats happening here so the
    tolerances (passed to math.is_close) are set to zero.

    We test explicitly for shape too.
    """

    def test_reshape_converts_1D_array_to_2D_array_with_shape_2_by_2(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0, 3.0, 4.0])
        result = backend.reshape(tensor, (2, 2))
        self.assertEqual(backend.shape(result), (2, 2))
        assert_nested_close(result, [[1.0, 2.0], [3.0, 4.0]], rel_tol=0, abs_tol=0)

    def test_reshape_converts_1D_array_to_2D_array_with_shape_2_by_3(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = backend.reshape(tensor, (2, 3))
        self.assertEqual(backend.shape(result), (2, 3))
        assert_nested_close(
            result,
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_reshape_converts_1D_array_to_3D_array_with_shape_2_by_2_by_2(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = backend.reshape(tensor, (2, 2, 2))
        self.assertEqual(backend.shape(result), (2, 2, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_reshape_converts_2D_array_to_1D_array_from_shape_2_by_2(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result = backend.reshape(tensor, (4,))
        self.assertEqual(backend.shape(result), (4,))
        assert_nested_close(result, [1.0, 2.0, 3.0, 4.0], rel_tol=0, abs_tol=0)

    def test_reshape_converts_2D_array_to_1D_array_from_shape_2_by_3(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = backend.reshape(tensor, (6,))
        self.assertEqual(backend.shape(result), (6,))
        assert_nested_close(
            result,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            rel_tol=0,
            abs_tol=0,
        )

    def test_reshape_converts_2D_array_to_3D_array_with_shape_2_by_1_by_3(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = backend.reshape(tensor, (2, 1, 3))
        self.assertEqual(backend.shape(result), (2, 1, 3))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_reshape_converts_3D_array_to_1D_array_from_shape_2_by_2_by_2(self):
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        result = backend.reshape(tensor, (8,))
        self.assertEqual(backend.shape(result), (8,))
        assert_nested_close(
            result,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            rel_tol=0,
            abs_tol=0,
        )

    def test_reshape_converts_3D_array_to_2D_array_from_shape_2_by_2_by_2(self):
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        result = backend.reshape(tensor, (4, 2))
        self.assertEqual(backend.shape(result), (4, 2))
        assert_nested_close(
            result,
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_reshape_converts_3D_array_to_4D_array_from_shape_2_by_2_by_2(self):
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        result = backend.reshape(tensor, (2, 2, 1, 2))
        self.assertEqual(backend.shape(result), (2, 2, 1, 2))
        assert_nested_close(
            result,
            [
                [[[1.0, 2.0]], [[3.0, 4.0]]],
                [[[5.0, 6.0]], [[7.0, 8.0]]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_reshape_converts_4D_array_to_2D_array_from_shape_2_by_2_by_1_by_2(self):
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[[1.0, 2.0]], [[3.0, 4.0]]],
                [[[5.0, 6.0]], [[7.0, 8.0]]],
            ]
        )
        result = backend.reshape(tensor, (2, 4))
        self.assertEqual(backend.shape(result), (2, 4))
        assert_nested_close(
            result,
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_reshape_raises_when_target_shape_changes_element_count(self):
        """
        A reshaped tensor must have the same number of elements as the
        input tensor. I.e. if there are 8 elements in the input then the
        values comprising the shape must, when multiplied together, result
        in 8.

        This test takes a range of input tensors and in each case tries
        to reshape them with a target shape which breaks this rule.
        """
        backend = self.make_backend()
        invalid_cases = [
            (
                backend.to_tensor([1.0, 2.0, 3.0, 4.0]),
                [
                    (3,),
                    (2, 3),
                ],
            ),
            (
                backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
                [
                    (3,),
                    (3, 2),
                ],
            ),
            (
                backend.to_tensor(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]
                ),
                [
                    (7,),
                    (2, 2, 3),
                    (3, 3),
                ],
            ),
        ]

        for tensor, invalid_shapes in invalid_cases:
            for shape in invalid_shapes:
                with self.subTest(
                    input_shape=backend.shape(tensor),
                    target_shape=shape,
                ):
                    with self.assertRaises(
                        ValueError,
                        msg=(
                            "reshape accepted a target shape with a different "
                            "number of elements when it should reject it"
                        ),
                    ):
                        backend.reshape(tensor, shape)

    def test_reshape_supports_zero_length_dimensions_when_called_with_an_empty_array(
        self,
    ):
        """
        This tests some subtle behaviour, related to shape, and the expected tensors
        returned (in particular) are not very intuitive.

        If any dimension in the target shape is 0, the reshaped array has 0
        elements, because the total number of elements in an array is the product
        of its dimensions. For example, (0, 1), (1, 0) and (2, 0, 3) all
        contain no scalar values.

        Because reshape must preserve the number of elements, such target shapes
        are only valid when the input array is empty.

        These cases are hard to follow because the Python nested-list model for
        representing tensors eliminates axes if they are preceded with a zero-value
        dimension. If the first target dimension is 0, the  result is simply [],
        whatever later dimensions may be. If an earlier dimension is non-zero and a
        later one is 0, the earlier dimensions are still visible, so (1, 0)
        becomes [[]] and (2, 0, 3) becomes [[], []].

        This does not make later dimensions meaningless. Shapes such as (0, 1)
        and (2, 0, 3) still differ from (0,) and (2,) in rank and shape,
        even though that difference can only be seen by checking the shape rather
        than the values.
        """
        # It is not completely clear, at this stage of the application's development,
        # what the real-world use of these types of tensor is in the context of a neural
        # network. As and when this becomes clear it would be helpful to add to this
        # docstring. In the meantime, it's expected that a reshape method can handle
        # them so we need to test for it
        backend = self.make_backend()
        valid_cases = [
            (
                backend.to_tensor([]),
                (0, 1),
                [],
            ),
            (
                backend.to_tensor([]),
                (0, 2, 3),
                [],
            ),
            (
                backend.to_tensor([]),
                (1, 0),
                [[]],
            ),
            (
                backend.to_tensor([]),
                (2, 0),
                [[], []],
            ),
            (
                backend.to_tensor([]),
                (2, 0, 3),
                [[], []],
            ),
        ]

        for tensor, shape, expected in valid_cases:
            with self.subTest(
                input_shape=backend.shape(tensor),
                target_shape=shape,
            ):
                result = backend.reshape(tensor, shape)
                self.assertEqual(backend.shape(result), shape)
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_reshape_rejects_zero_length_dimensions_when_called_with_a_non_empty_array(
        self,
    ):
        backend = self.make_backend()
        invalid_cases = [
            (
                backend.to_tensor([1.0, 2.0]),
                [
                    (0, 2),
                    (2, 0),
                ],
            ),
            (
                backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
                [
                    (0, 2, 2),
                    (2, 0),
                ],
            ),
        ]

        for tensor, invalid_shapes in invalid_cases:
            for shape in invalid_shapes:
                with self.subTest(
                    input_shape=backend.shape(tensor),
                    target_shape=shape,
                ):
                    with self.assertRaises(
                        ValueError,
                        msg=(
                            "reshape accepted a target shape containing a zero-length "
                            "dimension for a non-empty array when it should reject it"
                        ),
                    ):
                        backend.reshape(tensor, shape)

    def test_reshape_rejects_negative_values_in_target_shape(self):
        """
        The backend contract explicitly does not support the '-1' notation
        which is accepted natively by NumPy's reshape method (to infer the
        length of a single dimension). It is not needed and arguably confusing
        as a result of having nothing to do with other uses of negative values
        for dimensions which are supported and are analagous to Python's
        syntax for indexes of lists, tuples etc

        To make things simple the contract states reshape will not accept any
        negative values (even though NumPy only supports '-1') and so we test
        for several negative values here.
        """
        backend = self.make_backend()
        invalid_cases = [
            (
                backend.to_tensor([1.0, 2.0, 3.0, 4.0]),
                (2, -1),
            ),
            (
                backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
                (-3, 2),
            ),
            (
                backend.to_tensor(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]
                ),
                (2, -10),
            ),
        ]

        for tensor, shape in invalid_cases:
            with self.subTest(
                input_shape=backend.shape(tensor),
                target_shape=shape,
            ):
                with self.assertRaises(
                    ValueError,
                    msg=(
                        "reshape accepted a target shape containing a negative value "
                        "when it should reject it"
                    ),
                ):
                    backend.reshape(tensor, shape)
