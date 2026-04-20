from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close


class BackendContractTransposeMixin(BackendContractBase):
    """
    Test that transpose methods conform to the contract and describe
    the behaviour of the transpose reference implementation (NumPy).

    Note that when passing tuples as values for the transpose axes
    argument we are passing axes indices, which are zero-based. This
    can be confusing because the notation is similar to that for
    passing shapes to other methods, where the values in the tuples
    refer to the *sizes* of axes/dimensions.

    Where 'axes=None' is passed as an argument, this means transpose
    should reverse the axes (unless it's a 1D tensor). It's a NumPy
    convention we will implement in any other tensor backends.

    The axes argument in respect of 1D tensors can be a bit confusing.
    A transpose of a 1D tensor always returns the same 1D tensor. Valid
    values for axes are (0,), (-1,) or None, which all do the same
    thing.

    For other methods, the contract allows an integer in place of
    a singleton tuple. I.e. 1 in place of (1,). However, transpose
    always requires a tuple because the axes argument refers to the
    *complete new order* of the axes, rather than specifying which
    axis/axes to operate on.

    There is some duplication in the tests for reversing axes (2D,
    3D and 4D). In particular, we only need to demonstrate that
    transpose's default axes argument is None once. However, it
    was preferable to keep the three test methods consistent.

    Because rounding isn't an issue for these tests (we're not
    performing numerical operations on any values) the tolerances
    passed to assert_nested_close (from there, to math.is_close)
    are set to zero.
    """

    def test_transpose_returns_same_1D_tensor_when_axes_is_none(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0, 3.0])
        transposed_tensor = backend.transpose(tensor)
        result = backend.to_python(transposed_tensor)
        self.assertEqual(backend.shape(transposed_tensor), (3,))
        assert_nested_close(result, [1.0, 2.0, 3.0], rel_tol=0, abs_tol=0)

    def test_transpose_accepts_a_singleton_axes_tuple_for_a_1D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([1.0, 2.0, 3.0])
        transposed_tensor = backend.transpose(tensor, axes=(0,))
        result = backend.to_python(transposed_tensor)
        self.assertEqual(backend.shape(transposed_tensor), (3,))
        assert_nested_close(result, [1.0, 2.0, 3.0], rel_tol=0, abs_tol=0)

    def test_transpose_reverses_axes_of_2D_tensor(self):
        """
        Test that transpose reverses the axes of a 2D tensor when:
          - we pass the axes in reverse explicitly (1,0) as the axes argument
          - we pass None explicitly as the axes argument
          - we do not pass an axes argument (proving the default arg is None)

        All of these are equivalent. In the latter two cases the transpose
        reverses the axes automatically.

        The input tensor has shape (2, 3) so, when the axes are reversed, the
        new tensor has shape (3, 2).

        A value which was at position (row, column) in the input moves to
        position (column, row) in the result. So, for example, the value
        2.0 moves from position (0, 1) to position (1, 0).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected = [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]

        calls = [
            ("omitted", lambda: backend.transpose(tensor)),
            ("none", lambda: backend.transpose(tensor, axes=None)),
            ("explicit", lambda: backend.transpose(tensor, axes=(1, 0))),
        ]

        for mode, call in calls:
            with self.subTest(mode=mode):
                transposed_tensor = call()
                result = backend.to_python(transposed_tensor)
                self.assertEqual(backend.shape(transposed_tensor), (3, 2))
                assert_nested_close(
                    result,
                    expected,
                    rel_tol=0,
                    abs_tol=0,
                )

    def test_transpose_reverses_axes_of_3D_tensor(self):
        """
        Test that transpose reverses the axes of a 3D tensor when:
          - we pass the axes in reverse explicitly (2, 1, 0) as the axes argument
          - we pass None explicitly as the axes argument
          - we do not pass an axes argument (proving the default arg is None)

        Shape (2, 3, 4) becomes shape (4, 3, 2) in all cases.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                ],
            ]
        )
        expected = [
            [[1.0, 13.0], [5.0, 17.0], [9.0, 21.0]],
            [[2.0, 14.0], [6.0, 18.0], [10.0, 22.0]],
            [[3.0, 15.0], [7.0, 19.0], [11.0, 23.0]],
            [[4.0, 16.0], [8.0, 20.0], [12.0, 24.0]],
        ]

        calls = [
            ("omitted", lambda: backend.transpose(tensor)),
            ("none", lambda: backend.transpose(tensor, axes=None)),
            ("explicit", lambda: backend.transpose(tensor, axes=(2, 1, 0))),
        ]

        for mode, call in calls:
            with self.subTest(mode=mode):
                transposed_tensor = call()
                result = backend.to_python(transposed_tensor)
                self.assertEqual(backend.shape(transposed_tensor), (4, 3, 2))
                assert_nested_close(
                    result,
                    expected,
                    rel_tol=0,
                    abs_tol=0,
                )

    def test_transpose_reorders_axes_of_3D_tensor(self):
        """
        Tests that we can pass an arbitrary order of axes to transpose and it
        will re-order them accordingly in the output tensor.

        This is only possible with input tensors of 3 dimensions or upwards. The
        number of axes passed to the axes argument must be the same as the number
        of axes in the input tensor. With a 2D tensor there is only one possible,
        different permutation of axes we can ask for (reversal).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        transposed_tensor = backend.transpose(tensor, axes=(1, 2, 0))
        result = backend.to_python(transposed_tensor)
        self.assertEqual(backend.shape(transposed_tensor), (2, 3, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]],
                [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_transpose_returns_same_3D_tensor_when_passed_axes_in_existing_order(self):
        """
        Test that if we pass transpose a tensor and pass a tuple to the axes
        argument which has each of the axes in their existing order, we get the
        same tensor back.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        transposed_tensor = backend.transpose(tensor, axes=(0, 1, 2))
        result = backend.to_python(transposed_tensor)
        self.assertEqual(backend.shape(transposed_tensor), (2, 2, 3))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_transpose_reverses_axes_of_4D_tensor(self):
        """
        Test that transpose reverses the axes of a 4D tensor when:
          - we pass the axes in reverse explicitly (3, 2, 1, 0) as the axes argument
          - we pass None explicitly as the axes argument
          - we do not pass an axes argument (proving the default arg is None)

        Shape (2, 1, 2, 3) becomes shape (3, 2, 1, 2) in all cases.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
            ]
        )
        expected = [
            [
                [[1.0, 7.0]],
                [[4.0, 10.0]],
            ],
            [
                [[2.0, 8.0]],
                [[5.0, 11.0]],
            ],
            [
                [[3.0, 9.0]],
                [[6.0, 12.0]],
            ],
        ]

        calls = [
            ("omitted", lambda: backend.transpose(tensor)),
            ("none", lambda: backend.transpose(tensor, axes=None)),
            ("explicit", lambda: backend.transpose(tensor, axes=(3, 2, 1, 0))),
        ]

        for mode, call in calls:
            with self.subTest(mode=mode):
                transposed_tensor = call()
                result = backend.to_python(transposed_tensor)
                self.assertEqual(backend.shape(transposed_tensor), (3, 2, 1, 2))
                assert_nested_close(
                    result,
                    expected,
                    rel_tol=0,
                    abs_tol=0,
                )

    def test_transpose_reorders_axes_of_4D_tensor(self):
        """
        Tests that we can pass an arbitrary order of axes to transpose and it
        will re-order them accordingly in the output tensor.

        With 4D tensor there are more possible permutations of axes (24) than
        a 3D tensor (6). The number of permutations is n! for an nD tensor.

        We test three permutations of axes here, which is more than enough
        to demonstrate that transpose can re-order axes according to an arbitrary
        permutation of axes.

        In all cases we start with the same input tensor with shape (2, 2, 2, 3).
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
                [
                    [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                    [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
                ],
            ]
        )
        valid_cases = [
            (
                (1, 3, 0, 2),  # axes argument
                (2, 3, 2, 2),  # expected shape
                [
                    [
                        [[1.0, 4.0], [13.0, 16.0]],
                        [[2.0, 5.0], [14.0, 17.0]],
                        [[3.0, 6.0], [15.0, 18.0]],
                    ],
                    [
                        [[7.0, 10.0], [19.0, 22.0]],
                        [[8.0, 11.0], [20.0, 23.0]],
                        [[9.0, 12.0], [21.0, 24.0]],
                    ],
                ],
            ),
            (
                (2, 0, 3, 1),  # axes argument
                (2, 2, 3, 2),  # expected shape
                [
                    [
                        [[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]],
                        [[13.0, 19.0], [14.0, 20.0], [15.0, 21.0]],
                    ],
                    [
                        [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]],
                        [[16.0, 22.0], [17.0, 23.0], [18.0, 24.0]],
                    ],
                ],
            ),
            (
                (3, 0, 1, 2),  # axes argument
                (3, 2, 2, 2),  # expected shape
                [
                    [
                        [[1.0, 4.0], [7.0, 10.0]],
                        [[13.0, 16.0], [19.0, 22.0]],
                    ],
                    [
                        [[2.0, 5.0], [8.0, 11.0]],
                        [[14.0, 17.0], [20.0, 23.0]],
                    ],
                    [
                        [[3.0, 6.0], [9.0, 12.0]],
                        [[15.0, 18.0], [21.0, 24.0]],
                    ],
                ],
            ),
        ]

        for axes, expected_shape, expected in valid_cases:
            with self.subTest(axes=axes):
                transposed_tensor = backend.transpose(tensor, axes=axes)
                result = backend.to_python(transposed_tensor)
                self.assertEqual(backend.shape(transposed_tensor), expected_shape)
                assert_nested_close(
                    result,
                    expected,
                    rel_tol=0,
                    abs_tol=0,
                )

    def test_transpose_accepts_negative_axes_in_axes_tuple(self):
        """
        Test that transpose accepts negative values in the axes argument.
        As with Python lists, tuples etc negative values are used to
        count back from the end of the sequence.

        In addition to more general cases involving a 3D tensor this tests
        that the notation works with the specific case of a singleton
        tuple passed when transposing a 1D tensor. As with all valid
        calls to transpose when passing a 1D tensor we get the same
        1D tensor back, unchanged.
        """
        backend = self.make_backend()
        valid_cases = [
            (
                backend.to_tensor([1.0, 2.0, 3.0]),
                (-1,),
                (3,),
                [1.0, 2.0, 3.0],
            ),
            (
                backend.to_tensor(
                    [
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    ]
                ),
                (-2, -1, -3),
                (2, 3, 2),
                [
                    [[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]],
                    [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]],
                ],
            ),
        ]

        for tensor, axes, expected_shape, expected in valid_cases:
            with self.subTest(input_shape=backend.shape(tensor), axes=axes):
                transposed_tensor = backend.transpose(tensor, axes=axes)
                result = backend.to_python(transposed_tensor)
                self.assertEqual(backend.shape(transposed_tensor), expected_shape)
                assert_nested_close(
                    result,
                    expected,
                    rel_tol=0,
                    abs_tol=0,
                )

    def test_transpose_handles_zero_length_dimensions_when_axes_are_reversed(self):
        """
        Test that transpose can reverse the axes of an input tensor (this
        time by passing no axes argument) when one of the dimensions has
        zero length.

        Such a tensor will always have no values as the number of values
        is the product of the length of each of the dimensions. But it
        still has a structure and transpose should handle it.
        """
        backend = self.make_backend()
        tensor = backend.zeros((2, 0, 3))
        transposed_tensor = backend.transpose(tensor)
        result = backend.to_python(transposed_tensor)
        self.assertEqual(backend.shape(transposed_tensor), (3, 0, 2))
        assert_nested_close(
            result,
            [[], [], []],
            rel_tol=0,
            abs_tol=0,
        )

    def test_transpose_handles_zero_length_dimensions_when_axes_are_reordered(self):
        """
        Test that transpose can handle zero-length dimensions when passing an
        arbitrary permutation of axes as the axes argument.
        """
        backend = self.make_backend()
        valid_cases = [
            (
                backend.zeros((2, 0, 3)),
                (1, 2, 0),
                (0, 3, 2),
                [],
            ),
            (
                backend.zeros((0, 2, 3)),
                (1, 2, 0),
                (2, 3, 0),
                [[[], [], []], [[], [], []]],
            ),
            (
                backend.zeros((2, 3, 0)),
                (2, 0, 1),
                (0, 2, 3),
                [],
            ),
        ]

        for tensor, axes, expected_shape, expected in valid_cases:
            with self.subTest(
                input_shape=backend.shape(tensor),
                axes=axes,
            ):
                transposed_tensor = backend.transpose(tensor, axes=axes)
                result = backend.to_python(transposed_tensor)
                self.assertEqual(backend.shape(transposed_tensor), expected_shape)
                assert_nested_close(
                    result,
                    expected,
                    rel_tol=0,
                    abs_tol=0,
                )

    def test_transpose_rejects_duplicate_axes_in_axes_tuple(self):
        """
        The axes argument passed to transpose must be a permutation of the
        existing axes. I.e. each of the existing axes must appear once and
        only once in the tuple.

        This test passes several permutations as the axes argument, each of
        which includes a duplicate. In each case the *number* of axes is correct
        meaning that one of the axes does not appear in the tuple.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        invalid_axes = [
            (0, 0, 1),
            (2, 1, 1),
            (1, -2, 0),
        ]

        for axes in invalid_axes:
            with self.subTest(axes=axes):
                with self.assertRaises(
                    ValueError,
                    msg=(
                        "transpose accepted an axes tuple containing duplicate "
                        "axes when it should reject it"
                    ),
                ):
                    backend.transpose(tensor, axes=axes)

    def test_transpose_rejects_axes_outside_the_valid_range(self):
        """
        Test that if we pass a value for one of the axes to the axes argument
        which is less than or greater than the range of indices for the axes
        of the input tensor, we get an exception.

        Note that in some cases we are using negative values for axes indices
        in this test. In each case, counting back from the last index, the value
        results in an index of less than zero.
        """
        backend = self.make_backend()
        invalid_cases = [
            (
                backend.to_tensor([1.0, 2.0, 3.0]),
                [
                    (1,),
                    (-2,),
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
                    (0, 1, 3),
                    (0, 1, -4),
                ],
            ),
        ]

        for tensor, invalid_axes in invalid_cases:
            for axes in invalid_axes:
                with self.subTest(input_shape=backend.shape(tensor), axes=axes):
                    with self.assertRaises(
                        ValueError,
                        msg=(
                            "transpose accepted an axes tuple containing an "
                            "out-of-range axis when it should reject it"
                        ),
                    ):
                        backend.transpose(tensor, axes=axes)

    def test_transpose_rejects_axes_tuple_which_is_shorter_than_tensor_rank(self):
        """
        Tests that transpose rejects an axes tuple if it does not include all
        axes of the input tensor (even if there is nothing wrong with the values
        it does provide, i.e. they are in range and not duplicates).
        """
        backend = self.make_backend()
        invalid_cases = [
            (
                backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
                (1,),
            ),
            (
                backend.to_tensor(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]
                ),
                (2, 1),
            ),
        ]

        for tensor, axes in invalid_cases:
            with self.subTest(input_shape=backend.shape(tensor), axes=axes):
                with self.assertRaises(
                    ValueError,
                    msg=(
                        "transpose accepted an axes tuple shorter than the "
                        "tensor rank when it should reject it"
                    ),
                ):
                    backend.transpose(tensor, axes=axes)

    def test_transpose_rejects_axes_tuple_which_is_longer_than_tensor_rank(self):
        """
        Tests that transpose rejects an axes tuple if it has too many axes.
        """
        backend = self.make_backend()
        invalid_cases = [
            (
                backend.to_tensor([1.0, 2.0, 3.0]),
                (0, 1),
            ),
            (
                backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
                (1, 0, 2),
            ),
        ]

        for tensor, axes in invalid_cases:
            with self.subTest(input_shape=backend.shape(tensor), axes=axes):
                with self.assertRaises(
                    ValueError,
                    msg=(
                        "transpose accepted an axes tuple longer than the "
                        "tensor rank when it should reject it"
                    ),
                ):
                    backend.transpose(tensor, axes=axes)
