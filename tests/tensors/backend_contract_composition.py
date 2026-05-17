"""
Tests for the two composition operations: concatenate and stack.

Both operations combine a sequence of tensors into a single tensor along
a specified axis. They differ in what that axis must look like and what
the result shape will be.

concatenate joins tensors along an existing axis, extending it. All input
tensors must already have the same number of dimensions, and their shapes
must match on every axis except the one being joined. The result has the
same rank as the inputs, with the chosen axis extended by the sum of its
sizes across all inputs. concatenate is used to extend an existing
dimension, for example, appending rows to a dataset.

stack introduces a new axis and places each input tensor at one position
along it. All input tensors must have identical shapes. The result has one
more dimension than the inputs, with size equal to the number of inputs
along the new axis. stack is used when to group tensors together  without
merging their contents — for example, collecting per-sample gradient
tensors into a single batch tensor.

The tests in this module cover the shared contract that every backend must
satisfy for both operations: the ordinary result shapes and values across
1D, 2D, and 3D inputs; negative axis indexing; inputs of more than two
tensors; singleton input sequences; and the rejection of invalid inputs
including empty sequences, out-of-range axes, rank mismatches, and shape
mismatches.
"""

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures
from tests.helpers.tensor_helpers import assert_nested_close


@EnforceSharedNumericFixtures()
class BackendContractConcatenateSemanticsMixin(BackendContractBase):
    def test_concatenate_joins_1D_tensors_along_axis_0(self):
        """
        Tests that concatenate joins two 1D tensors end-to-end along axis 0.

        Concatenating along axis 0 of a 1D tensor means extending the single
        sequence by appending all elements of each subsequent tensor in order.
        The two input tensors are:

        [1.0, 2.0]
        [3.0, 4.0, 5.0]

        The result is [1.0, 2.0, 3.0, 4.0, 5.0] with shape (5,).
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([1.0, 2.0]),
            backend.to_tensor([3.0, 4.0, 5.0]),
        ]
        result_tensor = backend.concatenate(tensors, axis=0)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (5,))
        assert_nested_close(result, [1.0, 2.0, 3.0, 4.0, 5.0], rel_tol=0, abs_tol=0)

    def test_concatenate_joins_2D_tensors_along_axis_0(self):
        """
        Tests that concatenate joins two 2D tensors by adding rows along axis 0.

        Concatenating along axis 0 of a 2D tensor means stacking the input
        tensors vertically: the rows of each subsequent tensor are appended
        below the rows of the previous one. The number of columns must match.

        The two input tensors are:

        [1.0, 2.0]    (shape (2, 2))
        [3.0, 4.0]

        [5.0, 6.0]    (shape (1, 2))

        The result is:

        [1.0, 2.0]
        [3.0, 4.0]    (shape (3, 2))
        [5.0, 6.0]
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0]]),
        ]
        result_tensor = backend.concatenate(tensors, axis=0)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (3, 2))
        assert_nested_close(
            result,
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_concatenate_joins_2D_tensors_along_axis_1(self):
        """
        Tests that concatenate joins two 2D tensors by adding columns along axis 1.

        Concatenating along axis 1 of a 2D tensor means placing the input
        tensors side-by-side: the columns of each subsequent tensor are
        appended to the right of the columns of the previous one. The number
        of rows must match.

        The two input tensors are:

        [1.0, 2.0]    (shape (2, 2))
        [3.0, 4.0]

        [5.0]         (shape (2, 1))
        [6.0]

        The result is:

        [1.0, 2.0, 5.0]
        [3.0, 4.0, 6.0]    (shape (2, 3))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0], [6.0]]),
        ]
        result_tensor = backend.concatenate(tensors, axis=1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 3))
        assert_nested_close(
            result,
            [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_concatenate_joins_3D_tensors_along_axis_1(self):
        """
        Tests that concatenate joins two 3D tensors by extending axis 1.

        Concatenating along axis 1 of a 3D tensor means extending the middle
        axis while preserving the outer and inner dimensions. The sizes of
        axes 0 and 2 must match.

        The two input tensors are:

        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]    (shape (2, 2, 2))

        [
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
            [[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]]
        ]    (shape (2, 3, 2))

        The result is:

        [
            [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
            [[5.0, 6.0], [7.0, 8.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]]
        ]    (shape (2, 5, 2))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
            backend.to_tensor(
                [
                    [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
                    [[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
                ]
            ),
        ]
        result_tensor = backend.concatenate(tensors, axis=1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 5, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
                [[5.0, 6.0], [7.0, 8.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_concatenate_joins_3D_tensors_along_axis_2(self):
        """
        Tests that concatenate joins two 3D tensors by extending axis 2.

        Concatenating along axis 2 of a 3D tensor means extending the inner
        axis while preserving the outer and middle dimensions. The sizes of
        axes 0 and 1 must match.

        The two input tensors are:

        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]    (shape (2, 2, 2))

        [
            [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
            [[15.0, 16.0, 17.0], [18.0, 19.0, 20.0]]
        ]    (shape (2, 2, 3))

        The result is:

        [
            [[1.0, 2.0, 9.0, 10.0, 11.0], [3.0, 4.0, 12.0, 13.0, 14.0]],
            [[5.0, 6.0, 15.0, 16.0, 17.0], [7.0, 8.0, 18.0, 19.0, 20.0]]
        ]    (shape (2, 2, 5))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
            backend.to_tensor(
                [
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                    [[15.0, 16.0, 17.0], [18.0, 19.0, 20.0]],
                ]
            ),
        ]
        result_tensor = backend.concatenate(tensors, axis=2)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 2, 5))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0, 9.0, 10.0, 11.0], [3.0, 4.0, 12.0, 13.0, 14.0]],
                [[5.0, 6.0, 15.0, 16.0, 17.0], [7.0, 8.0, 18.0, 19.0, 20.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_concatenate_accepts_negative_axis(self):
        """
        Tests that negative values for the axis argument are treated in
        the same way as negative values when passed as an index to
        Python's list class etc
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
            backend.to_tensor(
                [
                    [[9.0], [10.0]],
                    [[11.0], [12.0]],
                ]
            ),
        ]
        result_tensor = backend.concatenate(tensors, axis=-1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 2, 3))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0, 9.0], [3.0, 4.0, 10.0]],
                [[5.0, 6.0, 11.0], [7.0, 8.0, 12.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_concatenate_joins_more_than_two_2D_tensors(self):
        """
        Tests that concatenate can join more than two tensors along axis 0.

        Concatenating along axis 0 of a 2D tensor means appending the rows of
        each subsequent tensor below those of the previous one. This test uses
        three input tensors to ensure the method is not limited to combining
        exactly two tensors.

        The three input tensors are:

        [1.0, 2.0]    (shape (2, 2))
        [3.0, 4.0]

        [5.0, 6.0]    (shape (1, 2))

        [7.0, 8.0]    (shape (2, 2))
        [9.0, 10.0]

        The result is:

        [1.0, 2.0]
        [3.0, 4.0]
        [5.0, 6.0]
        [7.0, 8.0]    (shape (5, 2))
        [9.0, 10.0]
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0]]),
            backend.to_tensor([[7.0, 8.0], [9.0, 10.0]]),
        ]
        result_tensor = backend.concatenate(tensors, axis=0)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (5, 2))
        assert_nested_close(
            result,
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_concatenate_joins_more_than_two_3D_tensors_along_axis_1(self):
        """
        Tests that concatenate can join more than two 3D tensors along axis 1.

        Concatenating along axis 1 of a 3D tensor means appending the slices
        on the middle axis of each subsequent tensor after those of the
        previous one. This test uses three input tensors to ensure the method
        is not limited to combining exactly two 3D tensors.

        The three input tensors are:

        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]    (shape (2, 2, 2))

        [
            [[9.0, 10.0]],
            [[11.0, 12.0]]
        ]    (shape (2, 1, 2))

        [
            [[13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0]]
        ]    (shape (2, 2, 2))

        The result is:

        [
            [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0], [13.0, 14.0], [15.0, 16.0]],
            [[5.0, 6.0], [7.0, 8.0], [11.0, 12.0], [17.0, 18.0], [19.0, 20.0]]
        ]    (shape (2, 5, 2))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
            backend.to_tensor(
                [
                    [[9.0, 10.0]],
                    [[11.0, 12.0]],
                ]
            ),
            backend.to_tensor(
                [
                    [[13.0, 14.0], [15.0, 16.0]],
                    [[17.0, 18.0], [19.0, 20.0]],
                ]
            ),
        ]
        result_tensor = backend.concatenate(tensors, axis=1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 5, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0], [13.0, 14.0], [15.0, 16.0]],
                [[5.0, 6.0], [7.0, 8.0], [11.0, 12.0], [17.0, 18.0], [19.0, 20.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_concatenate_accepts_singleton_sequence(self):
        """
        Confirms that if we pass a single tensort to concatanate
        we get it back unchanged.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result_tensor = backend.concatenate([tensor], axis=0)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 2))
        assert_nested_close(
            result,
            [[1.0, 2.0], [3.0, 4.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_concatenate_rejects_empty_sequence(self):
        backend = self.make_backend()

        with self.assertRaises(ValueError):
            backend.concatenate([], axis=0)

    def test_concatenate_rejects_axis_out_of_range(self):
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]

        test_cases = [
            ("positive_out_of_range", 2),
            ("negative_out_of_range", -3),
        ]

        for case_name, axis in test_cases:
            with self.subTest(case=case_name):
                with self.assertRaises(ValueError):
                    backend.concatenate(tensors, axis=axis)

    def test_concatenate_rejects_rank_mismatch(self):
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        ]

        with self.assertRaises(ValueError):
            backend.concatenate(tensors, axis=0)

    def test_concatenate_rejects_shape_mismatch_outside_chosen_axis(self):
        backend = self.make_backend()
        test_cases = [
            (
                "axis_0_column_mismatch",
                [
                    backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
                    backend.to_tensor([[5.0, 6.0, 7.0]]),
                ],
                0,
            ),
            (
                "axis_1_row_mismatch",
                [
                    backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
                    backend.to_tensor([[5.0], [6.0], [7.0]]),
                ],
                1,
            ),
        ]

        for case_name, tensors, axis in test_cases:
            with self.subTest(case=case_name):
                with self.assertRaises(ValueError):
                    backend.concatenate(tensors, axis=axis)


@EnforceSharedNumericFixtures()
class BackendContractStackSemanticsMixin(BackendContractBase):
    def test_stack_joins_1D_tensors_along_axis_0(self):
        """
        Tests that stack joins two 1D tensors by inserting a new leading axis.

        Stacking along axis 0 of a 1D tensor means creating a new outer axis
        and placing each input tensor as one row of the result.

        The two input tensors are:

        [1.0, 2.0, 3.0]
        [4.0, 5.0, 6.0]

        The result is:

        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]    (shape (2, 3))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([1.0, 2.0, 3.0]),
            backend.to_tensor([4.0, 5.0, 6.0]),
        ]
        result_tensor = backend.stack(tensors, axis=0)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 3))
        assert_nested_close(
            result,
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_stack_joins_1D_tensors_along_axis_1(self):
        """
        Tests that stack joins two 1D tensors by inserting a new trailing axis.

        Stacking along axis 1 inserts a new axis after the existing one. The
        original value positions become the rows of the result, and each input
        tensor contributes one column: the value it held at that position. With
        two inputs of shape (3,), the result has shape (3, 2) — 3 rows (one per
        original position) and 2 columns (one per input tensor).

        The two input tensors are:

        [1.0, 2.0, 3.0]
        [4.0, 5.0, 6.0]

        The result is:

        [1.0, 4.0]    <- values at position 0 from each input
        [2.0, 5.0]    <- values at position 1 from each input
        [3.0, 6.0]    <- values at position 2 from each input
        (shape (3, 2))

        In this case the effect is to stack the tensors along axis 0 then rotate
        and flip the result but this description does not hold for
        higher-dimension operations.
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([1.0, 2.0, 3.0]),
            backend.to_tensor([4.0, 5.0, 6.0]),
        ]
        result_tensor = backend.stack(tensors, axis=1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (3, 2))
        assert_nested_close(
            result,
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_stack_joins_2D_tensors_along_axis_0(self):
        """
        Tests that stack joins two 2D tensors by inserting a new leading axis.

        Stacking along axis 0 of a 2D tensor means creating a new outer axis
        and placing each input tensor as one layer of the result. With two
        inputs of shape (2, 2), the result has shape (2, 2, 2): 2 layers
        (one per input tensor), each of shape (2, 2).

        The two input tensors are:

        [1.0, 2.0]    (shape (2, 2))
        [3.0, 4.0]

        [5.0, 6.0]    (shape (2, 2))
        [7.0, 8.0]

        The result is:

        [
            [[1.0, 2.0], [3.0, 4.0]],    <- first input becomes layer 0
            [[5.0, 6.0], [7.0, 8.0]]     <- second input becomes layer 1
        ]    (shape (2, 2, 2))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        result_tensor = backend.stack(tensors, axis=0)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 2, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_stack_joins_2D_tensors_along_axis_1(self):
        """
        Tests that stack joins two 2D tensors by inserting a new middle axis.

        Stacking along axis 1 inserts a new axis between the row and column
        axes. For each row position, the corresponding rows from each input
        tensor are grouped together. With two inputs of shape (2, 2), the
        result has shape (2, 2, 2): 2 row positions, each containing 2 rows
        (one per input tensor) of length 2.

        The two input tensors are:

        [1.0, 2.0]    (shape (2, 2))
        [3.0, 4.0]

        [5.0, 6.0]    (shape (2, 2))
        [7.0, 8.0]

        The result is:

        [
            [[1.0, 2.0], [5.0, 6.0]],    <- row 0 from each input
            [[3.0, 4.0], [7.0, 8.0]]     <- row 1 from each input
        ]    (shape (2, 2, 2))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        result_tensor = backend.stack(tensors, axis=1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 2, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0], [5.0, 6.0]],
                [[3.0, 4.0], [7.0, 8.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_stack_joins_2D_tensors_along_axis_2(self):
        """
        Tests that stack joins two 2D tensors by inserting a new trailing axis.

        Stacking along axis 2 inserts a new innermost axis. For each individual
        value position (row, column), the values from each input tensor at that
        position are grouped together. With two inputs of shape (2, 2), the
        result has shape (2, 2, 2): the same (2, 2) grid of positions, each
        now holding 2 values (one per input tensor).

        The two input tensors are:

        [1.0, 2.0]    (shape (2, 2))
        [3.0, 4.0]

        [5.0, 6.0]    (shape (2, 2))
        [7.0, 8.0]

        The result is:

        [
            [[1.0, 5.0], [2.0, 6.0]],    <- position (0,0) and (0,1) from each input
            [[3.0, 7.0], [4.0, 8.0]]     <- position (1,0) and (1,1) from each input
        ]    (shape (2, 2, 2))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        result_tensor = backend.stack(tensors, axis=2)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 2, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 5.0], [2.0, 6.0]],
                [[3.0, 7.0], [4.0, 8.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_stack_joins_3D_tensors_along_axis_1(self):
        """
        Tests that stack joins two 3D tensors by inserting a new second axis.

        Stacking along axis 1 inserts a new axis after the first. For each
        axis-0 position, the corresponding 2D matrices from each input tensor
        are grouped together. With two inputs of shape (2, 3, 2), the result
        has shape (2, 2, 3, 2): the same 2 outer positions, each now containing
        2 matrices (one per input tensor) of shape (3, 2).

        The two input tensors (the 'larger_middle_dimension' case) are:

        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ]    (shape (2, 3, 2))

        [
            [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
            [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]
        ]    (shape (2, 3, 2))

        The result is:

        [
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],      <- axis-0 pos 0 from input 0
                [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]  <- axis-0 pos 0 from input 1
            ],
            [
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],    <- axis-0 pos 1 from input 0
                [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]  <- axis-0 pos 1 from input 1
            ]
        ]    (shape (2, 2, 3, 2))

        The test also includes a 'singleton_dimension' sub-case where the middle
        axis has size 1, to confirm that backends can handle the irregular shape.
        """
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                [
                    backend.to_tensor(
                        [
                            [[1.0, 2.0, 3.0]],
                            [[4.0, 5.0, 6.0]],
                        ]
                    ),
                    backend.to_tensor(
                        [
                            [[7.0, 8.0, 9.0]],
                            [[10.0, 11.0, 12.0]],
                        ]
                    ),
                ],
                [
                    [
                        [[1.0, 2.0, 3.0]],
                        [[7.0, 8.0, 9.0]],
                    ],
                    [
                        [[4.0, 5.0, 6.0]],
                        [[10.0, 11.0, 12.0]],
                    ],
                ],
                (2, 2, 1, 3),
            ),
            (
                "larger_middle_dimension",
                [
                    backend.to_tensor(
                        [
                            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                        ]
                    ),
                    backend.to_tensor(
                        [
                            [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
                            [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
                        ]
                    ),
                ],
                [
                    [
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
                    ],
                    [
                        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                        [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
                    ],
                ],
                (2, 2, 3, 2),
            ),
        ]

        for case_name, tensors, expected, expected_shape in test_cases:
            result_tensor = backend.stack(tensors, axis=1)
            result = backend.to_python(result_tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), expected_shape)
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_stack_joins_3D_tensors_along_axis_3(self):
        """
        Tests that stack joins two 3D tensors by inserting a new trailing axis.

        Stacking along axis 3 inserts a new innermost axis. For each individual
        value position (i, j, k) in the inputs, the values from each input tensor
        at that position are grouped together. With two inputs of shape (2, 3, 2),
        the result has shape (2, 3, 2, 2): the same (2, 3, 2) grid of positions,
        each now holding 2 values (one per input tensor).

        The two input tensors (the 'larger_middle_dimension' case) are:

        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ]    (shape (2, 3, 2))

        [
            [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
            [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]
        ]    (shape (2, 3, 2))

        The result is:

        [
            [
                [[1.0, 13.0], [2.0, 14.0]],
                [[3.0, 15.0], [4.0, 16.0]],
                [[5.0, 17.0], [6.0, 18.0]]
            ],
            [
                [[7.0, 19.0], [8.0, 20.0]],
                [[9.0, 21.0], [10.0, 22.0]],
                [[11.0, 23.0], [12.0, 24.0]]
            ]
        ]    (shape (2, 3, 2, 2))

        The test also includes a 'singleton_dimension' sub-case where the middle
        axis has size 1, to confirm the operation is not sensitive to that dimension.
        """
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                [
                    backend.to_tensor(
                        [
                            [[1.0, 2.0, 3.0]],
                            [[4.0, 5.0, 6.0]],
                        ]
                    ),
                    backend.to_tensor(
                        [
                            [[7.0, 8.0, 9.0]],
                            [[10.0, 11.0, 12.0]],
                        ]
                    ),
                ],
                [
                    [
                        [[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]],
                    ],
                    [
                        [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]],
                    ],
                ],
                (2, 1, 3, 2),
            ),
            (
                "larger_middle_dimension",
                [
                    backend.to_tensor(
                        [
                            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                        ]
                    ),
                    backend.to_tensor(
                        [
                            [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
                            [[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
                        ]
                    ),
                ],
                [
                    [
                        [[1.0, 13.0], [2.0, 14.0]],
                        [[3.0, 15.0], [4.0, 16.0]],
                        [[5.0, 17.0], [6.0, 18.0]],
                    ],
                    [
                        [[7.0, 19.0], [8.0, 20.0]],
                        [[9.0, 21.0], [10.0, 22.0]],
                        [[11.0, 23.0], [12.0, 24.0]],
                    ],
                ],
                (2, 3, 2, 2),
            ),
        ]

        for case_name, tensors, expected, expected_shape in test_cases:
            result_tensor = backend.stack(tensors, axis=3)
            result = backend.to_python(result_tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), expected_shape)
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_stack_joins_more_than_two_tensors(self):
        """
        Tests that stack can join more than two tensors along axis 0.

        Stacking along axis 0 of a 2D tensor means creating a new outer axis
        and placing each input tensor as one layer of the result. This test
        uses three input tensors to ensure the method is not limited to
        combining exactly two tensors.

        The three input tensors are:

        [1.0, 2.0]    (shape (2, 2))
        [3.0, 4.0]

        [5.0, 6.0]    (shape (2, 2))
        [7.0, 8.0]

        [9.0, 10.0]    (shape (2, 2))
        [11.0, 12.0]

        The result is:

        [
            [[1.0, 2.0], [3.0, 4.0]],     <- first input becomes layer 0
            [[5.0, 6.0], [7.0, 8.0]],     <- second input becomes layer 1
            [[9.0, 10.0], [11.0, 12.0]]   <- third input becomes layer 2
        ]    (shape (3, 2, 2))
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0], [7.0, 8.0]]),
            backend.to_tensor([[9.0, 10.0], [11.0, 12.0]]),
        ]
        result_tensor = backend.stack(tensors, axis=0)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (3, 2, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_stack_accepts_singleton_sequence(self):
        """
        Confirms that passing a single tensor to stack returns it wrapped in
        a new axis of size 1.

        With one input of shape (2, 2) stacked along axis 1, the result has
        shape (2, 1, 2): the new axis is inserted between the row and column
        axes, holding the single input tensor at position 0.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result_tensor = backend.stack([tensor], axis=1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 1, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_stack_accepts_negative_axis(self):
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        result_tensor = backend.stack(tensors, axis=-1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 2, 2))
        assert_nested_close(
            result,
            [
                [[1.0, 5.0], [2.0, 6.0]],
                [[3.0, 7.0], [4.0, 8.0]],
            ],
            rel_tol=0,
            abs_tol=0,
        )

    def test_stack_rejects_empty_sequence(self):
        backend = self.make_backend()

        with self.assertRaises(ValueError):
            backend.stack([], axis=0)

    def test_stack_rejects_axis_out_of_range(self):
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        test_cases = [
            ("positive_out_of_range", 3),
            ("negative_out_of_range", -4),
        ]

        for case_name, axis in test_cases:
            with self.subTest(case=case_name):
                with self.assertRaises(ValueError):
                    backend.stack(tensors, axis=axis)

    def test_stack_rejects_rank_mismatch(self):
        """
        Tests that stack raises when the input tensors have different numbers
        of dimensions.

        Unlike concatenate, which requires matching shapes only on the axes
        that are not being joined, stack requires all input tensors to have
        identical shapes. A rank mismatch is therefore always invalid.
        """
        backend = self.make_backend()
        tensors = [
            backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
            backend.to_tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        ]

        with self.assertRaises(ValueError):
            backend.stack(tensors, axis=0)

    def test_stack_rejects_shape_mismatch(self):
        """
        Tests that stack raises when the input tensors have the same number of
        dimensions but differ in shape.

        Stack always inserts a new axis and keeps each input tensor intact
        within the result, so all inputs must have exactly the same shape.
        Any difference in size along any existing axis is invalid.
        """
        backend = self.make_backend()
        test_cases = [
            (
                "1D_length_mismatch",
                [
                    backend.to_tensor([1.0, 2.0, 3.0]),
                    backend.to_tensor([4.0, 5.0]),
                ],
            ),
            (
                "2D_shape_mismatch",
                [
                    backend.to_tensor([[1.0, 2.0], [3.0, 4.0]]),
                    backend.to_tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
                ],
            ),
        ]

        for case_name, tensors in test_cases:
            with self.subTest(case=case_name):
                with self.assertRaises(ValueError):
                    backend.stack(tensors, axis=0)
