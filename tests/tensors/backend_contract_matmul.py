"""Backend contract matmul tests

This module tries to cover the various ways in which we might go wrong
with the backend implementations without trying to catch every possible
edge case.

In particular, as the number of dimensions increases the number of
reasons for a mismatch between the 2 arrays being multiplied increases.
It is not practical to try to cover all of them.

An important thing to understand with these operations is that matmul,
as the name suggests, multiplies *matrices*. It can handle cases where
one operand is a 1D array (vector) by temporarily treating that array as
a single-row or single-column matrix for the calculation, then removing
that extra dimension from the result.

It can also handle arrays with more than 2 dimensions, but the actual
multiplication is still done only on 2D slices taken from the last axes.
The rules for how those slices are chosen, and what happens to the other
axes, are not always easy to understand. The docstrings for the relevant
tests try to explain this.

This module does not test all possible matmul behaviour. Its coverage of
broadcasting is limited to a small number of representative, leading-axis
cases, because that is the only kind of broadcasting used by NumPy-style
matmul and it is not relied upon heavily in the NNfSiP book.

The tests stop at 3 dimensions for now.

Work to further generalise the network may require pinning down more
matmul behaviour using, as always, NumPy as the reference.

"""

# It is really only the 2D * 1D and 2D * 2D tests below which
# cover functionality used in the NNfSiP book but the rest were
# added early to ensure backed implementations are general enough
# that they can be extended reasonably straightforwardly when
# needed

# TODO - 1D @ 3D. 3D @ 1D in semantics mixin
# (2, 2, 2, 3) @ (2, 2, 3, 2) -> (2, 2, 2, 2) in semantics mixin
# (2, 1, 2, 3) @ (1, 2, 3, 2) -> (2, 2, 2, 2) in broadcasting mixin
# (2, 2, 2, 3) @ (3, 2, 3, 2) should raise in broadcasting mixin

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures


@EnforceSharedNumericFixtures()
class BackendContractMatmulSemanticsMixin(BackendContractBase):

    def test_matmul_multiplies_two_square_2D_tensors(self):
        backend = self.make_backend()

        a = backend.to_tensor([[2.0, 0.0], [1.0, 3.0]])
        b = backend.to_tensor([[4.0, 1.0], [2.0, 5.0]])

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [8.0, 2.0],
            [10.0, 16.0],
        ]
        self.assertEqual(backend.shape(tensor), (2, 2))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_multiplies_square_and_non_square_2D_tensors(self):
        backend = self.make_backend()

        a = backend.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        b = backend.to_tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [21.0, 24.0, 27.0],
            [47.0, 54.0, 61.0],
        ]
        self.assertEqual(backend.shape(tensor), (2, 3))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_multiplies_two_non_square_2D_tensors_with_different_shapes(self):
        backend = self.make_backend()

        a = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = backend.to_tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [58.0, 64.0],
            [139.0, 154.0],
        ]
        self.assertEqual(backend.shape(tensor), (2, 2))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_multiplies_1D_tensor_and_2D_tensor(self):
        """
        A 1D array can be multiplied by a matrix if the length of the 1D
        array matches the number of rows in the matrix.

        The 1D array is treated as though it were a single row in a 2D
        matrix:

        (k,) @ (k, n) -> (n,)

        This coercion of the 1D array to 2D is temporary. The result is
        1D.
        """
        backend = self.make_backend()

        a = backend.to_tensor([1.0, 2.0, 3.0])
        b = backend.to_tensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [40.0, 46.0]
        self.assertEqual(backend.shape(tensor), (2,))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_multiplies_2D_tensor_and_1D_tensor(self):
        """
        A matrix can be multiplied by a 1D array if the number of columns in
        the matrix matches the length of the 1D array.

        The 1D array is treated as though it were a single column in a 2D
        matrix:

        (m, k) @ (k,) -> (m,)

        This coercion of the 1D array to 2D is temporary. The result is
        1D.
        """
        backend = self.make_backend()

        a = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = backend.to_tensor([7.0, 8.0, 9.0])

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [50.0, 122.0]
        self.assertEqual(backend.shape(tensor), (2,))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_multiplies_two_1D_tensors(self):
        """
        A 1D array can also be multiplied by another 1D array if they have the
        same length.

        In this case the result is a single scalar value rather than another
        array.

        For the calculation, the left-hand array is treated as a single-row
        matrix and the right-hand array as a single-column matrix. The temporary
        1 x 1 result is then returned as a scalar.
        """
        backend = self.make_backend()

        a = backend.to_tensor([1.0, 2.0, 3.0])
        b = backend.to_tensor([4.0, 5.0, 6.0])

        expected = 32.0
        # This is a scalar value already so we do not need to convert it with to_python
        result = backend.matmul(a, b)
        self.assertNotIsInstance(result, (list, tuple))
        self.assertIsInstance(result, (int, float))
        self.assertEqual(result, expected)

    def test_matmul_multiplies_corresponding_3D_chunks_when_both_operands_are_3D_tensors(
        self,
    ):
        """
        Matrix multiplication also works when both operands have more than 2
        dimensions.

        The matrix part of the calculation is still taken from the last 2 axes.
        Any earlier axes are used to group the matrices into separate chunks.
        In this case, each chunk on the left is multiplied by the matching chunk
        on the right because the leading axes have the same length, so the
        chunks match one-to-one without broadcasting.

        In this test the left-hand array has shape:

        (2, 2, 3)

        This means it contains 2 matrices, each with shape:

        (2, 3)

        The right-hand array has shape:

        (2, 3, 2)

        This means it also contains 2 matrices, each with shape:

        (3, 2)

        The first matrix in the left-hand array is multiplied by the first
        matrix in the right-hand array, and the second left-hand matrix is
        multiplied by the second right-hand matrix.

        The first chunk of the left-hand array is:

        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

        The first chunk of the right-hand array is:

        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]

        So the top-left value in the first result matrix is:

        [1.0, 2.0, 3.0] with [1.0, 3.0, 5.0]
        = 1.0*1.0 + 2.0*3.0 + 3.0*5.0
        = 22.0

        The top-right value in the first result matrix is:

        [1.0, 2.0, 3.0] with [2.0, 4.0, 6.0]
        = 1.0*2.0 + 2.0*4.0 + 3.0*6.0
        = 28.0

        The same pattern is then repeated for the second pair of chunks. So
        the leading axis of length 2 is preserved, and the result contains 2
        output matrices, each with shape:

        (2, 2)

        The overall result therefore has shape:

        (2, 2, 2)
        """
        backend = self.make_backend()

        a = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        b = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 0.0], [1.0, 2.0], [0.0, 1.0]],
            ]
        )

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [[22.0, 28.0], [49.0, 64.0]],
            [[22.0, 25.0], [31.0, 34.0]],
        ]
        self.assertEqual(backend.shape(tensor), (2, 2, 2))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_multiplies_each_3D_chunk_of_the_left_hand_tensor_by_the_same_2D_tensor(
        self,
    ):
        """
        Matrix multiplication still works when one or both operands have more
        than 2 dimensions (a wide range of combinations are possible as long
        as specific axes in each operand match, in this case the last axis of
        the left-hand operand and the second-to-last axis of the right-hand
        operand, which both have a length of 3).

        The matrix part of the calculation is always taken from the last axes.
        Any earlier axes are not multiplied together. Instead, they are used to
        group the matrices into separate chunks, and the same matrix
        multiplication is repeated for each chunk.

        In this test the left-hand array has shape:

        (2, 2, 3)

        This means:

        - the first axis has length 2, so there are 2 chunks
        - each chunk is a matrix with shape (2, 3)

        The right-hand array has shape:

        (3, 2)

        So each (2, 3) matrix in the left-hand array can be multiplied by the
        same (3, 2) matrix on the right.

        The first chunk of the left-hand array is:

        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

        The right-hand matrix is:

        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]

        So the top-left value in the first result matrix is:

        [1.0, 2.0, 3.0] with [1.0, 3.0, 5.0]
        = 1.0*1.0 + 2.0*3.0 + 3.0*5.0
        = 22.0

        The top-right value in the first result matrix is:

        [1.0, 2.0, 3.0] with [2.0, 4.0, 6.0]
        = 1.0*2.0 + 2.0*4.0 + 3.0*6.0
        = 28.0

        The same calculation is then repeated for the second chunk of the
        left-hand array. So the leading axis of length 2 is preserved, and the
        result contains 2 output matrices, each with shape:

        (2, 2)

        The overall result therefore has shape:

        (2, 2, 2)
        """
        backend = self.make_backend()

        a = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        b = backend.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [[22.0, 28.0], [49.0, 64.0]],
            [[76.0, 100.0], [103.0, 136.0]],
        ]
        self.assertEqual(backend.shape(tensor), (2, 2, 2))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_multiplies_the_same_2D_tensor_by_each_3D_chunk_on_the_right(
        self,
    ):
        """
        Matrix multiplication also works when the right-hand operand has more
        than 2 dimensions.

        The matrix part of the calculation is still taken from the last axes.
        Any earlier axes are used to group the matrices into separate chunks.

        In this test the left-hand matrix has shape:

        (2, 3)

        The right-hand tensor has shape:

        (2, 3, 2)

        This means the right-hand tensor contains 2 matrices, each with shape:

        (3, 2)

        So the same (2, 3) matrix on the left is multiplied by each (3, 2)
        matrix in the right-hand tensor. The result therefore contains 2 output
        matrices, each with shape:

        (2, 2)

        The overall result therefore has shape:

        (2, 2, 2)
        """
        backend = self.make_backend()

        a = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 0.0], [1.0, 2.0], [0.0, 1.0]],
            ]
        )

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [[22.0, 28.0], [49.0, 64.0]],
            [[4.0, 7.0], [13.0, 16.0]],
        ]
        self.assertEqual(backend.shape(tensor), (2, 2, 2))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_raises_when_2D_and_2D_inner_dimensions_do_not_match(self):
        """
        Matrix multiplication works by taking one row from the left matrix and
        one column from the right matrix, multiplying the values by index, and
        adding the results.

        So if the shapes are:

        (m, k) @ (k, n)

        then:

        - each row in the left matrix contains k values
        - each column in the right matrix also contains k values

        Those lengths must match, otherwise the row and column cannot be
        multiplied together.

        For example, if we multiply:

        [[a, b, c],
        [g, h, i]]

        by:

        [[d, j],
        [e, k],
        [f, l]]

        then the top-left value in the result is built from the first row of
        the left matrix and the first column of the right matrix:

        [a, b, c] with [d, e, f]
        = a*d + b*e + c*f

        The top-right value is built from the first row of the left matrix and
        the second column of the right matrix:

        [a, b, c] with [j, k, l]
        = a*j + b*k + c*l

        This is why the middle dimensions must match.

        In this test we try to multiply matrices with shapes:

        (2, 3) @ (2, 2)

        The rows of the left matrix have length 3, but the columns of the right
        matrix have length 2, so there is no way to pair up all the values for
        the calculation. The multiplication should therefore raise an exception.
        """
        backend = self.make_backend()

        a = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = backend.to_tensor([[7.0, 8.0], [9.0, 10.0]])

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_1D_and_2D_inner_dimensions_do_not_match(self):
        """
        A 1D array can only be multiplied by a matrix if the length of the
        1D array matches the number of rows in the matrix.
        """
        backend = self.make_backend()

        a = backend.to_tensor([1.0, 2.0, 3.0])
        b = backend.to_tensor([[4.0, 5.0], [6.0, 7.0]])

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_2D_and_1D_inner_dimensions_do_not_match(self):
        """
        A matrix can only be multiplied by a 1D array if the number of
        columns in the matrix matches the length of the 1D array.
        """
        backend = self.make_backend()

        a = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = backend.to_tensor([7.0, 8.0])

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_1D_and_1D_inner_dimensions_do_not_match(self):
        """
        A 1D array can only be multiplied by another 1D array if both arrays
        have the same length.
        """
        backend = self.make_backend()

        a = backend.to_tensor([1.0, 2.0, 3.0])
        b = backend.to_tensor([4.0, 5.0])

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_3D_and_2D_inner_dimensions_do_not_match(self):
        """
        A 3D array can only be multiplied by a matrix if the last axis of
        the 3D array matches the second-to-last axis of the matrix.

        In this test the left-hand array has shape:

        (2, 2, 3)

        So each chunk in the left-hand array is a matrix with shape:

        (2, 3)

        The right-hand matrix has shape:

        (4, 2)

        For the multiplication to work, each row of a (2, 3) matrix on the
        left must be able to multiply a column of the (4, 2) matrix on the
        right.

        But the rows on the left have length 3, while the columns on the right
        have length 4. So there is no way to pair up all the values by index
        for the calculation.

        The multiplication should therefore raise an exception.
        """
        backend = self.make_backend()

        a = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        b = backend.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_2D_and_3D_inner_dimensions_do_not_match(self):
        """
        A matrix can only be multiplied by a 3D array if the last axis of
        the matrix matches the second-to-last axis of each matrix in the
        3D array.

        The left-hand matrix has shape:

        (2, 3)

        Each matrix in the right-hand array has shape:

        (4, 2)

        For the multiplication to work, each row of the left-hand matrix must be
        able to multiply a column from one of the right-hand matrices.

        But the rows on the left have length 3, while the columns on the right
        have length 4. So there is no way to pair up all the values by index for
        the calculation.

        The multiplication should therefore raise an exception.
        """
        backend = self.make_backend()

        a = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[2.0, 0.0], [1.0, 2.0], [0.0, 1.0], [3.0, 4.0]],
            ]
        )

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_3D_and_3D_inner_dimensions_do_not_match(self):
        """
        A 3D array can only be multiplied by another 3D array if the last axis
        of each matrix in the left-hand array matches the second-to-last axis
        of each matrix in the right-hand array.

        In this test the left-hand array has shape:

        (2, 2, 3)

        So each chunk in the left-hand array is a matrix with shape:

        (2, 3)

        The right-hand array has shape:

        (2, 4, 2)

        So each chunk in the right-hand array is a matrix with shape:

        (4, 2)

        For the multiplication to work, each row of a (2, 3) matrix on the
        left must be able to multiply a column of a (4, 2) matrix on the
        right.

        But the rows on the left have length 3, while the columns on the right
        have length 4. So there is no way to pair up all the values by index
        for the calculation.

        The multiplication should therefore raise an exception.
        """
        backend = self.make_backend()

        a = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        b = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[2.0, 0.0], [1.0, 2.0], [0.0, 1.0], [3.0, 4.0]],
            ]
        )

        with self.assertRaises(ValueError):
            backend.matmul(a, b)


@EnforceSharedNumericFixtures()
class BackendContractMatmulBroadcastingMixin(BackendContractBase):
    """
    Broadcasting in matmul applies only to the leading axes. It can help when
    those axes differ, but only if one of them has length 1 and can be reused.
    It cannot help if the matrix dimensions themselves do not match, because
    the row-by-column multiplication would still be impossible. The
    higher-dimensional tests in this class, taken together, try to demonstrate
    this.
    """

    def test_matmul_reuses_the_only_right_hand_3D_chunk_when_leading_axis_broadcasting_applies(
        self,
    ):
        """
        Matrix multiplication can also reuse chunks from one operand when a
        leading axis has length 1.

        In this test the left-hand array has shape:

        (2, 2, 3)

        so it contains 2 matrices, each with shape:

        (2, 3)

        The right-hand array has shape:

        (1, 3, 2)

        so it contains just 1 matrix, with shape:

        (3, 2)

        The matrix dimensions are valid because:

        (2, 3) @ (3, 2)

        The leading axes do not match, but they are still compatible
        because one of them has length 1 and, with broadcasting, a leading
        axis of length 1 can be reused. In this case, the single matrix in
        the right-hand array is used for both matrices in the left-hand
        array.

        Broadcasting works here because the leading axes are:

        2 and 1

        This is allowed. A case such as 2 and 3 would not be allowed, because
        neither axis has length 1 and so neither can be reused.

        The first chunk of the left-hand array is:

        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

        The only chunk of the right-hand array is:

        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]

        So the top-left value in the first result matrix is:

        [1.0, 2.0, 3.0] with [1.0, 3.0, 5.0]
        = 1.0*1.0 + 2.0*3.0 + 3.0*5.0
        = 22.0

        The same right-hand matrix is then reused for the second chunk on the
        left. The result therefore contains 2 output matrices, each with
        shape:

        (2, 2)

        The overall result therefore has shape:

        (2, 2, 2)
        """
        backend = self.make_backend()

        a = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        b = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ]
        )

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [[22.0, 28.0], [49.0, 64.0]],
            [[76.0, 100.0], [103.0, 136.0]],
        ]
        self.assertEqual(backend.shape(tensor), (2, 2, 2))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_reuses_the_only_left_hand_3D_chunk_when_leading_axis_broadcasting_applies(
        self,
    ):
        """
        Leading-axis broadcasting also works the other way round, when the
        left-hand tensor has the axis of length 1.

        In this test the left-hand tensor has shape:

        (1, 2, 3)

        so it contains just 1 matrix, with shape:

        (2, 3)

        The right-hand tensor has shape:

        (2, 3, 2)

        so it contains 2 matrices, each with shape:

        (3, 2)

        The matrix dimensions are valid because:

        (2, 3) @ (3, 2)

        The same broadcasting rules apply as when the right-hand tensor has the
        axis of length 1.

        The only chunk of the left-hand tensor is:

        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

        The first chunk of the right-hand tensor is:

        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]

        So the top-left value in the first result matrix is:

        [1.0, 2.0, 3.0] with [1.0, 3.0, 5.0]
        = 1.0*1.0 + 2.0*3.0 + 3.0*5.0
        = 22.0

        The same left-hand matrix is then reused for the second chunk on the
        right. The result therefore contains 2 output matrices, each with
        shape:

        (2, 2)

        The overall result therefore has shape:

        (2, 2, 2)
        """
        backend = self.make_backend()

        a = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ]
        )
        b = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 0.0], [1.0, 2.0], [0.0, 1.0]],
            ]
        )

        tensor = backend.matmul(a, b)
        result = backend.to_python(tensor)

        expected = [
            [[22.0, 28.0], [49.0, 64.0]],
            [[4.0, 7.0], [13.0, 16.0]],
        ]
        self.assertEqual(backend.shape(tensor), (2, 2, 2))
        assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_matmul_raises_when_3D_and_3D_leading_axes_do_not_match(self):
        """
        A 3D array can also fail to multiply with another 3D array when
        the matrix dimensions are valid.

        In this test the left-hand array has shape:

        (2, 2, 3)

        and the right-hand array has shape:

        (3, 3, 2)

        The matrix part of the shapes is compatible:

        (2, 3) @ (3, 2)

        If we looked at just one matrix from the left-hand array and one
        matrix from the right-hand array, the multiplication would work.

        But the earlier axes do not match. The left-hand array is grouped into
        2 chunks, while the right-hand array is grouped into 3 chunks.

        If the leading axes did match, the first matrix in the left-hand array
        would be multiplied by the first matrix in the right-hand array, the
        second by the second etc

        So there is no one-to-one pairing between the matrices in the
        left-hand array and the matrices in the right-hand array for this
        calculation: the right-hand array contains a third chunk with no
        matching chunk on the left.

        Broadcasting does not help here because the leading axes are 2 and 3,
        (neither of them is 1).

        The multiplication should therefore raise an exception.
        """
        backend = self.make_backend()

        a = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        b = backend.to_tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 0.0], [1.0, 2.0], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            ]
        )

        with self.assertRaises(ValueError):
            backend.matmul(a, b)
