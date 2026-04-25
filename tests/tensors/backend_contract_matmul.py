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
multiplication is still carried out only on matrices formed from the
last two axes of each operand. What can be harder to understand is how
the leading axes are handled. The docstrings for the relevant tests try
to explain this.

The terms 'leading axes matmul' and 'trailing axes matmul' refer to this
same behaviour. The first term emphasises that the earlier axes have to
be handled coherently; the second that it is the last to axes (forming
a matrix) that we actually multiply.

This module does not test all possible higher-dimension matmul behaviour.
Tesing is limited to a small number of representative cases.

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
    """
    This class pins down the contract for matrix multiplication.

    Matrix multiplication is not an elementwise operation. It works by
    multiplying the rows of the first operand by the columns of the second.
    This convention can seem a bit arbitrary at first but makes sense when
    it comes to chaining multiple operations together.

    Each result value is formed from a whole row of the left-hand matrix
    and a whole column of the right-hand matrix, not from two values in the
    same position.

    This is what allows chaining. If an input row vector x is first
    multiplied by a matrix A, and the result is then multiplied by a matrix
    B, the whole calculation can be represented by a single matrix A @ B:

    (x @ A) @ B == x @ (A @ B)
    """

    def test_matmul_multiplies_two_square_2D_tensors(self):
        """
        This tests matrix multiplication for the simplest case.

        The left-hand tensor is:

        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]

        The right-hand tensor is:

        [
            [5.0, 6.0],
            [7.0, 8.0]
        ]

        The result should be:

        [
            [19.0, 22.0],
            [43.0, 50.0]
        ]

        Each result value is calculated by multiplying a row from the left-hand
        tensor by a column from the right-hand tensor and summing the products.

        So:

        - top left: [1.0, 2.0] with [5.0, 7.0] -> 1.0*5.0 + 2.0*7.0 = 19.0
        - top right: [1.0, 2.0] with [6.0, 8.0] -> 1.0*6.0 + 2.0*8.0 = 22.0
        - bottom left: [3.0, 4.0] with [5.0, 7.0] -> 3.0*5.0 + 4.0*7.0 = 43.0
        - bottom right: [3.0, 4.0] with [6.0, 8.0] -> 3.0*6.0 + 4.0*8.0 = 50.0
        """
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
        """
        These tensors can be multiplied without any broadcasting even though
        their shapes differ. What matters is that the number of columns in
        the left-hand tensor matches the number of rows in the right-hand tensor.
        Here the left-hand tensor has shape (2, 2) and the right-hand tensor has
        shape (2, 3), so each row from the left-hand tensor has length 2 and
        each column from the right-hand tensor also has length 2.

        The left-hand tensor is:

        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]

        The right-hand tensor is:

        [
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0]
        ]

        The result should be:

        [
            [21.0, 24.0, 27.0],
            [47.0, 54.0, 61.0]
        ]

        Each result value is calculated by multiplying a row from the left-hand
        tensor by a column from the right-hand tensor and summing the products.

        So:

        - top left: [1.0, 2.0] with [5.0, 8.0] -> 1.0*5.0 + 2.0*8.0 = 21.0
        - top middle: [1.0, 2.0] with [6.0, 9.0] -> 1.0*6.0 + 2.0*9.0 = 24.0
        - top right: [1.0, 2.0] with [7.0, 10.0] -> 1.0*7.0 + 2.0*10.0 = 27.0
        - bottom left: [3.0, 4.0] with [5.0, 8.0] -> 3.0*5.0 + 4.0*8.0 = 47.0
        - bottom middle: [3.0, 4.0] with [6.0, 9.0] -> 3.0*6.0 + 4.0*9.0 = 54.0
        - bottom right: [3.0, 4.0] with [7.0, 10.0] -> 3.0*7.0 + 4.0*10.0 = 61.0
        """
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
        """
        These tensors can be multiplied without raising an exception or requiring
        broadcasting because the length of each row in the first tensor is the same
        as the length of each column in the second tensor.

        The left-hand tensor is:

        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]

        The right-hand tensor is:

        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ]

        The result should be:

        [
            [58.0, 64.0],
            [139.0, 154.0]
        ]

        Each result value is calculated by multiplying a row from the left-hand
        tensor by a column from the right-hand tensor and summing the products.

        So:

        - top left: [1.0, 2.0, 3.0] with [7.0, 9.0, 11.0] -> 1.0*7.0 + 2.0*9.0 + 3.0*11.0 = 58.0
        - top right: [1.0, 2.0, 3.0] with [8.0, 10.0, 12.0] -> 1.0*8.0 + 2.0*10.0 + 3.0*12.0 = 64.0
        - bottom left: [4.0, 5.0, 6.0] with [7.0, 9.0, 11.0] -> 4.0*7.0 + 5.0*9.0 + 6.0*11.0 = 139.0
        - bottom right: [4.0, 5.0, 6.0] with [8.0, 10.0, 12.0] -> 4.0*8.0 + 5.0*10.0 + 6.0*12.0 = 154.0
        """
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

    def test_matmul_multiplies_paired_matrices_when_both_operands_are_3D_tensors(
        self,
    ):
        """
        Matrix multiplication also works when both operands are 3D tensors.

        A 3D tensor can be thought of as a stack of matrices. The last two axes
        form each matrix, and the first axis is the position in the stack.

        In this test the left-hand array has shape (2, 2, 3), so it is a stack
        of 2 matrices each with shape (2, 3). The right-hand array has shape
        (2, 3, 2), so it is also a stack of 2 matrices each with shape (3, 2).

        Because both stacks have the same length, matmul pairs up the matrices
        by position: the first matrix on the left is multiplied by the first
        matrix on the right, and the second by the second.

        As with a normal 2D @ 2D matmul, the matrices in this case can be
        multiplied because each left-hand matrix has shape (2, 3) and each
        right-hand matrix has shape (3, 2). I.e. the number of columns in the
        left-hand matrix matches the number of rows in the right-hand matrix,
        so each row on the left can be paired with each column on the right.

        The first matrix in the left-hand array is:

        [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]]

        The first matrix in the right-hand array is:

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

        The same calculation is then applied to the second pair of matrices.
        Each result matrix has shape (2, 2), and there are 2 of them, so the
        overall result has shape (2, 2, 2).
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

    def test_matmul_multiplies_a_2D_matrix_by_each_matrix_in_a_3D_stack(
        self,
    ):
        """
        When one operand is a 3D tensor (i.e. a stack of matrices) and the
        other is a 2D tensor (matrix), matmul multiplies the single matrix
        by each matrix in the 3D stack in turn.

        In this test the left-hand tensor has shape (2, 2, 3), so it is a stack
        of 2 matrices each with shape (2, 3). The right-hand matrix has shape
        (3, 2). Because each left-hand matrix has 3 columns and the right-hand
        matrix has 3 rows, the multiplication is valid for each pair.

        The first matrix in the left-hand tensor is:

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

        The same right-hand matrix is then applied to the second matrix in the
        stack. Each result matrix has shape (2, 2), and there are 2 of them, so
        the overall result has shape (2, 2, 2).
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

    def test_matmul_multiplies_each_matrix_in_a_3D_stack_by_a_2D_matrix(
        self,
    ):
        """
        The mirror case of the previous test: when the left-hand operand is a
        plain 2D matrix and the right-hand operand is a 3D tensor (a stack of
        matrices), matmul multiplies the single matrix by each matrix in the
        3D stack in turn.

        In this test the left-hand matrix has shape (2, 3). The right-hand
        tensor has shape (2, 3, 2), so it is a stack of 2 matrices each with
        shape (3, 2). Because the left-hand matrix has 3 columns and each
        right-hand matrix has 3 rows, the multiplication is valid for each pair.

        The left-hand matrix is:

        [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]]

        The first matrix in the right-hand stack is:

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

        The same left-hand matrix is then applied to the second matrix in the
        stack. Each result matrix has shape (2, 2), and there are 2 of them, so
        the overall result has shape (2, 2, 2).
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

    def test_matmul_raises_when_2D_columns_do_not_match_2D_rows(self):
        """
        Matrix multiplication requires the number of columns in the left matrix
        to equal the number of rows in the right matrix.

        So if the shapes are:

        (m, k) @ (k, n)

        then:

        - each row in the left matrix contains k values
        - each column in the right matrix also contains k values

        Those lengths must match, otherwise each row and column cannot be
        multiplied together element by element and summed.

        For example, if we multiply:

        [[A, B, C],
        [D, E, F]]

        by:

        [[P, Q],
        [R, S],
        [T, U]]

        then the top-left value in the result is built from the first row of
        the left matrix and the first column of the right matrix:

        [A, B, C] with [P, R, T]
        = A*P + B*R + C*T

        The top-right value is built from the first row of the left matrix and
        the second column of the right matrix:

        [A, B, C] with [Q, S, U]
        = A*Q + B*S + C*U

        The number of columns in the left matrix must equal the number of rows
        in the right matrix, because each value in a row must be paired with
        exactly one value from a column.

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

    def test_matmul_raises_when_1D_length_does_not_match_2D_rows(self):
        """
        A 1D tensor can only be multiplied by a matrix if the length of the
        1D tensor matches the number of rows in the matrix.
        """
        backend = self.make_backend()

        a = backend.to_tensor([1.0, 2.0, 3.0])
        b = backend.to_tensor([[4.0, 5.0], [6.0, 7.0]])

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_2D_columns_do_not_match_1D_length(self):
        """
        A matrix can only be multiplied by a 1D tensor if the number of
        columns in the matrix matches the length of the 1D tensor.
        """
        backend = self.make_backend()

        a = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = backend.to_tensor([7.0, 8.0])

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_two_1D_tensors_have_different_lengths(self):
        """
        A 1D tensor can only be multiplied by another 1D tensor if both tensors
        have the same length.
        """
        backend = self.make_backend()

        a = backend.to_tensor([1.0, 2.0, 3.0])
        b = backend.to_tensor([4.0, 5.0])

        with self.assertRaises(ValueError):
            backend.matmul(a, b)

    def test_matmul_raises_when_3D_stack_columns_do_not_match_2D_rows(self):
        """
        When a 3D tensor (a stack of matrices) is multiplied by a 2D matrix,
        the number of columns in each matrix in the stack must equal the number
        of rows in the right-hand matrix.

        In this test the left-hand tensor has shape (2, 2, 3), so it is a stack
        of 2 matrices each with shape (2, 3). The right-hand matrix has shape
        (4, 2).

        For the multiplication to work, the rows of each left-hand matrix must
        be able to multiply the columns of the right-hand matrix element by
        element and be summed.

        But the rows of each left-hand matrix have length 3, while the columns
        of the right-hand matrix have length 4. So there is no way to pair up
        all the values for the calculation.

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

    def test_matmul_raises_when_2D_columns_do_not_match_3D_stack_rows(self):
        """
        When a 2D matrix is multiplied by a 3D tensor (a stack of matrices),
        the number of columns in the left-hand matrix must equal the number of
        rows in each matrix in the stack.

        In this test the left-hand matrix has shape (2, 3). The right-hand
        array has shape (2, 4, 2), so it is a stack of 2 matrices each with
        shape (4, 2).

        For the multiplication to work, the rows of the left-hand matrix must
        be able to multiply the columns of each right-hand matrix element by
        element and be summed.

        But the rows of the left-hand matrix have length 3, while the columns
        of each right-hand matrix have length 4. So there is no way to pair up
        all the values for the calculation.

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

    def test_matmul_raises_when_3D_stack_columns_do_not_match_3D_stack_rows(self):
        """
        When both operands are 3D tensors (stacks of matrices), the number of
        columns in each matrix in the left-hand stack must equal the number of
        rows in each matrix in the right-hand stack.

        In this test the left-hand array has shape (2, 2, 3), so it is a stack
        of 2 matrices each with shape (2, 3). The right-hand array has shape
        (2, 4, 2), so it is also a stack of 2 matrices each with shape (4, 2).

        For the multiplication to work, the rows of each left-hand matrix must
        be able to multiply the columns of the corresponding right-hand matrix
        element by element and be summed.

        But the rows of each left-hand matrix have length 3, while the columns
        of each right-hand matrix have length 4. So there is no way to pair up
        all the values for the calculation.

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
    The 3D @ 3D tests in BackendContractMatmulSemanticsMixin always use two
    stacks of the same length. The tests in this class cover what happens
    when the two stacks have different lengths.

    When both operands are 3D tensors with stacks of the same length, matmul
    pairs up the matrices by position. When the stack lengths differ, there
    is no straightforward pairing — unless one of the stacks has length 1.

    A stack of length 1 contains just a single matrix. When one stack has
    length 1, matmul reuses that single matrix for every position in the
    other stack. This reuse is called broadcasting. The two happy-path tests
    in this class demonstrate this, one for each side (left and right) having
    the stack of length 1.

    If neither stack has length 1 and the lengths are unequal, there is no
    valid pairing and no reuse is possible. The raises test in this class
    demonstrates this.

    Broadcasting only resolves the stack length mismatch. Even when one stack
    has length 1, the matrix dimensions themselves must still be compatible:
    the number of columns in each left-hand matrix must equal the number of
    rows in each right-hand matrix. A mismatch there will still raise an
    exception.
    """

    def test_matmul_reuses_single_right_hand_matrix_when_3D_stack_lengths_differ(
        self,
    ):
        """
        When the right-hand stack has length 1, its single matrix is reused
        for each matrix in the left-hand stack.

        In this test the left-hand tensor has shape (2, 2, 3), so it is a
        stack of 2 matrices each with shape (2, 3). The right-hand tensor has
        shape (1, 3, 2), so it is a stack containing just 1 matrix with shape
        (3, 2).

        The matrix dimensions are compatible:

        (2, 3) @ (3, 2)

        Because the right-hand stack has length 1, the single right-hand
        matrix is used for both matrices in the left-hand stack.

        The first matrix in the left-hand stack is:

        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

        The only matrix in the right-hand stack is:

        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]

        So the top-left value in the first result matrix is:

        [1.0, 2.0, 3.0] with [1.0, 3.0, 5.0]
        = 1.0*1.0 + 2.0*3.0 + 3.0*5.0
        = 22.0

        The same right-hand matrix is then reused for the second matrix in
        the left-hand stack. Each result matrix has shape (2, 2), and there
        are 2 of them, so the overall result has shape (2, 2, 2).
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

    def test_matmul_reuses_single_left_hand_matrix_when_3D_stack_lengths_differ(
        self,
    ):
        """
        When the left-hand stack has length 1, its single matrix is reused for
        each matrix in the right-hand stack.

        In this test the left-hand tensor has shape (1, 2, 3), so it is a
        stack containing just 1 matrix with shape (2, 3). The right-hand
        tensor has shape (2, 3, 2), so it is a stack of 2 matrices each with
        shape (3, 2).

        The matrix dimensions are compatible:

        (2, 3) @ (3, 2)

        Because the left-hand stack has length 1, the single left-hand matrix
        is used for both matrices in the right-hand stack.

        The only matrix in the left-hand stack is:

        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

        The first matrix in the right-hand stack is:

        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]]

        So the top-left value in the first result matrix is:

        [1.0, 2.0, 3.0] with [1.0, 3.0, 5.0]
        = 1.0*1.0 + 2.0*3.0 + 3.0*5.0
        = 22.0

        The same left-hand matrix is then reused for the second matrix in
        the right-hand stack. Each result matrix has shape (2, 2), and there
        are 2 of them, so the overall result has shape (2, 2, 2).
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

    def test_matmul_raises_when_3D_stack_lengths_differ_and_neither_is_1(self):
        """
        When two 3D stacks have different lengths and neither length is 1,
        matmul cannot pair up the matrices and cannot broadcast, so it raises
        an exception.

        In this test the left-hand tensor has shape (2, 2, 3), so it is a
        stack of 2 matrices each with shape (2, 3). The right-hand tensor has
        shape (3, 3, 2), so it is a stack of 3 matrices each with shape (3, 2).

        The matrix dimensions are compatible:

        (2, 3) @ (3, 2)

        So the failure is not caused by the matrix dimensions. It is caused
        by the stack lengths: 2 and 3. Neither is 1, so there is no single
        matrix that can be reused, and the stacks are not the same length,
        so there is no position-by-position pairing.

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

    def test_matmul_raises_when_3D_stack_matrix_dimensions_incompatible_despite_broadcastable_lengths(
        self,
    ):
        """
        Broadcasting resolves a stack length mismatch, but it cannot fix
        incompatible matrix dimensions. Even when one stack has length 1, the
        number of columns in each left-hand matrix must still equal the number
        of rows in each right-hand matrix.

        In this test the left-hand tensor has shape (2, 2, 3), so it is a
        stack of 2 matrices each with shape (2, 3). The right-hand tensor has
        shape (1, 2, 2), so it is a stack containing just 1 matrix with shape
        (2, 2).

        The stack lengths are 2 and 1, so broadcasting would normally allow
        the single right-hand matrix to be reused. But the matrix dimensions
        are not compatible:

        (2, 3) @ (2, 2)

        The rows of each left-hand matrix have length 3, while the columns of
        the right-hand matrix have length 2. So there is no way to pair up all
        the values for the calculation.

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
                [[1.0, 2.0], [3.0, 4.0]],
            ]
        )

        with self.assertRaises(ValueError):
            backend.matmul(a, b)
