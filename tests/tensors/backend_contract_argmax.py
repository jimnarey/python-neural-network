"""Backend contract argmax tests

This module tests the shared backend contract for argmax.

Unlike max, which returns the largest value itself, argmax returns the
index of the largest value.

When no axis is provided as an argument it flattens the tensor into a
single dimension and returns the largest value's index within that
dimension.

When an axis is provided, the result is a tensor rather than a single
integer. Searching through one axis means running the search independently
once for each position in every other axis. For a tensor of shape (2, 3)
with axis=0, there are 3 positions along axis 1 (the column axis), so
argmax runs 3 separate searches — one per column — and returns one index
from each search, giving shape (3,). With axis=1 there are 2 positions
along axis 0 (the row axis), giving shape (2,). In general the result shape
is the input shape with the searched axis removed.

The axis argument selects the dimension you search *through*, not a
dimension you preserve. Passing axis=0 means searching through the row
axis: for each fixed column position, compare all the row values and
return the row index of the largest. Passing axis=1 means searching
through the column axis: for each fixed row position, compare all the
column values and return the column index of the largest.

The tests in this module focus on the parts of that behaviour which every
backend must share: the ordinary result shapes and values, the accepted
and rejected forms of the axis argument, the fact that keepdims is not
supported, and the behaviour when the maximum value is tied.
"""

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures


@EnforceSharedNumericFixtures()
class BackendContractArgMaxSemanticsMixin(BackendContractBase):
    """
    This class tests that argmax behaves as expected. Each of the methods
    tests that we get both the expected value and type (always int).

    Values in a tensor can be described by their positions. For a 2D tensor,
    the position of each value is given by a (row, column) pair. The axis
    argument to argmax names the dimension to search through:

    - axis=0 means searching through axis 0 (the row axis). For each fixed
      column position j, argmax compares the values at (0, j), (1, j), etc.
      and returns the row index i at which the maximum was found.

    - axis=1 means searching through axis 1 (the column axis). For each fixed
      row position i, argmax compares the values at (i, 0), (i, 1), etc.
      and returns the column index j at which the maximum was found.

    In both cases the result has one fewer dimension than the input: the axis
    that was searched through is removed and replaced by a single index.
    """

    def test_argmax_returns_expected_result_for_1D_tensor(self):
        """
        Tests that argmax returns the index of the largest value in a 1D tensor.

        The tensor is [3.0, 7.0, 5.0]. The largest value is 7.0, at index 1.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([3.0, 7.0, 5.0])

        result = backend.argmax(tensor)

        self.assertEqual(result, 1)
        self.assertIsInstance(result, int)

    def test_argmax_returns_int_when_called_without_axis_on_3D_tensor(self):
        """
        Tests argmax without an axis argument on a 3D tensor.

        When no axis is passed, argmax flattens the tensor into a single
        sequence in row-major order and returns the index of the largest
        value in that sequence. The flattened sequence in this test is:

        [1.0, 5.0, 7.0, 2.0, 3.0, 9.0, 4.0, 6.0]

        The largest value is 9.0, which is at index 5 in this sequence,
        so the result should be the scalar 5.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 5.0], [7.0, 2.0]],
                [[3.0, 9.0], [4.0, 6.0]],
            ]
        )

        result = backend.argmax(tensor)

        self.assertEqual(result, 5)
        self.assertIsInstance(result, int)

    def test_argmax_returns_1D_tensor_when_called_on_2D_tensor_with_axis_0(self):
        """
        Tests argmax when searching through axis 0 of a 2D tensor.

        Searching through axis 0 means comparing row values for each column
        position. The tensor has 3 column positions, so there are 3 searches:

        - column 0: values 1.0 (row 0) and 4.0 (row 1) -> largest at row index 1
        - column 1: values 5.0 (row 0) and 2.0 (row 1) -> largest at row index 0
        - column 2: values 3.0 (row 0) and 6.0 (row 1) -> largest at row index 1

        The result is [1, 0, 1] with shape (3,): one row index per column.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

        result_tensor = backend.argmax(tensor, axis=0)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (3,))
        self.assertEqual(result, [1, 0, 1])
        for value in result:
            self.assertIsInstance(value, int)

    def test_argmax_returns_1D_tensor_when_called_on_2D_tensor_with_axis_1(self):
        """
        Tests argmax when searching through axis 1 of a 2D tensor.

        Searching through axis 1 means comparing column values for each row
        position. The tensor has 2 row positions, so there are 2 searches:

        - row 0: values 1.0, 5.0, 3.0 at column positions 0, 1, 2: column index 1
        - row 1: values 4.0, 2.0, 6.0 at column positions 0, 1, 2: column index 2

        The result is [1, 2] with shape (2,): one column index per row.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

        result_tensor = backend.argmax(tensor, axis=1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2,))
        self.assertEqual(result, [1, 2])
        for value in result:
            self.assertIsInstance(value, int)

    def test_argmax_returns_2D_tensor_when_called_on_3D_tensor_with_axis_1(self):
        """
        Tests argmax when searching through axis 1 of a 3D tensor.

        The tensor has shape (2, 3, 2). Searching through axis 1 means comparing
        all 3 axis-1 values for each combination of an axis-0 position and an
        axis-2 position. There are 2 * 2 = 4 such combinations, giving a result
        of shape (2, 2):

        - axis-0 pos 0, axis-2 pos 0: values 1.0, 7.0, 3.0: index 1
        - axis-0 pos 0, axis-2 pos 1: values 5.0, 2.0, 9.0: index 2
        - axis-0 pos 1, axis-2 pos 0: values 4.0, 6.0, 2.0: index 1
        - axis-0 pos 1, axis-2 pos 1: values 8.0, 1.0, 10.0: index 2

        The result is [[1, 2], [1, 2]].
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 5.0], [7.0, 2.0], [3.0, 9.0]],
                [[4.0, 8.0], [6.0, 1.0], [2.0, 10.0]],
            ]
        )

        result_tensor = backend.argmax(tensor, axis=1)
        result = backend.to_python(result_tensor)

        self.assertEqual(backend.shape(result_tensor), (2, 2))
        self.assertEqual(result, [[1, 2], [1, 2]])
        for row in result:
            for value in row:
                self.assertIsInstance(value, int)


@EnforceSharedNumericFixtures()
class BackendContractArgMaxAxisArgumentMixin(BackendContractBase):
    """
    This class tests the accepted and rejected forms of the axis argument.

    The accepted forms are: an integer axis index, a negative integer (counting
    from the end), and None (equivalent to no axis). The rejected forms are:
    a tuple of axes (argmax can only search through one axis at a time, unlike
    the reduction methods), an integer outside the valid range for the tensor's
    number of dimensions, and a non-integer type.
    """

    def test_argmax_accepts_axis_none(self):
        """
        Tests that passing axis=None has the same effect as not
        passing an axis argument.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 5.0], [7.0, 2.0]],
                [[3.0, 9.0], [4.0, 6.0]],
            ]
        )
        result = backend.argmax(tensor, axis=None)
        self.assertEqual(result, 5)
        self.assertIsInstance(result, int)

    def test_argmax_accepts_zero_and_positive_axis_arguments(self):
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 5.0], [7.0, 2.0]],
                [[3.0, 9.0], [4.0, 6.0]],
            ]
        )
        accepted_axes = [0, 2]
        for axis in accepted_axes:
            with self.subTest(axis=axis):
                backend.argmax(tensor, axis=axis)

    def test_argmax_treats_negative_axis_as_counting_from_the_end(self):
        """
        Tests that a negative axis is treated in the same way as a negative
        value when passed as a Python list index.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor(
            [
                [[1.0, 5.0], [7.0, 2.0]],
                [[3.0, 9.0], [4.0, 6.0]],
            ]
        )
        positive_axis_result = backend.to_python(backend.argmax(tensor, axis=2))
        negative_axis_result = backend.to_python(backend.argmax(tensor, axis=-1))
        self.assertEqual(negative_axis_result, positive_axis_result)

    def test_argmax_rejects_axis_tuple(self):
        """
        argmax's axis param takes only a single axis value
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 5.0], [7.0, 2.0]])
        with self.assertRaises((TypeError, ValueError)):
            backend.argmax(tensor, axis=(1,))

    def test_argmax_rejects_axis_outside_the_valid_range(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 5.0], [7.0, 2.0]])
        invalid_axes = [2, -3]
        for axis in invalid_axes:
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    backend.argmax(tensor, axis=axis)

    def test_argmax_rejects_non_integer_axis(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 5.0], [7.0, 2.0]])
        invalid_axes = [1.5, "1"]
        for axis in invalid_axes:
            with self.subTest(axis=axis):
                with self.assertRaises((TypeError, ValueError)):
                    backend.argmax(tensor, axis=axis)


@EnforceSharedNumericFixtures()
class BackendContractArgMaxKeepdimsMixin(BackendContractBase):
    """
    keepdims is supported by the numeric reduction methods because it helps
    preserve shape for later tensor operations. We do not support it for
    argmax because argmax returns indices rather than reduced numeric values,
    so the practical benefit is smaller while the extra API and shape
    complexity would still have to be implemented in every backend.
    """

    def test_argmax_rejects_keepdims_argument(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[1.0, 5.0], [7.0, 2.0]])

        with self.assertRaises((TypeError, ValueError)):
            backend.argmax(tensor, keepdims=True)


@EnforceSharedNumericFixtures()
class BackendContractArgMaxTieBehaviourMixin(BackendContractBase):
    """
    This class encodes NumPy's tie-breaking convention for argmax explicitly,
    so that non-NumPy backends cannot silently differ.
    """

    def test_argmax_returns_first_index_when_maximum_value_is_tied(self):
        """
        This encodes NumPy's behaviour when the maximum value is tied.

        When the largest value appears more than once, argmax returns the
        index of the first occurrence rather than raising an exception or
        choosing any later occurrence.
        """
        backend = self.make_backend()
        tensor = backend.to_tensor([3.0, 7.0, 7.0, 5.0])

        result = backend.argmax(tensor)

        self.assertEqual(result, 1)
        self.assertIsInstance(result, int)
