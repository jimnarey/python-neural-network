"""Contract tests for creating tensors from Python values.

The backend contract requires that all backend implementations have a method
(`to_tensor`) for converting a built-in Python scalar or a Python list/tuple
into the native tensor representation used by that backend.

A scalar represents a rank-zero tensor. Lists and tuples express tensor axes
through their nesting and must form a rectangular structure.

We need complementary tests at the implementation level for each backend.
These can inspect the backend's native tensor representation directly without
converting to lists/tuples.

This is particularly important for empty tensors. Some empty shapes,
such as (2, 0, 3), are valid but collapse to the same Python representation
as other shapes with empty dimensions.

The value tests require a round trip through `to_python`, but do not make
implementation-specific assertions about the scalar type it returns. The
backend implementation tests inspect native tensor representation directly.
"""

from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures
from tests.helpers.tensor_helpers import assert_nested_close
from tests.tensors.contract.shared import BackendContractBase


@EnforceSharedNumericFixtures()
class BackendContractToTensorValidInputTypeMixin(BackendContractBase):
    """
    Check which Python input types `to_tensor` accepts.

    Successful calls are enough for accepted types; their resulting shapes
    and values are tested by the other mixins and by implementation tests.
    """

    def test_to_tensor_accepts_list_input(self):
        backend = self.make_backend()
        backend.to_tensor([1.0, 2.0, 3.0])

    def test_to_tensor_accepts_tuple_input(self):
        backend = self.make_backend()
        backend.to_tensor((1.0, 2.0, 3.0))

    def test_to_tensor_accepts_builtin_float_and_int_scalar_input(self):
        backend = self.make_backend()
        for data in (1.0, 1):
            with self.subTest(data=data):
                backend.to_tensor(data)


class BackendContractToTensorInvalidInputTypeMixin(BackendContractBase):
    """
    Check which Python input types `to_tensor` rejects.
    """

    def test_to_tensor_rejects_non_numeric_input(self):
        backend = self.make_backend()
        non_numeric_inputs = [
            "a",
            True,
            None,
            dict(),
            set(),
            ["a", "b"],
            [[1.0, 2.0], ["a", "b"]],
            [1.0, None, 3.0],
            [False],
            [0.0, True],
            ((1.0, 2.0), (3.0, "x")),
        ]

        for data in non_numeric_inputs:
            with self.subTest(data=data):
                with self.assertRaises(
                    (TypeError, ValueError),
                    msg="to_tensor accepted non-numeric input when it should reject it",
                ):
                    backend.to_tensor(data)


@EnforceSharedNumericFixtures()
class BackendContractToTensorShapeInputMixin(BackendContractBase):

    def test_to_tensor_correctly_infers_shape(self):
        backend = self.make_backend()
        cases = (
            (3.0, ()),
            ([3.0], (1,)),
            ((3.0,), (1,)),
            ([[3.0]], (1, 1)),
            ([[[3.0]]], (1, 1, 1)),
        )
        for data, expected_shape in cases:
            with self.subTest(data=data):
                tensor = backend.to_tensor(data)
                self.assertEqual(backend.shape(tensor), expected_shape)

    def test_to_tensor_converts_empty_1D_input_to_tensor(self):
        """
        Check the native tensor shape directly because an empty Python
        structure cannot preserve every possible empty tensor shape.
        """
        backend = self.make_backend()
        test_cases = [([], (0,)), ((), (0,))]
        for data, expected_shape in test_cases:
            with self.subTest(data=data):
                tensor = backend.to_tensor(data)
                self.assertEqual(backend.shape(tensor), expected_shape)

    def test_to_tensor_converts_nested_empty_input_to_tensor_with_expected_shape(
        self,
    ):
        """
        Check that nesting before an empty sequence is retained as tensor axes.
        """
        backend = self.make_backend()
        test_cases = [
            # Empty 2D tensor represented with nested lists
            ([[], []], (2, 0)),
            # Empty 2D tensor represented with nested tuples
            (((), ()), (2, 0)),
            # Empty 2D tensor represented with mixed list/tuple nesting
            ([(), ()], (2, 0)),
            # Empty 2D tensor represented with mixed tuple/list nesting
            (([], []), (2, 0)),
            # Empty 3D tensor represented with nested lists
            ([[[]], [[]]], (2, 1, 0)),
            # Empty 3D tensor represented with mixed list/tuple nesting
            (([[]], [[]]), (2, 1, 0)),
            # Empty 4D tensor represented with nested lists
            ([[[[]]]], (1, 1, 1, 0)),
            # Empty 4D tensor represented with nested tuples/lists
            ((([[]],),), (1, 1, 1, 0)),
        ]
        for data, expected_shape in test_cases:
            with self.subTest(data=data):
                tensor = backend.to_tensor(data)
                self.assertEqual(backend.shape(tensor), expected_shape)

    def test_to_tensor_rejects_ragged_input(self):
        backend = self.make_backend()
        ragged_inputs = [
            [[1.0, 2.0], [3.0]],
            ((1.0, 2.0), (3.0,)),
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]],
            (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0),)),
        ]

        for data in ragged_inputs:
            with self.subTest(data=data):
                with self.assertRaises(
                    ValueError,
                    msg="to_tensor accepted ragged input when it should reject it",
                ):
                    backend.to_tensor(data)

    def test_to_tensor_rejects_inconsistent_nesting_depth(self):
        """
        Test that a structure containing values at inconsistent nesting
        depths is rejected rather than assigned an ambiguous shape.
        """
        backend = self.make_backend()
        inconsistent_inputs = [
            [[], 1.0],
            [[(), 0.0], 1.0, 2.0],
            [[(1.0,), 0.0], 1.0, 2.0],
        ]

        for data in inconsistent_inputs:
            with self.subTest(data=data):
                with self.assertRaises(
                    ValueError,
                    msg=(
                        "to_tensor accepted input with inconsistent nesting depth "
                        "when it should reject it"
                    ),
                ):
                    backend.to_tensor(data)


@EnforceSharedNumericFixtures()
class BackendContractToTensorValueInputMixin(BackendContractBase):

    def test_to_tensor_preserves_scalar_value(self):
        backend = self.make_backend()
        cases = (
            (0.0, 0.0),
            (3.0, 3.0),
            (-4.0, -4.0),
            (0, 0.0),
            (3, 3.0),
            (-4, -4.0),
        )
        for data, expected in cases:
            with self.subTest(data=data):
                result = backend.to_python(backend.to_tensor(data))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)
