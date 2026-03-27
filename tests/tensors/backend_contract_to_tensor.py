"""Test classes for the creation of tensors from Python lists/tuples

The backend contract requires that all backend implementations have a method
(to_tensor) for converting a Python list/tuple into the native tensor
representation used by that backend (e.g. np.array in the case of NumPy).

This method must be able to handle nested lists/tuples, which are needed
to represent tensors with more than one dimension.

This module has several classes which, together, enforce the backend
contract for the to_tensor method.
"""

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import to_python


class BackendContractToTensorTypeInputMixin(BackendContractBase):
    def test_to_tensor_accepts_list_input(self):
        backend = self.make_backend()
        backend.to_tensor([1, 2, 3])

    def test_to_tensor_accepts_tuple_input(self):
        backend = self.make_backend()
        backend.to_tensor((1, 2, 3))

    def test_to_tensor_rejects_plain_scalar_values(self):
        backend = self.make_backend()
        for data in (1, 1.5):
            with self.subTest(data=data):
                with self.assertRaises(
                    ValueError,
                    msg="to_tensor accepted a plain scalar value when it should reject it",
                ):
                    backend.to_tensor(data)

    def test_to_tensor_rejects_non_numeric_input(self):
        backend = self.make_backend()
        non_numeric_inputs = [
            ["a", "b"],
            [[1, 2], ["a", "b"]],
            [1, None, 3],
            [False],
            [0, True],
            ((1, 2), (3, "x")),
        ]

        for data in non_numeric_inputs:
            with self.subTest(data=data):
                with self.assertRaises(
                    (TypeError, ValueError),
                    msg="to_tensor accepted non-numeric input when it should reject it",
                ):
                    backend.to_tensor(data)


class BackendContractToTensorShapeInputMixin(BackendContractBase):

    def test_to_tensor_converts_1D_input_to_tensor(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor([1, 2, 3]))
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_to_tensor_converts_2D_input_to_tensor(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor(((1, 2), (3, 4))))
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_to_tensor_converts_3D_input_to_tensor(self):
        backend = self.make_backend()
        result = to_python(
            backend.to_tensor(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                ]
            )
        )
        self.assertEqual(
            result,
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        )

    def test_to_tensor_rejects_ragged_input(self):
        backend = self.make_backend()
        ragged_inputs = [
            [[1, 2], [3]],
            ((1, 2), (3,)),
            [[[1, 2], [3, 4]], [[5, 6]]],
            (((1, 2), (3, 4)), ((5, 6),)),
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
        Test that the backends do not try to treat empty lists/tuples
        or those woth a single value as valid. What is tested here is
        partly about structure (shape) and partly about type.
        """
        backend = self.make_backend()
        inconsistent_inputs = [
            [[], 1],
            [[(), 0], 1, 2],
            [[(1,), 0], 1, 2],
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


class BackendContractToTensorValueMixin(BackendContractBase):
    """
    All tensor representations are required to use floats but
    to_tensor should convert ints to floats when used to create
    new tensors.
    """

    def test_to_tensor_converts_integer_values_to_float(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor([1, 2, 3]))
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_to_tensor_preserves_float_values(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor([1.5, 2.5, 3.5]))
        self.assertEqual(result, [1.5, 2.5, 3.5])

    def test_to_tensor_normalises_mixed_numeric_input_to_float(self):
        backend = self.make_backend()
        result = to_python(backend.to_tensor([1, 2.5, 3]))
        self.assertEqual(result, [1.0, 2.5, 3.0])
