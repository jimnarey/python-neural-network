"""Test classes for the creation of tensors from Python lists/tuples

The backend contract requires that all backend implementations have a method
(to_tensor) for converting a Python list/tuple into the native tensor
representation used by that backend (e.g. np.array in the case of NumPy).

This method must be able to handle nested lists/tuples, which are needed
to represent tensors with more than one dimension.

This module has several classes which, together, enforce the backend
contract for the to_tensor method.

The shared to_tensor tests in this module cover only behaviour which can
be expressed and checked using plain Python list and tuple structures.
This includes, for example, ordinary 1D/2D/3D inputs and those empty
inputs whose structure is still visible in Python, such as [] and
[[], []].

This means that we need complementary tests at the implementation level
for each backend. These can inspect the backend's native tensor
representation directly without converting to lists/tuples.

This is particualaly important for empty tensors. Some empty shapes,
such as (2, 0, 3), are valid but collapse to the same Python representation
as other shapes with empty dimensions.

This module is intentionally light on round-trip tests (i.e. create a tensor
with to_tensor and check what we get when we call to_python on it). The
shared tests for to_python necessarily work this way and will catch various
classes of problem with round-trip behaviour, including to_tensor mangling
the tensor creation in some way.
"""

from tests.tensors.backend_contract_shared import BackendContractBase


class BackendContractToTensorTypeInputMixin(BackendContractBase):
    """
    Invalid inputs should throw an exception and that is all we test
    for here. I.e. in the happy path cases we do not test what we get
    as a result of calling to_tensor (we can't here, as it's
    implementation-specifc).
    """

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

    def test_to_tensor_converts_empty_1D_input_to_tensor(self):
        """
        This uses the backend's to_python method, as well as calling
        to_tensor, to test the outcome of the round-trip. Accordingly
        it is really just a sense check. We need implementation-level
        tests to be sure we are getting the proper behaviour.
        """
        backend = self.make_backend()
        test_cases = [
            [],
            (),
        ]

        for data in test_cases:
            with self.subTest(data=data):
                result = backend.to_python(backend.to_tensor(data))
                self.assertEqual(result, [])

    def test_to_tensor_converts_empty_2D_input_to_tensor(self):
        """
        Relies on to_python as well as to_tensor.
        """
        backend = self.make_backend()
        test_cases = [
            [[], []],
            ((), ()),
            [(), ()],
            ([], []),
        ]

        for data in test_cases:
            with self.subTest(data=data):
                result = backend.to_python(backend.to_tensor(data))
                self.assertEqual(result, [[], []])

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
