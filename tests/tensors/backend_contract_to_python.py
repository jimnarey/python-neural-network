"""Test classes for the conversion of native tensors to Python lists

The backend contract requires that all backend implementations have a method
(to_python) for converting a tensor in the native representation used by that
backend to a Python list structure containing scalar values represented
by built-in Python types.

This module has several classes which, together, enforce the backend
contract for the to_python method.

The shared to_python tests in this module cover only behaviour which can
be expressed and checked using plain Python list structures.
This includes, for example, ordinary 1D/2D/3D tensors and those empty
tensors whose structure is still visible in Python, such as [] and
[[], []].

This means that we need complementary tests at the implementation level
for each backend. These can inspect the backend's native tensor
representation before conversion to lists and more meaningfully
compare the input (native) and expected (Python) values/tensors.

This is particularly important for empty tensors. Some empty shapes,
such as (2, 0, 3), are valid but collapse to the same Python
representation as other shapes with empty dimensions.

By necessity, the tests here rely on a round-trip (i.e. create a tensor
with to_tensor and check what we get when we call to_python on it). For
this reason the round-trip tests in the shared to_tensor test module
have been kept light.
"""

from tests.tensors.backend_contract_shared import BackendContractBase


class BackendContractToPythonMixin(BackendContractBase):
    def test_to_python_converts_1D_tensor_to_list(self):
        backend = self.make_backend()
        result = backend.to_python(backend.to_tensor([1.0, 2.0, 3.0]))
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_to_python_converts_empty_1D_tensor_to_list(self):
        backend = self.make_backend()
        test_cases = [
            [],
            (),
        ]

        for data in test_cases:
            with self.subTest(data=data):
                result = backend.to_python(backend.to_tensor(data))
                self.assertEqual(result, [])

    def test_to_python_converts_2D_tensor_to_nested_list(self):
        backend = self.make_backend()
        result = backend.to_python(
            backend.to_tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_to_python_converts_3D_tensor_to_nested_list(self):
        backend = self.make_backend()
        result = backend.to_python(
            backend.to_tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
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

    def test_to_python_converts_4D_tensor_to_nested_list(self):
        backend = self.make_backend()
        result = backend.to_python(
            backend.to_tensor(
                [
                    [
                        [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    ],
                    [
                        [
                            [7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0],
                        ]
                    ],
                ]
            )
        )
        self.assertEqual(
            result,
            [
                [
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ]
                ],
                [
                    [
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ]
                ],
            ],
        )

    def test_to_python_returns_nested_list_containers(self):
        backend = self.make_backend()
        result = backend.to_python(
            backend.to_tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            )
        )

        self.assertIs(type(result), list)
        self.assertIs(type(result[0]), list)
        self.assertIs(type(result[0][0]), list)

    def test_to_python_converts_empty_tensors_to_nested_list(self):
        """
        Some empty tensor shapes cannot be distinguished once they have
        been converted to nested Python lists. For example, shapes
        (2, 0) and (2, 0, 3) both contain two empty outer elements and
        therefore both convert to [[], []]. Similarly, any shape whose
        first dimension is 0, such as (0,), (0, 1), or (0, 2, 3),
        converts to [].

        This test therefore checks only the Python-visible structure
        produced by representable empty inputs. It does not try to prove
        that all shape information survives conversion to Python lists.
        Tests which need to distinguish empty tensor shapes must inspect
        the backend-native tensor shape separately.
        """
        backend = self.make_backend()
        test_cases = [
            # Empty 2D tensor represented with nested lists
            ([[], []], [[], []]),
            # Empty 2D tensor represented with nested tuples
            (((), ()), [[], []]),
            # Empty 2D tensor represented with mixed list/tuple nesting
            ([(), ()], [[], []]),
            # Empty 2D tensor represented with mixed tuple/list nesting
            (([], []), [[], []]),
            # Empty 3D tensor represented with nested lists
            ([[[]], [[]]], [[[]], [[]]]),
            # Empty 3D tensor represented with mixed list/tuple nesting
            (([[]], [[]]), [[[]], [[]]]),
            # Empty 4D tensor represented with nested lists
            ([[[[]]]], [[[[]]]]),
            # Empty 4D tensor represented with nested tuples/lists
            ((([[]],),), [[[[]]]]),
        ]

        for data, expected in test_cases:
            with self.subTest(data=data):
                result = backend.to_python(backend.to_tensor(data))
                self.assertEqual(result, expected)

    def test_to_python_returns_builtin_python_scalar_values(self):
        backend = self.make_backend()
        test_cases = [
            [1.0, 2.0, 3.0],
            [1, 2, 3],
        ]

        for data in test_cases:
            with self.subTest(data=data):
                result = backend.to_python(backend.to_tensor(data))
                self.assertIs(type(result), list)
                self.assertEqual(len(result), len(data))
                for value in result:
                    self.assertIn(type(value), (int, float))
