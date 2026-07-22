"""Test classes for tensor creation methods (contract)

These test classes enforce the contract for those methods which create
new tensors according to a specification (as distinct from converting
an existing structure into a tensor as to_tensor does).
"""

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures


@EnforceSharedNumericFixtures()
class BackendContractZerosOnesAndFullMixin(BackendContractBase):
    def test_zeros_ones_and_full_return_requested_1D_tensors_with_expected_values(
        self,
    ):
        backend = self.make_backend()
        requested_shape = (3,)
        creation_methods = [
            ("zeros", lambda: backend.zeros(requested_shape), [0.0, 0.0, 0.0]),
            ("ones", lambda: backend.ones(requested_shape), [1.0, 1.0, 1.0]),
            ("full", lambda: backend.full(requested_shape, 7.0), [7.0, 7.0, 7.0]),
        ]

        for method_name, call, expected in creation_methods:
            with self.subTest(method=method_name):
                tensor = call()
                result = backend.to_python(tensor)
                self.assertEqual(backend.shape(tensor), requested_shape)
                self.assertEqual(result, expected)

    def test_zeros_ones_and_full_return_requested_2D_tensors_with_expected_values(
        self,
    ):
        backend = self.make_backend()
        requested_shape = (2, 3)
        creation_methods = [
            (
                "zeros",
                lambda: backend.zeros(requested_shape),
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ),
            (
                "ones",
                lambda: backend.ones(requested_shape),
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ),
            (
                "full",
                lambda: backend.full(requested_shape, 7.0),
                [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]],
            ),
        ]

        for method_name, call, expected in creation_methods:
            with self.subTest(method=method_name):
                tensor = call()
                result = backend.to_python(tensor)
                self.assertEqual(backend.shape(tensor), requested_shape)
                self.assertEqual(result, expected)

    def test_zeros_ones_and_full_return_requested_3D_tensors_with_expected_values(
        self,
    ):
        """
        Ensure that any backends are generalised and do not treat the easier cases in respect of
        number of dimensions as special.
        """
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                (2, 1, 4),
                {
                    "zeros": [
                        [[0.0, 0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0]],
                    ],
                    "ones": [
                        [[1.0, 1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0, 1.0]],
                    ],
                    "full": [
                        [[7.0, 7.0, 7.0, 7.0]],
                        [[7.0, 7.0, 7.0, 7.0]],
                    ],
                },
            ),
            (
                "larger_middle_dimension",
                (2, 3, 2),
                {
                    "zeros": [
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    "ones": [
                        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    ],
                    "full": [
                        [[7.0, 7.0], [7.0, 7.0], [7.0, 7.0]],
                        [[7.0, 7.0], [7.0, 7.0], [7.0, 7.0]],
                    ],
                },
            ),
        ]

        for case_name, requested_shape, expected_by_method in test_cases:
            creation_methods = [
                ("zeros", lambda: backend.zeros(requested_shape)),
                ("ones", lambda: backend.ones(requested_shape)),
                ("full", lambda: backend.full(requested_shape, 7.0)),
            ]

            for method_name, call in creation_methods:
                with self.subTest(case=case_name, method=method_name):
                    tensor = call()
                    result = backend.to_python(tensor)
                    self.assertEqual(backend.shape(tensor), requested_shape)
                    self.assertEqual(result, expected_by_method[method_name])

    def test_full_accepts_int_fill_value(self):
        backend = self.make_backend()
        tensor = backend.full((2, 3), 7)
        result = backend.to_python(tensor)
        for row in result:
            for value in row:
                self.assertEqual(value, 7.0)


@EnforceSharedNumericFixtures()
class BackendContractLikeCreationMixin(BackendContractBase):
    """
    The value-returning *_like methods each take a tensor as an argument and
    return a tensor with the same shape but with zeros, ones or a specified value.
    """

    def test_like_creation_methods_return_expected_values_with_same_shape_as_input_for_1D_tensor(
        self,
    ):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([1.0, 2.0, 3.0])
        creation_methods = [
            (
                "zeros_like",
                lambda: backend.zeros_like(source_tensor),
                [0.0, 0.0, 0.0],
            ),
            (
                "ones_like",
                lambda: backend.ones_like(source_tensor),
                [1.0, 1.0, 1.0],
            ),
            (
                "full_like",
                lambda: backend.full_like(source_tensor, 7.0),
                [7.0, 7.0, 7.0],
            ),
        ]

        for method_name, call, expected in creation_methods:
            with self.subTest(method=method_name):
                tensor = call()
                result = backend.to_python(tensor)
                self.assertEqual(backend.shape(tensor), (3,))
                self.assertEqual(result, expected)

    def test_like_creation_methods_return_expected_values_with_same_shape_as_input_for_2D_tensor(
        self,
    ):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        creation_methods = [
            (
                "zeros_like",
                lambda: backend.zeros_like(source_tensor),
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ),
            (
                "ones_like",
                lambda: backend.ones_like(source_tensor),
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ),
            (
                "full_like",
                lambda: backend.full_like(source_tensor, 7.0),
                [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]],
            ),
        ]

        for method_name, call, expected in creation_methods:
            with self.subTest(method=method_name):
                tensor = call()
                result = backend.to_python(tensor)
                self.assertEqual(backend.shape(tensor), (2, 3))
                self.assertEqual(result, expected)

    def test_like_creation_methods_return_expected_values_with_same_shape_as_input_for_3D_tensors(
        self,
    ):
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                backend.to_tensor(
                    [
                        [[1.0, 2.0, 3.0, 4.0]],
                        [[5.0, 6.0, 7.0, 8.0]],
                    ]
                ),
                (2, 1, 4),
                {
                    "zeros_like": [
                        [[0.0, 0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0, 0.0]],
                    ],
                    "ones_like": [
                        [[1.0, 1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0, 1.0]],
                    ],
                    "full_like": [
                        [[7.0, 7.0, 7.0, 7.0]],
                        [[7.0, 7.0, 7.0, 7.0]],
                    ],
                },
            ),
            (
                "larger_middle_dimension",
                backend.to_tensor(
                    [
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                    ]
                ),
                (2, 3, 2),
                {
                    "zeros_like": [
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    "ones_like": [
                        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    ],
                    "full_like": [
                        [[7.0, 7.0], [7.0, 7.0], [7.0, 7.0]],
                        [[7.0, 7.0], [7.0, 7.0], [7.0, 7.0]],
                    ],
                },
            ),
        ]

        for case_name, source_tensor, expected_shape, expected_by_method in test_cases:
            creation_methods = [
                ("zeros_like", lambda: backend.zeros_like(source_tensor)),
                ("ones_like", lambda: backend.ones_like(source_tensor)),
                ("full_like", lambda: backend.full_like(source_tensor, 7.0)),
            ]

            for method_name, call in creation_methods:
                with self.subTest(case=case_name, method=method_name):
                    tensor = call()
                    result = backend.to_python(tensor)
                    self.assertEqual(backend.shape(tensor), expected_shape)
                    self.assertEqual(result, expected_by_method[method_name])

    def test_full_like_accepts_int_fill_value(self):
        """
        The backend contract does not stipulate whether the values returned by
        creation or other methods are float-valued or int-valued. But it does
        require that either an int or float can be used as an argument for
        full_like (and some other methods).

        An int is used here in the assertion for consistency but Python returns
        True for 7 == 7.0 so a float could have been used just as well. We're
        not testing the type of the return value.
        """
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor = backend.full_like(source_tensor, 7)
        result = backend.to_python(tensor)
        self.assertEqual(backend.shape(tensor), (2, 3))
        for row in result:
            for value in row:
                self.assertEqual(value, 7)


@EnforceSharedNumericFixtures()
class BackendContractEmptyMixin(BackendContractBase):
    """
    This tests that empty returns tensors with the requested shapes.

    The values are not tested. In NumPy, which we are using as the reference
    implementation, empty creates an uninitialised array, so the values seen
    when we inspect the result may simply reflect whatever was already present
    in the allocated memory. Other backends may choose different placeholder
    values. The contract only guarantees the shape.
    """

    def test_empty_returns_tensor_with_requested_1D_shape(self):
        backend = self.make_backend()
        tensor = backend.empty((3,))
        self.assertEqual(backend.shape(tensor), (3,))

    def test_empty_returns_tensor_with_requested_2D_shape(self):
        backend = self.make_backend()
        tensor = backend.empty((2, 3))
        self.assertEqual(backend.shape(tensor), (2, 3))

    def test_empty_returns_3D_tensor_with_requested_shape(
        self,
    ):
        backend = self.make_backend()
        requested_shapes = [
            ("singleton_dimension", (2, 1, 4)),
            ("larger_middle_dimension", (2, 3, 2)),
        ]

        for case_name, requested_shape in requested_shapes:
            with self.subTest(case=case_name):
                tensor = backend.empty(requested_shape)
                self.assertEqual(backend.shape(tensor), requested_shape)

    def test_empty_like_returns_tensor_with_same_shape_as_input(self):
        backend = self.make_backend()
        source_tensor = backend.to_tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[5.0, 6.0, 7.0, 8.0]],
            ]
        )
        tensor = backend.empty_like(source_tensor)
        self.assertEqual(backend.shape(tensor), (2, 1, 4))


@EnforceSharedNumericFixtures()
class BackendContractCopyMixin(BackendContractBase):
    def test_copy_returns_tensor_with_same_shape_as_input(self):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor = backend.copy(source_tensor)
        self.assertEqual(backend.shape(tensor), (2, 3))

    def test_copy_returns_tensor_with_same_values_as_input(self):
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor = backend.copy(source_tensor)
        result = backend.to_python(tensor)
        self.assertEqual(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_copy_does_not_return_the_same_tensor_object(self):
        """
        This checks that when we copy a tensor we are not getting back the same
        Python object. A pass does not denote true independence. We could be dealing
        with a different object which ultimately points to the same region of
        memory. However, if it fails we can be sure we do not have a true copy.

        Unless we later add methods for tensor mutation into the backend contract
        we need implementation level tests for true independence.
        """
        backend = self.make_backend()
        source_tensor = backend.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tensor = backend.copy(source_tensor)
        self.assertNotEqual(id(tensor), id(source_tensor))


@EnforceSharedNumericFixtures()
class BackendContractEyeMixin(BackendContractBase):
    """
    eye creates an identity matrix.

    An identity matrix has 1s on the diagonal beginning at the top left
    and 0s everywhere else. It is called an identity matrix because, in
    matrix multiplication, it acts like the number 1 does in ordinary
    multiplication: multiplying another matrix by it leaves that matrix
    unchanged.

    For example, if:

        A = [
            [3, 5],
            [7, 11],
        ]

    and:

        I = [
            [1, 0],
            [0, 1],
        ]

    then:

        A @ I = [
            [3, 5],
            [7, 11],
        ]

    and:

        I @ A = [
            [3, 5],
            [7, 11],
        ]

    - top left: 3*1 + 5*0 = 3
    - top right: 3*0 + 5*1 = 5
    - bottom left: 7*1 + 11*0 = 7
    - bottom right: 7*0 + 11*1 = 11
    """

    def test_eye_returns_square_identity_matrix(self):
        backend = self.make_backend()
        tensor = backend.eye(3)
        result = backend.to_python(tensor)
        self.assertEqual(
            result,
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_eye_returns_rectangular_identity_matrix(self):
        backend = self.make_backend()
        test_cases = [
            (
                "two_by_three",
                2,
                3,
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            ),
            (
                "three_by_two",
                3,
                2,
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ],
            ),
            (
                "one_by_four",
                1,
                4,
                [
                    [1.0, 0.0, 0.0, 0.0],
                ],
            ),
        ]

        for case_name, n, m, expected in test_cases:
            with self.subTest(case=case_name):
                tensor = backend.eye(n, m)
                result = backend.to_python(tensor)
                self.assertEqual(result, expected)


@EnforceSharedNumericFixtures()
class BackendContractCreationZeroLengthDimensionMixin(BackendContractBase):
    def test_shape_based_creation_methods_support_zero_length_dimensions(self):
        backend = self.make_backend(seed=0)
        requested_shapes = [
            ("one_dimensional_zero_length", (0,)),
            ("two_dimensional_zero_length_trailing", (2, 0)),
            ("two_dimensional_zero_length_leading", (0, 3)),
            ("three_dimensional_zero_length_middle", (2, 0, 4)),
            ("three_dimensional_multiple_zero_lengths", (0, 2, 0)),
        ]

        for case_name, requested_shape in requested_shapes:
            creation_methods = [
                ("randn", lambda: backend.randn(requested_shape)),
                ("zeros", lambda: backend.zeros(requested_shape)),
                ("ones", lambda: backend.ones(requested_shape)),
                ("full", lambda: backend.full(requested_shape, 7.0)),
                ("empty", lambda: backend.empty(requested_shape)),
            ]

            for method_name, call in creation_methods:
                with self.subTest(case=case_name, method=method_name):
                    tensor = call()
                    self.assertEqual(backend.shape(tensor), requested_shape)


@EnforceSharedNumericFixtures()
class BackendContractCreationInputValidationMixin(BackendContractBase):
    def test_shape_based_creation_methods_reject_empty_shape(self):
        backend = self.make_backend()

        creation_methods = [
            ("randn", lambda: backend.randn(())),
            ("zeros", lambda: backend.zeros(())),
            ("ones", lambda: backend.ones(())),
            ("full", lambda: backend.full((), 7.0)),
            ("empty", lambda: backend.empty(())),
        ]

        for method_name, call in creation_methods:
            with self.subTest(method=method_name):
                with self.assertRaises(
                    ValueError,
                    msg=f"{method_name} accepted an empty shape when it should reject it",
                ):
                    call()
