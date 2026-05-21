"""
Tests for the unary tensor operations.

A unary operation takes a single tensor as input and returns a new tensor
whose shape matches the input shape.

The unary operations covered here each apply a rule to every value in the
input tensor independently:

- exp: replace each value x with e**x
- log: replace each positive value x with ln(x), i.e. log of x with
  base e (what power of e produces x).
- sqrt: replace each non-negative value x with its square root (the
  contract does not stipulate how negative values are handled)
- absolute: replace each value with its distance from zero
- sign: replace each value with -1.0, 0.0 or 1.0 according to whether it
  is negative, zero or positive (i.e. keep the value's sign but lose size)
- clip: force each value into a specified range, replacing values below
  the minimum with that minimum and values above the maximum with that
  maximum

e = 2.71828

The contract tests in this module cover the unary behaviour which all
backends must share. For sqrt, this includes the ordinary case where
all input values are non-negative and the expected results can be
expressed without relying on float-specific special values. For log and
sqrt, the contract tests do not define how values outside the ordinary
mathematical domain are handled. That more specific behaviour, including
the use of nan and inf at the Python boundary, is tested in the
reference test module instead. Some backends (e.g. inference only) may be
designed such that they never handle these cases at all.
"""

from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures
from tests.helpers.tensor_helpers import assert_nested_close

# At the current stage of the project we do not know how all of the unary
# operations will be  handled (if at all) in non-reference design backends.
# It may make sense to have all coverage of them in the reference tests
# eventually.


@EnforceSharedNumericFixtures()
class BackendContractUnaryShapeMixin(BackendContractBase):
    """
    Tests that we get the expected shape back from the three ops which can't be
    fully tested in the contract tests due to producing special, float-only values
    or non-integer floats in some cases.

    We test shape behaviour for the other ops in their own mixins so don't repeat
    that here.
    """

    def test_exp_and_log_return_1D_tensors_with_same_shape_as_input(self):
        backend = self.make_backend()
        input_tensors = {
            "exp": (backend.exp, backend.to_tensor([1.0, 2.0, 4.0])),
            "log": (backend.log, backend.to_tensor([1.0, 2.0, 4.0])),
        }

        for method_name, (method, tensor) in input_tensors.items():
            with self.subTest(method=method_name):
                result_tensor = method(tensor)
                self.assertEqual(backend.shape(result_tensor), (3,))

    def test_exp_and_log_return_2D_tensors_with_same_shape_as_input(self):
        backend = self.make_backend()
        input_tensors = {
            "exp": (backend.exp, backend.to_tensor([[1.0, 2.0], [4.0, 1.0]])),
            "log": (backend.log, backend.to_tensor([[1.0, 2.0], [4.0, 8.0]])),
        }

        for method_name, (method, tensor) in input_tensors.items():
            with self.subTest(method=method_name):
                result_tensor = method(tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 2))

    def test_exp_and_log_return_3D_tensors_with_same_shape_as_input(self):
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                {
                    "exp": backend.to_tensor([[[1.0, 2.0, 4.0]], [[1.0, 2.0, 4.0]]]),
                    "log": backend.to_tensor([[[1.0, 2.0, 4.0]], [[1.0, 2.0, 4.0]]]),
                },
                (2, 1, 3),
            ),
            (
                "larger_middle_dimension",
                {
                    "exp": backend.to_tensor(
                        [
                            [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]],
                            [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]],
                        ]
                    ),
                    "log": backend.to_tensor(
                        [
                            [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]],
                            [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]],
                        ]
                    ),
                },
                (2, 3, 2),
            ),
        ]

        for case_name, tensors, expected_shape in test_cases:
            input_tensors = {
                "exp": (backend.exp, tensors["exp"]),
                "log": (backend.log, tensors["log"]),
            }

            for method_name, (method, tensor) in input_tensors.items():
                with self.subTest(case=case_name, method=method_name):
                    result_tensor = method(tensor)
                    self.assertEqual(backend.shape(result_tensor), expected_shape)


@EnforceSharedNumericFixtures()
class BackendContractSqrtSemanticsMixin(BackendContractBase):
    """
    These tests cover only the ordinary case where all values are
    non-negative and the expected results are integer-valued. They do
    not test how backends handle negative inputs, since that behaviour
    is not defined by the backend contract.
    """

    def test_sqrt_returns_expected_values_for_1D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([0.0, 1.0, 4.0])
        result_tensor = backend.sqrt(tensor)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (3,))
        assert_nested_close(result, [0.0, 1.0, 2.0], rel_tol=0, abs_tol=0)

    def test_sqrt_returns_expected_values_for_2D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[0.0, 1.0], [4.0, 9.0]])
        result_tensor = backend.sqrt(tensor)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (2, 2))
        assert_nested_close(result, [[0.0, 1.0], [2.0, 3.0]], rel_tol=0, abs_tol=0)

    def test_sqrt_returns_expected_3D_tensors(self):
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                backend.to_tensor(
                    [
                        [[1.0, 4.0, 9.0]],
                        [[16.0, 25.0, 36.0]],
                    ]
                ),
                [
                    [[1.0, 2.0, 3.0]],
                    [[4.0, 5.0, 6.0]],
                ],
                (2, 1, 3),
            ),
            (
                "larger_middle_dimension",
                backend.to_tensor(
                    [
                        [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]],
                        [[49.0, 64.0], [81.0, 100.0], [121.0, 144.0]],
                    ]
                ),
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                ],
                (2, 3, 2),
            ),
        ]

        for case_name, tensor, expected, expected_shape in test_cases:
            result_tensor = backend.sqrt(tensor)
            result = backend.to_python(result_tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), expected_shape)
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)


@EnforceSharedNumericFixtures()
class BackendContractAbsoluteSemanticsMixin(BackendContractBase):
    """
    Test shape and values returned by absolute.
    """

    def test_absolute_returns_expected_1D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([-3.0, 0.0, 2.0])
        result_tensor = backend.absolute(tensor)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (3,))
        assert_nested_close(result, [3.0, 0.0, 2.0], rel_tol=0, abs_tol=0)

    def test_absolute_returns_expected_2D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[-3.0, 0.0], [2.0, -4.0]])
        result_tensor = backend.absolute(tensor)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (2, 2))
        assert_nested_close(
            result,
            [[3.0, 0.0], [2.0, 4.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_absolute_returns_expected_3D_tensors(self):
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                backend.to_tensor([[[-3.0, 0.0, 2.0]], [[-4.0, 5.0, -6.0]]]),
                [[[3.0, 0.0, 2.0]], [[4.0, 5.0, 6.0]]],
                (2, 1, 3),
            ),
            (
                "larger_middle_dimension",
                backend.to_tensor(
                    [
                        [[-3.0, 0.0], [2.0, -4.0], [5.0, -6.0]],
                        [[-7.0, 8.0], [-9.0, 10.0], [11.0, -12.0]],
                    ]
                ),
                [
                    [[3.0, 0.0], [2.0, 4.0], [5.0, 6.0]],
                    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                ],
                (2, 3, 2),
            ),
        ]

        for case_name, tensor, expected, expected_shape in test_cases:
            result_tensor = backend.absolute(tensor)
            result = backend.to_python(result_tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), expected_shape)
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)


@EnforceSharedNumericFixtures()
class BackendContractSignSemanticsMixin(BackendContractBase):
    """
    Test shape and values returned by sign.
    """

    def test_sign_returns_expected_1D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([-3.0, 0.0, 2.0])
        result_tensor = backend.sign(tensor)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (3,))
        assert_nested_close(
            result,
            [-1.0, 0.0, 1.0],
            rel_tol=0,
            abs_tol=0,
        )

    def test_sign_returns_expected_2D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[-3.0, 0.0], [2.0, -4.0]])
        result_tensor = backend.sign(tensor)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (2, 2))
        assert_nested_close(
            result,
            [[-1.0, 0.0], [1.0, -1.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_sign_returns_expected_3D_tensors(self):
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                backend.to_tensor([[[-3.0, 0.0, 2.0]], [[-4.0, 5.0, -6.0]]]),
                [[[-1.0, 0.0, 1.0]], [[-1.0, 1.0, -1.0]]],
                (2, 1, 3),
            ),
            (
                "larger_middle_dimension",
                backend.to_tensor(
                    [
                        [[-3.0, 0.0], [2.0, -4.0], [5.0, -6.0]],
                        [[-7.0, 8.0], [-9.0, 10.0], [11.0, -12.0]],
                    ]
                ),
                [
                    [[-1.0, 0.0], [1.0, -1.0], [1.0, -1.0]],
                    [[-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]],
                ],
                (2, 3, 2),
            ),
        ]

        for case_name, tensor, expected, expected_shape in test_cases:
            result_tensor = backend.sign(tensor)
            result = backend.to_python(result_tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), expected_shape)
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)


@EnforceSharedNumericFixtures()
class BackendContractClipSemanticsMixin(BackendContractBase):
    """
    Test shape and values returned by clip.
    """

    def test_clip_returns_expected_1D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([-2.0, 1.0, 5.0])
        result_tensor = backend.clip(tensor, 0.0, 3.0)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (3,))
        assert_nested_close(result, [0.0, 1.0, 3.0], rel_tol=0, abs_tol=0)

    def test_clip_returns_expected_2D_tensor(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([[-2.0, 1.0], [5.0, 3.0]])
        result_tensor = backend.clip(tensor, 0.0, 3.0)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (2, 2))
        assert_nested_close(
            result,
            [[0.0, 1.0], [3.0, 3.0]],
            rel_tol=0,
            abs_tol=0,
        )

    def test_clip_returns_expected_3D_tensors(self):
        backend = self.make_backend()
        test_cases = [
            (
                "singleton_dimension",
                backend.to_tensor([[[-2.0, 1.0, 5.0]], [[4.0, -1.0, 3.0]]]),
                [[[0.0, 1.0, 3.0]], [[3.0, 0.0, 3.0]]],
                (2, 1, 3),
            ),
            (
                "larger_middle_dimension",
                backend.to_tensor(
                    [
                        [[-2.0, 1.0], [5.0, 3.0], [-4.0, 2.0]],
                        [[4.0, -1.0], [3.0, 6.0], [0.0, 8.0]],
                    ]
                ),
                [
                    [[0.0, 1.0], [3.0, 3.0], [0.0, 2.0]],
                    [[3.0, 0.0], [3.0, 3.0], [0.0, 3.0]],
                ],
                (2, 3, 2),
            ),
        ]

        for case_name, tensor, expected, expected_shape in test_cases:
            result_tensor = backend.clip(tensor, 0.0, 3.0)
            result = backend.to_python(result_tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), expected_shape)
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_clip_accepts_int_bounds(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([-2.0, 1.0, 5.0])
        result_tensor = backend.clip(tensor, 0, 3)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (3,))
        assert_nested_close(result, [0.0, 1.0, 3.0], rel_tol=0, abs_tol=0)

    def test_clip_accepts_float_bounds(self):
        backend = self.make_backend()
        tensor = backend.to_tensor([-2.0, 1.0, 5.0])
        result_tensor = backend.clip(tensor, 0.0, 3.0)
        result = backend.to_python(result_tensor)
        self.assertEqual(backend.shape(result_tensor), (3,))
        assert_nested_close(result, [0.0, 1.0, 3.0], rel_tol=0, abs_tol=0)


@EnforceSharedNumericFixtures()
class BackendContractUnaryZeroLengthDimensionMixin(BackendContractBase):

    def test_exp_and_log_return_3D_tensors_with_same_shape_as_input_when_one_dimension_is_0(
        self,
    ):
        backend = self.make_backend()
        test_cases = [
            ("leading_zero", (0, 2, 3)),
            ("middle_zero", (2, 0, 3)),
            ("trailing_zero", (2, 3, 0)),
        ]

        for case_name, shape in test_cases:
            input_tensors = {
                "exp": (backend.exp, backend.ones(shape)),
                "log": (backend.log, backend.ones(shape)),
            }

            for method_name, (method, tensor) in input_tensors.items():
                with self.subTest(case=case_name, method=method_name):
                    result_tensor = method(tensor)
                    self.assertEqual(backend.shape(result_tensor), shape)

    def test_sqrt_returns_3D_tensors_with_same_shape_as_input_when_one_dimension_is_0(
        self,
    ):
        backend = self.make_backend()
        test_cases = [
            ("leading_zero", (0, 2, 3)),
            ("middle_zero", (2, 0, 3)),
            ("trailing_zero", (2, 3, 0)),
        ]

        for case_name, shape in test_cases:
            tensor = backend.ones(shape)
            result_tensor = backend.sqrt(tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), shape)

    def test_absolute_returns_3D_tensors_with_same_shape_as_input_when_one_dimension_is_0(
        self,
    ):
        backend = self.make_backend()
        test_cases = [
            ("leading_zero", (0, 2, 3)),
            ("middle_zero", (2, 0, 3)),
            ("trailing_zero", (2, 3, 0)),
        ]

        for case_name, shape in test_cases:
            tensor = backend.ones(shape)
            result_tensor = backend.absolute(tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), shape)

    def test_sign_returns_3D_tensors_with_same_shape_as_input_when_one_dimension_is_0(
        self,
    ):
        backend = self.make_backend()
        test_cases = [
            ("leading_zero", (0, 2, 3)),
            ("middle_zero", (2, 0, 3)),
            ("trailing_zero", (2, 3, 0)),
        ]

        for case_name, shape in test_cases:
            tensor = backend.ones(shape)
            result_tensor = backend.sign(tensor)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), shape)

    def test_clip_returns_3D_tensors_with_same_shape_as_input_when_one_dimension_is_0(
        self,
    ):
        backend = self.make_backend()
        test_cases = [
            ("leading_zero", (0, 2, 3)),
            ("middle_zero", (2, 0, 3)),
            ("trailing_zero", (2, 3, 0)),
        ]

        for case_name, shape in test_cases:
            tensor = backend.ones(shape)
            result_tensor = backend.clip(tensor, 0.0, 3.0)
            with self.subTest(case=case_name):
                self.assertEqual(backend.shape(result_tensor), shape)
