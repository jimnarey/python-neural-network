from tests.tensors.backend_contract_shared import BackendContractBase
from tests.helpers.tensor_assertions import assert_nested_close
from tests.helpers.shared_tests_enforcement import EnforceSharedNumericFixtures


@EnforceSharedNumericFixtures()
class BackendContractElementwiseSemanticsMixin(BackendContractBase):
    def test_elementwise_methods_apply_elementwise_to_same_shape_1D_tensors(self):
        backend = self.make_backend()
        a = backend.to_tensor([2.0, 6.0, 12.0])
        b = backend.to_tensor([1.0, 3.0, 4.0])

        elementwise_methods = [
            ("add", backend.add, [3.0, 9.0, 16.0]),
            ("subtract", backend.subtract, [1.0, 3.0, 8.0]),
            ("multiply", backend.multiply, [2.0, 18.0, 48.0]),
            ("divide", backend.divide, [2.0, 2.0, 3.0]),
            ("maximum", backend.maximum, [2.0, 6.0, 12.0]),
            ("minimum", backend.minimum, [1.0, 3.0, 4.0]),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (3,))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_apply_elementwise_to_same_shape_2D_tensors(self):
        backend = self.make_backend()
        a = backend.to_tensor([[2.0, 6.0], [12.0, 20.0]])
        b = backend.to_tensor([[1.0, 3.0], [4.0, 5.0]])

        elementwise_methods = [
            ("add", backend.add, [[3.0, 9.0], [16.0, 25.0]]),
            ("subtract", backend.subtract, [[1.0, 3.0], [8.0, 15.0]]),
            ("multiply", backend.multiply, [[2.0, 18.0], [48.0, 100.0]]),
            ("divide", backend.divide, [[2.0, 2.0], [3.0, 4.0]]),
            ("maximum", backend.maximum, [[2.0, 6.0], [12.0, 20.0]]),
            ("minimum", backend.minimum, [[1.0, 3.0], [4.0, 5.0]]),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 2))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_apply_elementwise_to_same_shape_3D_tensors(self):
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[2.0, 6.0], [12.0, 20.0]],
                [[4.0, 8.0], [10.0, 18.0]],
            ]
        )
        b = backend.to_tensor(
            [
                [[1.0, 3.0], [4.0, 5.0]],
                [[2.0, 4.0], [5.0, 6.0]],
            ]
        )

        elementwise_methods = [
            (
                "add",
                backend.add,
                [
                    [[3.0, 9.0], [16.0, 25.0]],
                    [[6.0, 12.0], [15.0, 24.0]],
                ],
            ),
            (
                "subtract",
                backend.subtract,
                [
                    [[1.0, 3.0], [8.0, 15.0]],
                    [[2.0, 4.0], [5.0, 12.0]],
                ],
            ),
            (
                "multiply",
                backend.multiply,
                [
                    [[2.0, 18.0], [48.0, 100.0]],
                    [[8.0, 32.0], [50.0, 108.0]],
                ],
            ),
            (
                "divide",
                backend.divide,
                [
                    [[2.0, 2.0], [3.0, 4.0]],
                    [[2.0, 2.0], [2.0, 3.0]],
                ],
            ),
            (
                "maximum",
                backend.maximum,
                [
                    [[2.0, 6.0], [12.0, 20.0]],
                    [[4.0, 8.0], [10.0, 18.0]],
                ],
            ),
            (
                "minimum",
                backend.minimum,
                [
                    [[1.0, 3.0], [4.0, 5.0]],
                    [[2.0, 4.0], [5.0, 6.0]],
                ],
            ),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 2, 2))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_apply_elementwise_to_same_shape_4D_tensors(self):
        pass

    def test_elementwise_methods_accept_a_scalar_right_hand_operand(self):
        pass

    def test_elementwise_methods_accept_a_scalar_right_hand_operand_for_3D_tensor(
        self,
    ):
        pass

    def test_divide_raises_when_scalar_right_hand_operand_is_zero(self):
        pass

    def test_divide_raises_when_tensor_right_hand_operand_contains_zero(self):
        pass


@EnforceSharedNumericFixtures()
class BackendContractElementwiseBroadcastingMixin(BackendContractBase):
    """
    The tests in BackendContractElementwiseSemanticsMixin only use operands
    with identical shapes. The tests in this class cover what happens when
    the shapes differ.

    Elementwise operations work by pairing up values at the same position.
    When both operands have the same shape this is straightforward. When
    they have different shapes, it is only possible if the shapes are
    compatible for broadcasting.

    Broadcasting compatibility is checked by comparing the two shapes from
    right to left, one axis at a time. At each position, the axis lengths
    are compatible if they are equal, or if one of them is 1. When an axis
    has length 1 on one side but a longer length on the other, the shorter
    operand is reused along that axis — the same values are used for each
    position in the longer operand.

    The same rule extends to cases where one operand has fewer dimensions.
    The right-to-left comparison still applies, but the lower-rank operand
    runs out of axes first. The remaining axes — those present only in the
    higher-rank operand — are treated as though the lower-rank operand had
    extra leading axes of length 1.

    An extra leading axis of length 1 simply wraps the existing data in one
    more layer: the added axis contains only the operand itself as its
    single element. For example, a 2D operand of shape (2, 3):

        [1.0, 2.0, 3.0]
        [4.0, 5.0, 6.0]

    treated as shape (1, 2, 3) still contains the same two rows — the
    operand itself is now the single element along the added axis. During
    the operation, it is reused for every corresponding position in the
    other operand, exactly as a length-1 axis is reused in any other
    context. It does not matter how many fewer dimensions the lower-rank
    operand has.

    This is broader than the broadcasting in matmul, where only the leading
    axes (the stack length) can be broadcast. Here, any axis can be
    broadcast, as long as the rule is satisfied at every position.

    If it is not — i.e. if two axes at the same position have lengths
    that are neither equal nor one is 1 — the operation should  raise an
    exception.
    """

    def test_elementwise_methods_broadcast_when_an_end_axis_has_length_1(self):
        pass

    def test_elementwise_methods_broadcast_when_a_middle_axis_has_length_1(self):
        pass

    def test_elementwise_methods_raise_when_tensor_shapes_are_not_broadcast_compatible(
        self,
    ):
        """
        This tests that elementwise operations raise when the operand shapes
        are not compatible for broadcasting.

        The left-hand tensor a has shape (2, 3) and the right-hand tensor b
        has shape (2, 2). Comparing shapes from right to left, the rightmost
        axes have lengths 3 and 2. Because these lengths are neither equal
        nor one of them is 1, there is no valid way to pair the values at
        those positions.

        This means the operations cannot be carried out elementwise and
        should therefore raise an exception.
        """
        backend = self.make_backend()
        a = backend.to_tensor([[2.0, 6.0, 12.0], [4.0, 12.0, 16.0]])
        b = backend.to_tensor([[1.0, 3.0], [4.0, 5.0]])

        elementwise_methods = [
            ("add", backend.add),
            ("subtract", backend.subtract),
            ("multiply", backend.multiply),
            ("divide", backend.divide),
            ("maximum", backend.maximum),
            ("minimum", backend.minimum),
        ]

        for method_name, method in elementwise_methods:
            with self.subTest(method=method_name):
                with self.assertRaises(ValueError):
                    method(a, b)

    def test_elementwise_methods_broadcast_1D_tensor_across_2D_tensor(self):
        """
        When a 1D tensor is combined with a 2D tensor, the 1D tensor is reused
        for each row of the 2D tensor.

        The 2D operand a has shape (2, 3) and the 1D operand b has shape (3,).
        Comparing shapes from right to left, the rightmost axes (both 3) are
        equal. The 1D tensor then has no more axes; the remaining axis of a
        (length 2) is treated as though b had shape (1, 3), so b is reused once
        for each row.

        The 2D operand a is:

            [2.0,  6.0, 12.0]
            [4.0, 12.0, 16.0]

        The 1D operand b is:

            [1.0, 3.0, 4.0]

        For add, the first row of the result is b added to the first row of a:

            [2.0 + 1.0,  6.0 + 3.0, 12.0 + 4.0] = [3.0, 9.0, 16.0]

        b is then reused for the second row:

            [4.0 + 1.0, 12.0 + 3.0, 16.0 + 4.0] = [5.0, 15.0, 20.0]

        The result has shape (2, 3). All six elementwise methods are tested with
        the same operands.
        """
        backend = self.make_backend()
        a = backend.to_tensor([[2.0, 6.0, 12.0], [4.0, 12.0, 16.0]])
        b = backend.to_tensor([1.0, 3.0, 4.0])

        elementwise_methods = [
            ("add", backend.add, [[3.0, 9.0, 16.0], [5.0, 15.0, 20.0]]),
            ("subtract", backend.subtract, [[1.0, 3.0, 8.0], [3.0, 9.0, 12.0]]),
            ("multiply", backend.multiply, [[2.0, 18.0, 48.0], [4.0, 36.0, 64.0]]),
            ("divide", backend.divide, [[2.0, 2.0, 3.0], [4.0, 4.0, 4.0]]),
            ("maximum", backend.maximum, [[2.0, 6.0, 12.0], [4.0, 12.0, 16.0]]),
            ("minimum", backend.minimum, [[1.0, 3.0, 4.0], [1.0, 3.0, 4.0]]),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 3))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_broadcast_2D_tensor_across_3D_tensor(self):
        """
        When a 2D tensor is combined with a 3D tensor, the 2D tensor is reused
        for each position along the leading axis of the 3D tensor.

        The 3D operand a has shape (2, 2, 2) and the 2D operand b has shape
        (2, 2). Comparing shapes from right to left, the two rightmost axes
        match (both 2, then both 2). The 2D tensor then has no more axes; the
        remaining leading axis of a (length 2) is treated as though b had shape
        (1, 2, 2), so b is reused once for each position along that axis.

        The first 2D tensor in a is:

            [2.0,  6.0]
            [12.0, 20.0]

        The second 2D tensor in a is:

            [2.0,  9.0]
            [8.0,  15.0]

        The 2D operand b is:

            [1.0, 3.0]
            [4.0, 5.0]

        For add, b is added to the first 2D tensor in a:

            [2.0 + 1.0,  6.0 + 3.0]  = [3.0,  9.0]
            [12.0 + 4.0, 20.0 + 5.0] = [16.0, 25.0]

        b is then reused and added to the second 2D tensor in a:

            [2.0 + 1.0,  9.0 + 3.0]  = [3.0,  12.0]
            [8.0 + 4.0,  15.0 + 5.0] = [12.0, 20.0]

        The result has shape (2, 2, 2).
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[2.0, 6.0], [12.0, 20.0]],
                [[2.0, 9.0], [8.0, 15.0]],
            ]
        )
        b = backend.to_tensor([[1.0, 3.0], [4.0, 5.0]])

        elementwise_methods = [
            (
                "add",
                backend.add,
                [
                    [[3.0, 9.0], [16.0, 25.0]],
                    [[3.0, 12.0], [12.0, 20.0]],
                ],
            ),
            (
                "subtract",
                backend.subtract,
                [
                    [[1.0, 3.0], [8.0, 15.0]],
                    [[1.0, 6.0], [4.0, 10.0]],
                ],
            ),
            (
                "multiply",
                backend.multiply,
                [
                    [[2.0, 18.0], [48.0, 100.0]],
                    [[2.0, 27.0], [32.0, 75.0]],
                ],
            ),
            (
                "divide",
                backend.divide,
                [
                    [[2.0, 2.0], [3.0, 4.0]],
                    [[2.0, 3.0], [2.0, 3.0]],
                ],
            ),
            (
                "maximum",
                backend.maximum,
                [
                    [[2.0, 6.0], [12.0, 20.0]],
                    [[2.0, 9.0], [8.0, 15.0]],
                ],
            ),
            (
                "minimum",
                backend.minimum,
                [
                    [[1.0, 3.0], [4.0, 5.0]],
                    [[1.0, 3.0], [4.0, 5.0]],
                ],
            ),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 2, 2))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_broadcast_2D_tensor_across_4D_tensor(self):
        """
        This tests broadcasting a 2D tensor across a 4D tensor.

        Each of the previous broadcasting tests involving added dimensions
        only requires one extra dimension to be added to the lower-rank operand.
        This test requires two.

        This helps guard against future backend implementations accidentally
        treating 1D tensors, or tensors only one dimension smaller than the
        other operand, as special cases.

        Otherwise, the principle under test is the same: compare the shapes
        from right to left, then treat the lower-rank operand as though
        leading axes of length 1 had been added.
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [
                    [[2.0, 6.0], [12.0, 20.0]],
                    [[4.0, 12.0], [16.0, 25.0]],
                ],
                [
                    [[3.0, 9.0], [8.0, 15.0]],
                    [[5.0, 15.0], [24.0, 30.0]],
                ],
            ]
        )
        b = backend.to_tensor([[1.0, 3.0], [4.0, 5.0]])

        elementwise_methods = [
            (
                "add",
                backend.add,
                [
                    [
                        [[3.0, 9.0], [16.0, 25.0]],
                        [[5.0, 15.0], [20.0, 30.0]],
                    ],
                    [
                        [[4.0, 12.0], [12.0, 20.0]],
                        [[6.0, 18.0], [28.0, 35.0]],
                    ],
                ],
            ),
            (
                "subtract",
                backend.subtract,
                [
                    [
                        [[1.0, 3.0], [8.0, 15.0]],
                        [[3.0, 9.0], [12.0, 20.0]],
                    ],
                    [
                        [[2.0, 6.0], [4.0, 10.0]],
                        [[4.0, 12.0], [20.0, 25.0]],
                    ],
                ],
            ),
            (
                "multiply",
                backend.multiply,
                [
                    [
                        [[2.0, 18.0], [48.0, 100.0]],
                        [[4.0, 36.0], [64.0, 125.0]],
                    ],
                    [
                        [[3.0, 27.0], [32.0, 75.0]],
                        [[5.0, 45.0], [96.0, 150.0]],
                    ],
                ],
            ),
            (
                "divide",
                backend.divide,
                [
                    [
                        [[2.0, 2.0], [3.0, 4.0]],
                        [[4.0, 4.0], [4.0, 5.0]],
                    ],
                    [
                        [[3.0, 3.0], [2.0, 3.0]],
                        [[5.0, 5.0], [6.0, 6.0]],
                    ],
                ],
            ),
            (
                "maximum",
                backend.maximum,
                [
                    [
                        [[2.0, 6.0], [12.0, 20.0]],
                        [[4.0, 12.0], [16.0, 25.0]],
                    ],
                    [
                        [[3.0, 9.0], [8.0, 15.0]],
                        [[5.0, 15.0], [24.0, 30.0]],
                    ],
                ],
            ),
            (
                "minimum",
                backend.minimum,
                [
                    [
                        [[1.0, 3.0], [4.0, 5.0]],
                        [[1.0, 3.0], [4.0, 5.0]],
                    ],
                    [
                        [[1.0, 3.0], [4.0, 5.0]],
                        [[1.0, 3.0], [4.0, 5.0]],
                    ],
                ],
            ),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 2, 2, 2))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_broadcast_when_a_leading_axis_has_length_1(self):
        """
        This tests broadcasting where one operand already has an explicit leading
        axis of length 1.

        This is a slightly less natural case than broadcasting a lower-rank
        operand across a higher-rank one, because the length-1 axis is already
        present rather than being treated as implicitly added on the left.
        Even so, it is a real broadcasting case and could arise naturally after
        an operation such as reshape, or after using keepdims=True in a reduction.
        """

        backend = self.make_backend()
        a = backend.to_tensor([[[12.0, 24.0], [18.0, 20.0]]])
        b = backend.to_tensor(
            [
                [[3.0, 6.0], [6.0, 5.0]],
                [[2.0, 4.0], [3.0, 4.0]],
            ]
        )

        elementwise_methods = [
            (
                "add",
                backend.add,
                [
                    [[15.0, 30.0], [24.0, 25.0]],
                    [[14.0, 28.0], [21.0, 24.0]],
                ],
            ),
            (
                "subtract",
                backend.subtract,
                [
                    [[9.0, 18.0], [12.0, 15.0]],
                    [[10.0, 20.0], [15.0, 16.0]],
                ],
            ),
            (
                "multiply",
                backend.multiply,
                [
                    [[36.0, 144.0], [108.0, 100.0]],
                    [[24.0, 96.0], [54.0, 80.0]],
                ],
            ),
            (
                "divide",
                backend.divide,
                [
                    [[4.0, 4.0], [3.0, 4.0]],
                    [[6.0, 6.0], [6.0, 5.0]],
                ],
            ),
            (
                "maximum",
                backend.maximum,
                [
                    [[12.0, 24.0], [18.0, 20.0]],
                    [[12.0, 24.0], [18.0, 20.0]],
                ],
            ),
            (
                "minimum",
                backend.minimum,
                [
                    [[3.0, 6.0], [6.0, 5.0]],
                    [[2.0, 4.0], [3.0, 4.0]],
                ],
            ),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 2, 2))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_raise_when_higher_rank_tensor_shapes_are_not_broadcast_compatible(
        self,
    ):
        pass
