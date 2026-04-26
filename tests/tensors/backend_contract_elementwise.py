"""Elementwise backend contract tests

This module contains the backend contract tests for the six elementwise
binary methods: add, subtract, multiply, divide, maximum, and minimum.
NumPy's array operations are used as the reference implementation.

The simplest case is where both operands have the same shape. No
broadcasting is needed: each operation is applied position by position.

This module also covers the two variations of broadcasting which can
be applied when carrying out the elementwise operations on a pair of
tensors which would otherwise be incompatible. One of these
(left-padding) is analagous to broadcasting in matmul (which has only
one broadcasting mechanism).

When the operands have the same number of dimensions but the shapes
differ, an axis of length 1 on one side is broadcast to match the
longer axis on the other. If no such pairing is possible — because
two corresponding axes have different lengths and neither is 1 — the
operation must raise an exception.

When the operands have different numbers of dimensions, the lower-rank
operand is treated as though leading axes of length 1 had been prepended
to its shape before the right-to-left comparison begins. Because those
added axes all have length 1, the length-1-axis rule then applies to
each of them — unless the corresponding leading axis of the higher-rank
operand also has length 1, in which case the two axes are already equal
and no expansion is needed.

Left-padding therefore almost always involves both mechanisms acting
together, so the split in this module across different mixins is a
little artificial (but makes what is being tested a bit clearer).

Tests also cover the case where both mechanisms apply within a single
operation but to different operands, and the case where left-padding
does not rescue an incompatible axis pair in the shared suffix of the
shapes.
"""

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
class BackendContractElementwiseLengthOneAxisBroadcastingMixin(BackendContractBase):
    """
    The tests in BackendContractElementwiseSemanticsMixin only use operands
    with identical shapes. The tests in this class cover the case where both
    operands have the same number of dimensions, but the shapes are not
    identical.

    When the shapes differ, the operation is only possible if the shapes are
    compatible for broadcasting. Compatibility is checked by comparing the
    shapes from right to left, one axis at a time. At each position, the
    axis lengths are compatible if they are equal, or if one of them is 1.

    When an axis has length 1 on one side but a longer length on the other,
    the single value along that axis is reused for each position in the longer
    operand — as though the length-1 axis were expanded to match.

    If two axes at the same position have lengths that are neither equal nor
    one of them is 1, the shapes are not compatible and the operation should
    raise an exception.
    """

    def test_elementwise_methods_broadcast_when_an_end_axis_has_length_1(self):
        """
        This tests broadcasting where an end axis has length 1.

        The two operands have the same number of dimensions. Broadcasting
        is possible because the rightmost axis of a has length 1 while the
        rightmost axis of b has length 3.

        The tensor a has shape (2, 1) and the tensor b has shape (2, 3).
        Comparing shapes from right to left, the rightmost axes have
        lengths 1 and 3, so the single value in each row of a is repeated
        across that axis. The leftmost axes both have length 2, so they
        already match.

        The tensor a is:

            [12.0]
            [18.0]

        The tensor b is:

            [3.0, 4.0, 6.0]
            [2.0, 3.0, 6.0]

        For add, the first value in a, 12.0, is repeated across the first
        row and added to the first row of b:

            [12.0 + 3.0, 12.0 + 4.0, 12.0 + 6.0] = [15.0, 16.0, 18.0]

        The second value in a, 18.0, is repeated across the second row and
        added to the second row of b:

            [18.0 + 2.0, 18.0 + 3.0, 18.0 + 6.0] = [20.0, 21.0, 24.0]

        The result therefore has shape (2, 3). All six elementwise methods
        are tested with the same operands.
        """
        backend = self.make_backend()
        a = backend.to_tensor([[12.0], [18.0]])
        b = backend.to_tensor([[3.0, 4.0, 6.0], [2.0, 3.0, 6.0]])

        elementwise_methods = [
            ("add", backend.add, [[15.0, 16.0, 18.0], [20.0, 21.0, 24.0]]),
            ("subtract", backend.subtract, [[9.0, 8.0, 6.0], [16.0, 15.0, 12.0]]),
            ("multiply", backend.multiply, [[36.0, 48.0, 72.0], [36.0, 54.0, 108.0]]),
            ("divide", backend.divide, [[4.0, 3.0, 2.0], [9.0, 6.0, 3.0]]),
            ("maximum", backend.maximum, [[12.0, 12.0, 12.0], [18.0, 18.0, 18.0]]),
            ("minimum", backend.minimum, [[3.0, 4.0, 6.0], [2.0, 3.0, 6.0]]),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 3))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_broadcast_when_a_middle_axis_has_length_1(self):
        """
        This tests broadcasting where a middle axis has length 1.

        The tensor a has shape (2, 1, 2) and the tensor b has shape
        (2, 3, 2). Comparing shapes from right to left, the rightmost
        axes both have length 2, so they already match. The middle axes
        have lengths 1 and 3, so the single row in each 2D tensor in a is
        repeated three times to match the corresponding 2D tensor in b.
        The leftmost axes both have length 2, so they also match.

        The tensor a is:

            [[12.0, 24.0]]
            [[18.0, 30.0]]

        The tensor b is:

            [[3.0, 6.0], [4.0, 8.0], [6.0, 12.0]]
            [[2.0, 5.0], [3.0, 6.0], [6.0, 10.0]]

        For add, the only row in the first 2D tensor in a,
        [12.0, 24.0], is repeated three times and added to the first
        2D tensor in b:

            [12.0 + 3.0, 24.0 + 6.0] = [15.0, 30.0]
            [12.0 + 4.0, 24.0 + 8.0] = [16.0, 32.0]
            [12.0 + 6.0, 24.0 + 12.0] = [18.0, 36.0]

        The same happens for the second 2D tensor in a, whose only row
        [18.0, 30.0] is repeated across the three rows of the second
        2D tensor in b.

        The result therefore has shape (2, 3, 2).
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[12.0, 24.0]],
                [[18.0, 30.0]],
            ]
        )
        b = backend.to_tensor(
            [
                [[3.0, 6.0], [4.0, 8.0], [6.0, 12.0]],
                [[2.0, 5.0], [3.0, 6.0], [6.0, 10.0]],
            ]
        )

        elementwise_methods = [
            (
                "add",
                backend.add,
                [
                    [[15.0, 30.0], [16.0, 32.0], [18.0, 36.0]],
                    [[20.0, 35.0], [21.0, 36.0], [24.0, 40.0]],
                ],
            ),
            (
                "subtract",
                backend.subtract,
                [
                    [[9.0, 18.0], [8.0, 16.0], [6.0, 12.0]],
                    [[16.0, 25.0], [15.0, 24.0], [12.0, 20.0]],
                ],
            ),
            (
                "multiply",
                backend.multiply,
                [
                    [[36.0, 144.0], [48.0, 192.0], [72.0, 288.0]],
                    [[36.0, 150.0], [54.0, 180.0], [108.0, 300.0]],
                ],
            ),
            (
                "divide",
                backend.divide,
                [
                    [[4.0, 4.0], [3.0, 3.0], [2.0, 2.0]],
                    [[9.0, 6.0], [6.0, 5.0], [3.0, 3.0]],
                ],
            ),
            (
                "maximum",
                backend.maximum,
                [
                    [[12.0, 24.0], [12.0, 24.0], [12.0, 24.0]],
                    [[18.0, 30.0], [18.0, 30.0], [18.0, 30.0]],
                ],
            ),
            (
                "minimum",
                backend.minimum,
                [
                    [[3.0, 6.0], [4.0, 8.0], [6.0, 12.0]],
                    [[2.0, 5.0], [3.0, 6.0], [6.0, 10.0]],
                ],
            ),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 3, 2))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

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

    def test_elementwise_methods_raise_when_3D_tensor_shapes_are_not_broadcast_compatible(
        self,
    ):
        """
        This tests that elementwise operations raise when higher-rank
        operand shapes are not compatible for broadcasting.

        The left-hand tensor a has shape (2, 3, 2) and the right-hand
        tensor b has shape (2, 2, 2). Comparing shapes from right to left,
        the rightmost axes both have length 2, so they match. The middle
        axis has of a has length 3 and b has length 2. Because these
        lengths are neither equal nor one of them is 1, there is no valid
        way to extend either of the counterpart axes.

        This means the operations cannot be carried out elementwise and
        should therefore raise an exception.
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[2.0, 6.0], [12.0, 20.0], [4.0, 12.0]],
                [[3.0, 9.0], [8.0, 15.0], [5.0, 15.0]],
            ]
        )
        b = backend.to_tensor(
            [
                [[1.0, 3.0], [4.0, 5.0]],
                [[2.0, 4.0], [5.0, 6.0]],
            ]
        )

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


@EnforceSharedNumericFixtures()
class BackendContractElementwiseLeftPaddingBroadcastingMixin(BackendContractBase):
    """
    The tests in BackendContractElementwiseLengthOneAxisBroadcastingMixin cover
    the case where both operands have the same number of dimensions. The
    tests in this class cover what happens when the operands have different
    numbers of dimensions.

    When one operand has fewer dimensions, the right-to-left shape
    comparison still applies, but the lower-rank operand runs out of axes
    first. The remaining axes — those present only in the higher-rank
    operand — are treated as though the lower-rank operand had extra leading
    axes of length 1 prepended to its shape. This left-padding step gives
    this class its name.

    An extra leading axis of length 1 simply wraps the existing data in one
    more layer: the added axis contains only the operand itself as its
    single element. For example, a 2D operand of shape (2, 3):

        [1.0, 2.0, 3.0]
        [4.0, 5.0, 6.0]

    treated as shape (1, 2, 3) still contains the same two rows — the
    operand itself is now the single element along the added axis. During
    the operation, it is reused for every corresponding position in the
    other operand, exactly as a length-1 axis is broadcast in
    BackendContractElementwiseLengthOneAxisBroadcastingMixin. It does not
    matter how many fewer dimensions the lower-rank operand has.

    The case where one operand already has an explicit leading axis of
    length 1 is also covered here: it behaves identically to the implicit
    left-padding case, and can arise naturally after operations such as
    reshape or a reduction with keepdims=True.
    """

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


@EnforceSharedNumericFixtures()
class BackendContractElementwiseDualBroadcastingMixin(BackendContractBase):
    """
    The previous broadcasting mixins test the two broadcasting mechanisms
    separately. The tests in this class cover cases where both mechanisms
    apply in the same operation.

    In BackendContractElementwiseLeftPaddingBroadcastingMixin, the lower-rank
    operand is left-padded to match the rank of the other operand, and those
    added axes — each of length 1 — are broadcast. The existing axes of the
    lower-rank operand match the corresponding axes of the higher-rank
    operand exactly, so no further broadcasting is needed.

    The tests in this class remove that restriction. After left-padding, at
    least one of the now-aligned axis pairs still has length 1 on one side,
    so the length-1-axis rule also applies. In some tests both mechanisms
    apply to the same operand; in others each mechanism applies to a
    different operand.

    These tests confirm that left-padding and the length-1-axis rule can
    operate together.
    """

    def test_elementwise_methods_broadcast_when_both_broadcasting_rules_apply_to_the_lower_rank_operand(
        self,
    ):
        """
        Both broadcasting mechanisms apply to the lower-rank operand b in this
        test.

        The tensor a has shape (2, 2, 3) and the tensor b has shape (2, 1).
        Because b has fewer dimensions, it is first treated as though it had
        shape (1, 2, 1). Comparing shapes from right to left: the rightmost
        axes have lengths 3 and 1, so b's single value per row is reused across
        the 3 positions in a; the middle axes both have length 2, so they match;
        the leftmost axes have lengths 2 and 1 — the 1 was added by left-padding
        — so b is also reused across the 2 positions along a's leading axis.

        The tensor a is:

            [[2.0,  6.0, 12.0], [4.0,  12.0, 16.0]]
            [[3.0,  9.0, 15.0], [6.0,  12.0, 18.0]]

        The tensor b is:

            [[1.0], [2.0]]

        For add, b's values are broadcast across the 3 columns and applied to
        the first 2D tensor in a:

            [2.0 + 1.0,  6.0 + 1.0, 12.0 + 1.0] = [3.0,  7.0, 13.0]
            [4.0 + 2.0, 12.0 + 2.0, 16.0 + 2.0] = [6.0, 14.0, 18.0]

        b is then reused for the second 2D tensor in a:

            [3.0 + 1.0,  9.0 + 1.0, 15.0 + 1.0] = [4.0, 10.0, 16.0]
            [6.0 + 2.0, 12.0 + 2.0, 18.0 + 2.0] = [8.0, 14.0, 20.0]

        The result has shape (2, 2, 3).
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[2.0, 6.0, 12.0], [4.0, 12.0, 16.0]],
                [[3.0, 9.0, 15.0], [6.0, 12.0, 18.0]],
            ]
        )
        b = backend.to_tensor([[1.0], [2.0]])

        elementwise_methods = [
            (
                "add",
                backend.add,
                [
                    [[3.0, 7.0, 13.0], [6.0, 14.0, 18.0]],
                    [[4.0, 10.0, 16.0], [8.0, 14.0, 20.0]],
                ],
            ),
            (
                "subtract",
                backend.subtract,
                [
                    [[1.0, 5.0, 11.0], [2.0, 10.0, 14.0]],
                    [[2.0, 8.0, 14.0], [4.0, 10.0, 16.0]],
                ],
            ),
            (
                "multiply",
                backend.multiply,
                [
                    [[2.0, 6.0, 12.0], [8.0, 24.0, 32.0]],
                    [[3.0, 9.0, 15.0], [12.0, 24.0, 36.0]],
                ],
            ),
            (
                "divide",
                backend.divide,
                [
                    [[2.0, 6.0, 12.0], [2.0, 6.0, 8.0]],
                    [[3.0, 9.0, 15.0], [3.0, 6.0, 9.0]],
                ],
            ),
            (
                "maximum",
                backend.maximum,
                [
                    [[2.0, 6.0, 12.0], [4.0, 12.0, 16.0]],
                    [[3.0, 9.0, 15.0], [6.0, 12.0, 18.0]],
                ],
            ),
            (
                "minimum",
                backend.minimum,
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                ],
            ),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 2, 3))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_broadcast_when_the_two_rules_apply_to_different_operands(
        self,
    ):
        """
        The two broadcasting mechanisms apply to different operands in this
        test: b is left-padded, while a has a length-1 middle axis that is
        broadcast.

        The tensor a has shape (2, 1, 3) and the tensor b has shape (2, 3).
        Because b has fewer dimensions, it is first treated as though it had
        shape (1, 2, 3). Comparing shapes from right to left: the rightmost
        axes both have length 3, so they match; the middle axes have lengths 1
        and 2, so a's single row is reused across b's 2 rows; the leftmost
        axes have lengths 2 and 1 — the 1 was added by left-padding — so b
        is also reused across a's 2 positions along the leading axis.

        The tensor a is:

            [[12.0, 24.0, 36.0]]
            [[18.0, 30.0, 42.0]]

        The tensor b is:

            [1.0, 2.0, 3.0]
            [2.0, 3.0, 6.0]

        For add, a's single row in its first 2D tensor is broadcast across b's
        2 rows:

            [12.0 + 1.0, 24.0 + 2.0, 36.0 + 3.0] = [13.0, 26.0, 39.0]
            [12.0 + 2.0, 24.0 + 3.0, 36.0 + 6.0] = [14.0, 27.0, 42.0]

        b is then reused for a's second 2D tensor:

            [18.0 + 1.0, 30.0 + 2.0, 42.0 + 3.0] = [19.0, 32.0, 45.0]
            [18.0 + 2.0, 30.0 + 3.0, 42.0 + 6.0] = [20.0, 33.0, 48.0]

        The result has shape (2, 2, 3).
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[12.0, 24.0, 36.0]],
                [[18.0, 30.0, 42.0]],
            ]
        )
        b = backend.to_tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 6.0]])

        elementwise_methods = [
            (
                "add",
                backend.add,
                [
                    [[13.0, 26.0, 39.0], [14.0, 27.0, 42.0]],
                    [[19.0, 32.0, 45.0], [20.0, 33.0, 48.0]],
                ],
            ),
            (
                "subtract",
                backend.subtract,
                [
                    [[11.0, 22.0, 33.0], [10.0, 21.0, 30.0]],
                    [[17.0, 28.0, 39.0], [16.0, 27.0, 36.0]],
                ],
            ),
            (
                "multiply",
                backend.multiply,
                [
                    [[12.0, 48.0, 108.0], [24.0, 72.0, 216.0]],
                    [[18.0, 60.0, 126.0], [36.0, 90.0, 252.0]],
                ],
            ),
            (
                "divide",
                backend.divide,
                [
                    [[12.0, 12.0, 12.0], [6.0, 8.0, 6.0]],
                    [[18.0, 15.0, 14.0], [9.0, 10.0, 7.0]],
                ],
            ),
            (
                "maximum",
                backend.maximum,
                [
                    [[12.0, 24.0, 36.0], [12.0, 24.0, 36.0]],
                    [[18.0, 30.0, 42.0], [18.0, 30.0, 42.0]],
                ],
            ),
            (
                "minimum",
                backend.minimum,
                [
                    [[1.0, 2.0, 3.0], [2.0, 3.0, 6.0]],
                    [[1.0, 2.0, 3.0], [2.0, 3.0, 6.0]],
                ],
            ),
        ]

        for method_name, method, expected in elementwise_methods:
            with self.subTest(method=method_name):
                result_tensor = method(a, b)
                result = backend.to_python(result_tensor)
                self.assertEqual(backend.shape(result_tensor), (2, 2, 3))
                assert_nested_close(result, expected, rel_tol=0, abs_tol=0)

    def test_elementwise_methods_raise_when_left_padding_cannot_resolve_an_aligned_axis_size_mismatch(
        self,
    ):
        """
        Left-padding makes the ranks equal, but cannot fix an axis pair where
        neither length is 1.

        The tensor a has shape (2, 3, 2) and the tensor b has shape (2, 2).
        Because b has fewer dimensions, it is first treated as though it had
        shape (1, 2, 2). Comparing shapes from right to left: the rightmost
        axes both have length 2, so they are compatible; the middle axes have
        lengths 3 and 2 — these are neither equal nor one of them is 1, so
        there is no valid way to pair the values at those positions. The
        leftmost axes would be compatible, but the check does not reach them.

        The operations cannot be carried out elementwise and should therefore
        raise an exception.
        """
        backend = self.make_backend()
        a = backend.to_tensor(
            [
                [[2.0, 6.0], [12.0, 20.0], [4.0, 12.0]],
                [[3.0, 9.0], [8.0, 15.0], [5.0, 15.0]],
            ]
        )
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
