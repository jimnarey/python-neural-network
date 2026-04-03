"""Test module for shared tests enforcement decorators

There is quite a lot to test here. The composite decorator which enforces
all our rules for shared (non-arithmetic) tests layers several patches on
top of each other and the patches for assert_nested_close have to be made
in two places to accomodate different import paths. When the patched versions
have run or if something goes wrong during the patching the original versions
must be restored (or in one edge case, a dire warning issued).

The low level functions are unit tested and there are integration tests for
the three individual decorators and the composite decorator. Three intermediate
functions in the decorator module which co-ordinate between the two other
layers are not tested but this is acceptable.
"""

from types import FunctionType
from unittest.mock import patch
from unittest import TestCase

from tests._fixture_with_assert_nested_close import (
    method_with_assert_nested_close_in_globals,
)
from tests._fixture_without_assert_nested_close import (
    method_without_assert_nested_close_in_globals,
)

from tests.helpers.shared_tests_enforcement import (
    EnforceIntegerValuedFloatsAndIntsInAssertNestedClose,
    EnforceIntegerValuedFloatsAndIntsInToTensorInputs,
    EnforceSharedNumericFixtures,
    EnforceZeroTolerancesInAssertNestedClose,
    PatchState,
    _assert_uses_only_integer_valued_floats,
    _patch_assert_nested_close,
    _patch_make_backend,
    _restore_assert_nested_close,
    _restore_make_backend,
    _wrap_test_methods,
)
from tests.helpers import tensor_assertions
from tests.helpers.tensor_assertions import assert_nested_close

SELF_TEST_CONTEXT = "SELF TEST"


class TestAssertUsesOnlyIntegerValuedFloats(TestCase):
    # Note that some of the nested structures used here are ragged and would
    # fail the tests for the backends' to_tensor methods (and others) but ragged
    # shapes are not the concern of this helper. All we care about here is that
    # the helper recurses through nested sequences until it gets to a non-sequence
    # value.

    def test_does_not_raise_when_passed_integer_valued_float(self):
        _assert_uses_only_integer_valued_floats(1.0, context=SELF_TEST_CONTEXT)

    def test_does_not_raise_when_passed_int(self):
        _assert_uses_only_integer_valued_floats(1, context=SELF_TEST_CONTEXT)

    def test_raises_when_passed_bool(self):
        with self.assertRaises(AssertionError):
            _assert_uses_only_integer_valued_floats(True, context=SELF_TEST_CONTEXT)

    def test_raises_when_passed_non_integer_float(self):
        with self.assertRaises(AssertionError):
            _assert_uses_only_integer_valued_floats(1.5, context=SELF_TEST_CONTEXT)

    def test_raises_when_passed_unsupported_non_numeric_value(self):
        with self.assertRaises(AssertionError):
            _assert_uses_only_integer_valued_floats("x", context=SELF_TEST_CONTEXT)

    def test_does_not_raise_when_passed_nested_structure_of_ints_and_integer_valued_floats(
        self,
    ):
        _assert_uses_only_integer_valued_floats(
            [
                [1.0, 2],
                (3.0, [4, 5.0]),
            ],
            context=SELF_TEST_CONTEXT,
        )

    def test_raises_when_passed_nested_structure_containing_bool(self):
        with self.assertRaises(AssertionError):
            _assert_uses_only_integer_valued_floats(
                [
                    [1.0, 2],
                    (3.0, [False, 5.0]),
                ],
                context=SELF_TEST_CONTEXT,
            )

    def test_raises_when_passed_nested_structure_containing_non_integer_float(self):
        with self.assertRaises(AssertionError):
            _assert_uses_only_integer_valued_floats(
                [
                    [1.0, 2],
                    (3.0, [4, 5.5]),
                ],
                context=SELF_TEST_CONTEXT,
            )


class TestWrapTestMethods(TestCase):
    def test_wrap_test_methods_replaces_methods_when_names_begin_with_test(self):
        class Dummy:
            def test_example(self):
                return "original test"

            def helper_method(self):
                return "original helper"

        original_test_method = Dummy.test_example
        original_helper_method = Dummy.helper_method

        def wrapper_factory(method):
            def wrapped(*args, **kwargs):
                return f"wrapped {method.__name__}"

            return wrapped

        wrapped_class = _wrap_test_methods(Dummy, wrapper_factory)

        self.assertIs(wrapped_class, Dummy)
        self.assertIsNot(Dummy.test_example, original_test_method)
        self.assertIs(Dummy.helper_method, original_helper_method)
        self.assertEqual(Dummy().test_example(), "wrapped test_example")
        self.assertEqual(Dummy().helper_method(), "original helper")


class TestPatchMethodAssertNestedClose(TestCase):
    def test_replaces_assert_nested_close_and_returns_original_state_when_assert_nested_close_is_not_in_function_globals(
        self,
    ):
        method = method_without_assert_nested_close_in_globals

        def replacement(*args, **kwargs):
            return None

        module_assert_nested_close = tensor_assertions.assert_nested_close

        with patch.object(
            tensor_assertions,
            "assert_nested_close",
            module_assert_nested_close,
        ):
            returned_module_state, returned_global_state = _patch_assert_nested_close(
                method, replacement
            )

            self.assertEqual(
                returned_module_state,
                PatchState(
                    had_original=True,
                    original_value=module_assert_nested_close,
                ),
            )
            self.assertEqual(
                returned_global_state,
                PatchState(
                    had_original=False,
                    original_value=None,
                ),
            )
            self.assertIs(tensor_assertions.assert_nested_close, replacement)
            self.assertNotIn("assert_nested_close", method.__globals__)

    def test_replaces_assert_nested_close_and_returns_original_state_when_assert_nested_close_is_in_function_globals(
        self,
    ):
        method = method_with_assert_nested_close_in_globals

        globals_assert_nested_close = method.__globals__["assert_nested_close"]

        def replacement(*args, **kwargs):
            return None

        module_assert_nested_close = tensor_assertions.assert_nested_close

        with (
            patch.object(
                tensor_assertions,
                "assert_nested_close",
                module_assert_nested_close,
            ),
            patch.dict(
                method.__globals__,
                {"assert_nested_close": globals_assert_nested_close},
                clear=False,
            ),
        ):
            returned_module_state, returned_global_state = _patch_assert_nested_close(
                method, replacement
            )

            self.assertEqual(
                returned_module_state,
                PatchState(
                    had_original=True,
                    original_value=module_assert_nested_close,
                ),
            )
            self.assertEqual(
                returned_global_state,
                PatchState(
                    had_original=True,
                    original_value=globals_assert_nested_close,
                ),
            )
            self.assertIs(tensor_assertions.assert_nested_close, replacement)
            self.assertIs(method.__globals__["assert_nested_close"], replacement)

    def test_raises_runtime_error_with_warning_message_when_patching_globals_fails_after_module_helper_is_patched(
        self,
    ):
        module_assert_nested_close = tensor_assertions.assert_nested_close

        def globals_assert_nested_close():
            return None

        def replacement(*args, **kwargs):
            return None

        class GlobalsWhichRaiseWhenPatched(dict):
            def __setitem__(self, key, value):
                if key == "assert_nested_close":
                    raise RuntimeError("patching globals failed")
                super().__setitem__(key, value)

        method_globals = GlobalsWhichRaiseWhenPatched(
            {
                "__builtins__": __builtins__,
                "assert_nested_close": globals_assert_nested_close,
            }
        )

        def template():
            pass

        method = FunctionType(template.__code__, method_globals)

        with patch.object(
            tensor_assertions,
            "assert_nested_close",
            module_assert_nested_close,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "shared_tests_enforcement failed while patching",
            ):
                _patch_assert_nested_close(method, replacement)

            self.assertIs(tensor_assertions.assert_nested_close, replacement)
            self.assertIs(
                method.__globals__["assert_nested_close"],
                globals_assert_nested_close,
            )


# We always want to patch assert_nested_close at the source module level but only
# at the test module level if it is present (it's in the test method's __globals__)
class TestRestoreMethodAssertNestedClose(TestCase):
    def test_restores_assert_nested_close_and_removes_temporary_name_when_assert_nested_close_is_not_in_function_globals(
        self,
    ):
        method = method_without_assert_nested_close_in_globals

        module_assert_nested_close = tensor_assertions.assert_nested_close

        def replacement_module_helper(*args, **kwargs):
            return None

        with (
            patch.object(
                tensor_assertions,
                "assert_nested_close",
                replacement_module_helper,
            ),
            patch.dict(
                method.__globals__,
                {"assert_nested_close": "TEMPORARY VALUE"},
                clear=False,
            ),
        ):
            _restore_assert_nested_close(
                method,
                PatchState(
                    had_original=True,
                    original_value=module_assert_nested_close,
                ),
                PatchState(
                    had_original=False,
                    original_value=None,
                ),
            )

            self.assertIs(
                tensor_assertions.assert_nested_close, module_assert_nested_close
            )
            self.assertNotIn("assert_nested_close", method.__globals__)

    def test_restores_assert_nested_close_and_restores_original_name_when_assert_nested_close_is_in_function_globals(
        self,
    ):
        method = method_with_assert_nested_close_in_globals

        module_assert_nested_close = tensor_assertions.assert_nested_close
        globals_assert_nested_close = method.__globals__["assert_nested_close"]

        def replacement_module_helper(*args, **kwargs):
            return None

        with (
            patch.object(
                tensor_assertions,
                "assert_nested_close",
                replacement_module_helper,
            ),
            patch.dict(
                method.__globals__,
                {"assert_nested_close": "TEMPORARY VALUE"},
                clear=False,
            ),
        ):
            _restore_assert_nested_close(
                method,
                PatchState(
                    had_original=True,
                    original_value=module_assert_nested_close,
                ),
                PatchState(
                    had_original=True,
                    original_value=globals_assert_nested_close,
                ),
            )

            self.assertIs(
                tensor_assertions.assert_nested_close, module_assert_nested_close
            )
            self.assertIs(
                method.__globals__["assert_nested_close"], globals_assert_nested_close
            )


class TestPatchMakeBackend(TestCase):
    def test_replaces_make_backend_and_returns_original_state_when_make_backend_is_not_in_test_self_dict(
        self,
    ):
        class TensorBackendDummy:
            def make_backend(self):
                return "class original"

        test_self = TensorBackendDummy()

        def replacement():
            return "replacement"

        returned_patch_state = _patch_make_backend(test_self, replacement)

        self.assertEqual(
            returned_patch_state,
            PatchState(
                had_original=False,
                original_value=None,
            ),
        )
        self.assertIs(test_self.__dict__["make_backend"], replacement)
        self.assertEqual(test_self.make_backend(), "replacement")

    def test_replaces_make_backend_and_returns_original_state_when_make_backend_is_in_test_self_dict(
        self,
    ):
        class TensorBackendDummy:
            def make_backend(self):
                return "class original"

        test_self = TensorBackendDummy()

        def original_instance_make_backend():
            return "instance original"

        def replacement():
            return "replacement"

        test_self.make_backend = original_instance_make_backend

        returned_patch_state = _patch_make_backend(test_self, replacement)

        self.assertEqual(
            returned_patch_state,
            PatchState(
                had_original=True,
                original_value=original_instance_make_backend,
            ),
        )
        self.assertIs(test_self.__dict__["make_backend"], replacement)
        self.assertEqual(test_self.make_backend(), "replacement")


class TestRestoreMakeBackend(TestCase):
    def test_restores_make_backend_and_removes_temporary_name_when_make_backend_is_not_in_test_self_dict(
        self,
    ):
        class TensorBackendDummy:
            def make_backend(self):
                return "class original"

        test_self = TensorBackendDummy()

        def replacement():
            return "replacement"

        test_self.make_backend = replacement

        _restore_make_backend(
            test_self,
            PatchState(
                had_original=False,
                original_value=None,
            ),
        )

        self.assertNotIn("make_backend", test_self.__dict__)
        self.assertEqual(test_self.make_backend(), "class original")

    def test_restores_make_backend_and_restores_original_name_when_make_backend_is_in_test_self_dict(
        self,
    ):
        class TensorBackendDummy:
            def make_backend(self):
                return "class original"

        test_self = TensorBackendDummy()

        def original_instance_make_backend():
            return "instance original"

        def replacement():
            return "replacement"

        test_self.make_backend = replacement

        _restore_make_backend(
            test_self,
            PatchState(
                had_original=True,
                original_value=original_instance_make_backend,
            ),
        )

        self.assertIs(
            test_self.__dict__["make_backend"], original_instance_make_backend
        )
        self.assertEqual(test_self.make_backend(), "instance original")


class TestEnforceIntegerValuedFloatsAndIntsInToTensorInputs(TestCase):
    def test_enforce_integer_valued_floats_in_to_tensor_inputs_allows_permissible_values(
        self,
    ):
        class TensorBackendDummy:
            def to_tensor(self, data):
                return data

        @EnforceIntegerValuedFloatsAndIntsInToTensorInputs()
        class DecoratedDummy:
            def make_backend(self):
                return TensorBackendDummy()

            def test_example(self, data):
                backend = self.make_backend()
                return backend.to_tensor(data)

        valid_inputs = [
            [1.0, 2.0],
            [1, 2],
            [[1.0, 2], (3.0, [4, 5.0])],
        ]

        for data in valid_inputs:
            with self.subTest(data=data):
                self.assertEqual(DecoratedDummy().test_example(data), data)

    def test_enforce_integer_valued_floats_in_to_tensor_inputs_raises_for_impermissible_values(
        self,
    ):
        class TensorBackendDummy:
            def to_tensor(self, data):
                return data

        @EnforceIntegerValuedFloatsAndIntsInToTensorInputs()
        class DecoratedDummy:
            def make_backend(self):
                return TensorBackendDummy()

            def test_example(self, data):
                backend = self.make_backend()
                return backend.to_tensor(data)

        invalid_inputs = [
            [1.0, 2.5],
            [1.0, True],
            [1.0, "x"],
        ]

        for data in invalid_inputs:
            with self.subTest(data=data):
                with self.assertRaises(AssertionError):
                    DecoratedDummy().test_example(data)

    def test_enforce_integer_valued_floats_in_to_tensor_inputs_restores_make_backend_when_wrapped_test_method_raises(
        self,
    ):
        class TensorBackendDummy:
            def to_tensor(self, data):
                return data

        @EnforceIntegerValuedFloatsAndIntsInToTensorInputs()
        class DecoratedDummy:
            def make_backend(self):
                return TensorBackendDummy()

            def test_example(self):
                assert "make_backend" in self.__dict__
                raise RuntimeError("wrapped test method failed")

        decorated_dummy = DecoratedDummy()

        with self.assertRaisesRegex(RuntimeError, "wrapped test method failed"):
            decorated_dummy.test_example()

        self.assertNotIn("make_backend", decorated_dummy.__dict__)


class TestEnforceIntegerValuedFloatsAndIntsInAssertNestedClose(TestCase):
    def test_enforce_integer_valued_floats_in_assert_nested_close_allows_permissible_values(
        self,
    ):
        @EnforceIntegerValuedFloatsAndIntsInAssertNestedClose()
        class DecoratedDummy:
            def test_example(self, actual, expected):
                tensor_assertions.assert_nested_close(
                    actual,
                    expected,
                    rel_tol=0,
                    abs_tol=0,
                )
                return "passed"

        valid_pairs = [
            ([1.0, 2.0], [1.0, 2.0]),
            ([1, 2], [1.0, 2.0]),
            ([1.0, 2.0], [1, 2]),
            ([1, 2], [1, 2]),
            ([[1.0, 2], (3.0, [4, 5.0])], [[1.0, 2], (3.0, [4, 5.0])]),
        ]

        for actual, expected in valid_pairs:
            with self.subTest(actual=actual, expected=expected):
                self.assertEqual(
                    DecoratedDummy().test_example(actual, expected),
                    "passed",
                )

    def test_enforce_integer_valued_floats_in_assert_nested_close_raises_for_impermissible_values(
        self,
    ):
        @EnforceIntegerValuedFloatsAndIntsInAssertNestedClose()
        class DecoratedDummy:
            def test_example(self, actual, expected):
                tensor_assertions.assert_nested_close(
                    actual,
                    expected,
                    rel_tol=0,
                    abs_tol=0,
                )

        invalid_pairs = [
            ([1.0, 2.5], [1.0, 2.5]),
            ([1.0, True], [1.0, True]),
            ([1.0, "x"], [1.0, "x"]),
        ]

        for actual, expected in invalid_pairs:
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaises(AssertionError):
                    DecoratedDummy().test_example(actual, expected)

    def test_enforce_integer_valued_floats_in_assert_nested_close_restores_assert_nested_close_when_wrapped_test_method_raises(
        self,
    ):
        module_assert_nested_close = tensor_assertions.assert_nested_close

        @EnforceIntegerValuedFloatsAndIntsInAssertNestedClose()
        class DecoratedDummy:
            def test_example(self):
                assert (
                    tensor_assertions.assert_nested_close
                    is not module_assert_nested_close
                )
                raise RuntimeError("wrapped test method failed")

        with self.assertRaisesRegex(RuntimeError, "wrapped test method failed"):
            DecoratedDummy().test_example()

        self.assertIs(tensor_assertions.assert_nested_close, module_assert_nested_close)

    def test_enforce_integer_valued_floats_in_assert_nested_close_applies_to_directly_imported_assert_nested_close(
        self,
    ):
        @EnforceIntegerValuedFloatsAndIntsInAssertNestedClose()
        class DecoratedDummy:
            def test_example(self):
                assert_nested_close(
                    [1.0, 2.5],
                    [1.0, 2.5],
                    rel_tol=0,
                    abs_tol=0,
                )

        with self.assertRaises(AssertionError):
            DecoratedDummy().test_example()


class TestEnforceZeroTolerancesInAssertNestedClose(TestCase):
    def test_enforce_zero_tolerances_in_assert_nested_close_allows_zero_tolerances(
        self,
    ):
        @EnforceZeroTolerancesInAssertNestedClose()
        class DecoratedDummy:
            def test_example(self):
                tensor_assertions.assert_nested_close(
                    [1.0, 2.0],
                    [1.0, 2.0],
                    rel_tol=0,
                    abs_tol=0,
                )
                return "passed"

        self.assertEqual(DecoratedDummy().test_example(), "passed")

    def test_enforce_zero_tolerances_in_assert_nested_close_raises_for_non_zero_or_default_tolerances(
        self,
    ):
        @EnforceZeroTolerancesInAssertNestedClose()
        class DecoratedDummy:
            def test_example(self, call_style):
                if call_style == "default":
                    tensor_assertions.assert_nested_close([1.0, 2.0], [1.0, 2.0])
                    return
                if call_style == "non_zero_rel_tol":
                    tensor_assertions.assert_nested_close(
                        [1.0, 2.0],
                        [1.0, 2.0],
                        rel_tol=1e-7,
                        abs_tol=0,
                    )
                    return
                if call_style == "non_zero_abs_tol":
                    tensor_assertions.assert_nested_close(
                        [1.0, 2.0],
                        [1.0, 2.0],
                        rel_tol=0,
                        abs_tol=1e-8,
                    )

        invalid_call_styles = [
            "default",
            "non_zero_rel_tol",
            "non_zero_abs_tol",
        ]

        for call_style in invalid_call_styles:
            with self.subTest(call_style=call_style):
                with self.assertRaises(AssertionError):
                    DecoratedDummy().test_example(call_style)

    def test_enforce_zero_tolerances_in_assert_nested_close_applies_to_directly_imported_assert_nested_close(
        self,
    ):
        @EnforceZeroTolerancesInAssertNestedClose()
        class DecoratedDummy:
            def test_example(self):
                assert_nested_close([1.0, 2.0], [1.0, 2.0])

        with self.assertRaises(AssertionError):
            DecoratedDummy().test_example()


class TestEnforceSharedNumericFixtures(TestCase):
    def test_enforce_shared_numeric_fixtures_allows_permissible_values_with_zero_tolerances(
        self,
    ):
        class TensorBackendDummy:
            def to_tensor(self, data):
                return data

        @EnforceSharedNumericFixtures()
        class DecoratedDummy:
            def make_backend(self):
                return TensorBackendDummy()

            def test_example(self):
                tensor_assertions.assert_nested_close(
                    [1.0, 2.0],
                    [1, 2],
                    rel_tol=0,
                    abs_tol=0,
                )
                return "passed"

        self.assertEqual(DecoratedDummy().test_example(), "passed")

    def test_enforce_shared_numeric_fixtures_raises_for_impermissible_values_or_tolerances(
        self,
    ):
        class TensorBackendDummy:
            def to_tensor(self, data):
                return data

        @EnforceSharedNumericFixtures()
        class DecoratedDummy:
            def make_backend(self):
                return TensorBackendDummy()

            def test_example(self, call_style):
                if call_style == "non-integer-valued float":
                    tensor_assertions.assert_nested_close(
                        [1.0, 2.5],
                        [1.0, 2.5],
                        rel_tol=0,
                        abs_tol=0,
                    )
                    return
                if call_style == "default_tolerances":
                    tensor_assertions.assert_nested_close(
                        [1.0, 2.0],
                        [1.0, 2.0],
                    )

        invalid_call_styles = [
            "non-integer-valued float",
            "default_tolerances",
        ]

        for call_style in invalid_call_styles:
            with self.subTest(call_style=call_style):
                with self.assertRaises(AssertionError):
                    DecoratedDummy().test_example(call_style)

    def test_enforce_shared_numeric_fixtures_raises_for_impermissible_assert_nested_close_values_before_tolerance_violation(
        self,
    ):
        class TensorBackendDummy:
            def to_tensor(self, data):
                return data

        @EnforceSharedNumericFixtures()
        class DecoratedDummy:
            def make_backend(self):
                return TensorBackendDummy()

            def test_example(self):
                tensor_assertions.assert_nested_close(
                    [1.0, 2.5],
                    [1.0, 2.5],
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                )

        with self.assertRaisesRegex(
            AssertionError,
            "non-integer-valued float",
        ):
            DecoratedDummy().test_example()
