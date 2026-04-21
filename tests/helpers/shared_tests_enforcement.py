"""Decorators and helpers to enforce re-usability of tensor tests

The reference design for tensor backends includes the principle that they
should be float based. However, we want to ensure that backend contract
tests can - as far as possible - be re-used with non-float-based backends
if needed later (e.g. backends designed for inference on embedded devices).

This means separating out tests for arithmetic, which will need new,
counterpart tests for non-float-based backends from those which test
behaviour in relation to shape, axes manipulation etc. We can ensure the
latter group can be used flexibly by only using integer-valued floats
and/or non-bool ints as input and expected values.

We use three decorators, combined in a composite decorator, to do this.
Between them, they patch assert_nested_close twice and the to_tensor
method of the backend under test to:
- ensure to_tensor and assert_nested_close are only passed
permissible values
- ensure the tolerances for assert_nested_close are set to zero

It's not strictly necessary to set the tolerances to zero in the
non-arithmetic tests but it's cleaner when we know we're dealing
with integer-valued floats/ints.

Much of the value here comes from using the decorator(s) to express
intent, so we know not to needlessly mix tests for arithmetic
with those relating to structure in the same test class.

The approach taken in this module is very conservative - and somewhat
verbose as a consequence - when it comes to restoring any shared state
(this is only really a concern with assert_nested_close).

Because these decorators/helpers contain non-trivial logic they are,
themselves, thoroughly tested.
"""

import functools
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from src.tensors.tensor_backend import TensorBackend
from tests.helpers import tensor_assertions
from tests.helpers.tensor_assertions import DEFAULT_ABS_TOL, DEFAULT_REL_TOL

T = TypeVar("T")
type AssertNestedClose = Callable[[Any, Any, float, float], None]
type MakeBackend = Callable[..., TensorBackend]


# Use Generic to effectively declare a group of types for PatchState
# depending on the type of original_value. This keeps mypy happy without
# very loose typing
@dataclass(frozen=True)
class PatchState(Generic[T]):
    """
    Used to track the what has been patched at what level (local,
    global, source module).

    When, for example, we look for a copy of assert_nested_close
    to patch in a test method's __globals__ attribute and don't
    find one original_value will be None. However, just in case
    the assert_nested_close name did exist but was set to None,
    and some future code cares about the difference (most likely
    future code which also carries out patching), we record the
    difference explicitly.

    Taking this approach means that when we restore the patched
    methods to their original values we get precisely the same
    state we started with.
    """

    had_original: bool
    original_value: T


AssertNestedClosePatchState = tuple[
    PatchState[AssertNestedClose],
    PatchState[AssertNestedClose | None],
]


def _assert_uses_only_integer_valued_floats(data: Any, context: str) -> None:
    """
    Used in the class decorators to ensure that values passed to specified
    methods are integer-valued floats or non-bool ints.
    """
    if isinstance(data, bool):
        raise AssertionError(
            f"Shared test {context} contains a boolean value: {data!r}"
        )

    if isinstance(data, int):
        return

    if isinstance(data, float):
        if not data.is_integer():
            raise AssertionError(
                f"Shared test {context} contains a non-integer-valued float: {data!r}"
            )
        return

    if isinstance(data, (list, tuple)):
        for item in data:
            _assert_uses_only_integer_valued_floats(item, context=context)
        return

    raise AssertionError(
        f"Shared test {context} contains an unsupported value: {data!r}"
    )


def _wrap_test_methods(
    cls: type, wrapper_factory: Callable[[Callable], Callable]
) -> type:
    """
    Used every time we need to wrap all the test methods in a decorated
    test class in an enclosing method, which enforces the wrapped-methods'
    behaviour in some way.
    """
    for name, value in list(cls.__dict__.items()):
        if name.startswith("test_") and callable(value):
            setattr(cls, name, wrapper_factory(value))
    return cls


def _patch_assert_nested_close(
    test_method: Callable, wrapped_assert_nested_close: AssertNestedClose
) -> AssertNestedClosePatchState:
    """
    We patch assert_nested_close in its source module and any
    copy found in the test method's __globals__ attribute so
    that the patch works indpendently of how assert_nested_close
    was imported and used:

    from tests.helpers import tensor_assertions
    tensor_assertions.assert_nested_close(x, y)

    from tests.helpers.tensor_assertions import assert_nested_close
    assert_nested_close(x, y)
    """
    module_patch_state = PatchState[AssertNestedClose](
        had_original=True,
        original_value=tensor_assertions.assert_nested_close,
    )
    globals_patch_state = PatchState[AssertNestedClose | None](
        had_original="assert_nested_close" in test_method.__globals__,
        original_value=test_method.__globals__.get("assert_nested_close"),
    )

    try:
        setattr(tensor_assertions, "assert_nested_close", wrapped_assert_nested_close)
        if globals_patch_state.had_original:
            test_method.__globals__["assert_nested_close"] = wrapped_assert_nested_close
    except Exception as exc:
        raise RuntimeError(
            "shared_tests_enforcement failed while patching "
            "assert_nested_close after shared state may already have been "
            "modified. The remainder of this test run should be treated as "
            "unreliable."
        ) from exc

    return module_patch_state, globals_patch_state


def _patch_assert_nested_close_to_enforce_integer_valued_floats(
    test_method: Callable,
) -> AssertNestedClosePatchState:
    """
    Creates a wrapped version of assert_nested_close and patches it into the
    places from which the given test method may resolve it.

    The wrapped version checks both the actual result and expected value
    recursively to ensure they contain only integer-valued floats or
    non-bool ints. If those checks pass, it calls the version of
    assert_nested_close which was in place at the point this function was
    called.

    This means the helper composes cleanly with other assert_nested_close
    patches: if another enforcement patch has already been applied, this
    wrapper delegates to that patched version rather than bypassing it.

    The return value is the patch state needed later to restore the original
    bindings.
    """
    original_module_assert_nested_close = tensor_assertions.assert_nested_close

    def checked_assert_nested_close(
        actual: Any,
        expected: Any,
        rel_tol: float = DEFAULT_REL_TOL,
        abs_tol: float = DEFAULT_ABS_TOL,
    ) -> None:
        _assert_uses_only_integer_valued_floats(actual, context="actual value")
        _assert_uses_only_integer_valued_floats(expected, context="expected value")
        return original_module_assert_nested_close(
            actual, expected, rel_tol=rel_tol, abs_tol=abs_tol
        )

    return _patch_assert_nested_close(test_method, checked_assert_nested_close)


def _patch_assert_nested_close_to_enforce_zero_tolerances(
    test_method: Callable,
) -> AssertNestedClosePatchState:
    """
    Create a wrapped version of assert_nested_close and patch it into the
    places from which the given test method may resolve that helper.

    The wrapped version first checks that assert_nested_close was called
    with rel_tol=0 and abs_tol=0. If that check passes, it calls the
    version of assert_nested_close which was in place at the point this
    function was called.

    This means the helper composes cleanly with other assert_nested_close
    patches: if another enforcement patch has already been applied, this
    wrapper delegates to that patched version rather than bypassing it.

    The return value is the patch state needed later to restore the original
    bindings.
    """
    original_module_assert_nested_close = tensor_assertions.assert_nested_close

    def checked_assert_nested_close(
        actual: Any,
        expected: Any,
        rel_tol: float = DEFAULT_REL_TOL,
        abs_tol: float = DEFAULT_ABS_TOL,
    ) -> None:
        if rel_tol != 0 or abs_tol != 0:
            raise AssertionError(
                "Shared tests must call assert_nested_close with "
                "rel_tol=0 and abs_tol=0."
            )
        return original_module_assert_nested_close(
            actual, expected, rel_tol=rel_tol, abs_tol=abs_tol
        )

    return _patch_assert_nested_close(test_method, checked_assert_nested_close)


def _restore_assert_nested_close(
    method: Callable,
    module_patch_state: PatchState[AssertNestedClose],
    globals_patch_state: PatchState[AssertNestedClose | None],
) -> None:
    """
    Restore assert_nested_close after a temporary enforcement patch.

    This function reverses the work done by _patch_assert_nested_close and
    the higher-level helpers built on top of it. It restores the original
    helper in the tensor_assertions module and, if the test method originally
    resolved assert_nested_close from its __globals__ namespace, restores that
    binding there as well.

    If no assert_nested_close name was originally present in the test
    method's __globals__ namespace, any temporary name introduced during
    patching is removed so that the namespace is returned to its original
    state.

    When multiple enforcement wrappers have been stacked, restoration happens
    one layer at a time as each decorator unwinds in its finally block. Each
    call to this function restores both relevant lookup locations for the
    single patch layer described by the supplied patch state.

    The patch state arguments record what assert_nested_close looked like
    before patching, and whether each copy was present at all. That lets the
    restore step put things back exactly as they were.
    """
    setattr(tensor_assertions, "assert_nested_close", module_patch_state.original_value)
    if globals_patch_state.had_original:
        method.__globals__["assert_nested_close"] = globals_patch_state.original_value
    else:
        method.__globals__.pop("assert_nested_close", None)


def _patch_make_backend(
    test_self: Any, replacement: MakeBackend
) -> PatchState[MakeBackend | None]:
    """
    Patch make_backend for the duration of a test by adding a wrapped version
    to the __dict__ of the current instance of the decorated test class.

    This does not patch the test method itself, and it does not modify the
    test class. Instead, it adds make_backend directly to the current
    instance, so normal attribute lookup finds that copy before the
    make_backend method defined on the class. That is why the patch affects
    only the current test run.

    Unlike the patches for assert_nested_close, we only ever patch
    make_backend in this one lookup location.

    The return value is the patch state needed later to restore the original
    setup.
    """
    make_backend_patch_state = PatchState[MakeBackend | None](
        had_original="make_backend" in test_self.__dict__,
        original_value=test_self.__dict__.get("make_backend"),
    )
    test_self.make_backend = replacement
    return make_backend_patch_state


def _patch_make_backend_to_enforce_integer_valued_to_tensor_inputs(
    test_self: Any,
) -> PatchState[MakeBackend | None]:
    """
    Create a wrapped version of make_backend and patch it in the scope of the
    current test method.

    The wrapped version first calls the version of make_backend which was in
    place when this function was called. It then wraps the backend's
    to_tensor method so that every call checks its input recursively to
    ensure it contains only integer-valued floats or non-bool ints before
    delegating to the original to_tensor method.

    This means the helper composes cleanly with the existing make_backend
    behaviour: it does not replace backend construction logic, but adds the
    shared-test values check immediately before tensor conversion.

    The return value is the patch state needed later to restore the original
    setup.
    """
    original_make_backend = test_self.make_backend

    def checked_make_backend(*backend_args: Any, **backend_kwargs: Any) -> Any:
        backend = original_make_backend(*backend_args, **backend_kwargs)
        original_to_tensor = backend.to_tensor

        def checked_to_tensor(
            data: Any, *to_tensor_args: Any, **to_tensor_kwargs: Any
        ) -> Any:
            _assert_uses_only_integer_valued_floats(data, context="to_tensor input")
            return original_to_tensor(data, *to_tensor_args, **to_tensor_kwargs)

        backend.to_tensor = checked_to_tensor
        return backend

    return _patch_make_backend(test_self, checked_make_backend)


def _restore_make_backend(
    test_self: Any, make_backend_patch_state: PatchState[MakeBackend | None]
) -> None:
    """
    Restore make_backend after a temporary enforcement patch.

    This function reverses the work done by _patch_make_backend and the
    higher-level helpers built on top of it. It restores the version of
    make_backend which was available to the current test method before
    patching.

    Because make_backend is patched only at test-method scope, restoration is
    simpler than for assert_nested_close: a single restore step returns that
    test's setup to its original state.
    """
    if not make_backend_patch_state.had_original:
        delattr(test_self, "make_backend")
    else:
        test_self.make_backend = make_backend_patch_state.original_value


class _PatchedTestClassDecorator:
    """
    Superclass for decorators which patch helper methods before a test method
    runs and restore them afterwards.

    When one of these decorators is applied to a test class, it wraps each
    method whose name begins with test_. The wrapped method then follows the
    same sequence every time it is called:

    1. apply a patch and record whatever state will be needed for restoration
    2. run the original test method
    3. restore the original state in a finally block

    The concrete subclasses decide what gets patched and how it is restored.
    This keeps the wrapping logic in one place and makes it easier to follow
    the flow of control through the module.
    """

    def __call__(self, cls: type) -> type:
        return _wrap_test_methods(cls, self._wrap_test_method)

    def _apply_patch(self, test_method: Callable, test_self: Any) -> Any:
        raise NotImplementedError

    def _restore_patch(
        self, test_method: Callable, test_self: Any, patch_state: Any
    ) -> None:
        raise NotImplementedError

    def _wrap_test_method(self, method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapped(test_self: Any, *args: Any, **kwargs: Any) -> Any:
            patch_state = self._apply_patch(method, test_self)

            try:
                return method(test_self, *args, **kwargs)
            finally:
                self._restore_patch(method, test_self, patch_state)

        return wrapped


class EnforceIntegerValuedFloatsAndIntsInToTensorInputs(_PatchedTestClassDecorator):
    """
    Wraps a test class and ensures that all calls to a backend's to_tensor
    method from the class's test methods are patched in a way which enforces
    the use of integer-valued floats or non-bool ints.
    """

    def _apply_patch(
        self, test_method: Callable, test_self: Any
    ) -> PatchState[MakeBackend | None]:
        return _patch_make_backend_to_enforce_integer_valued_to_tensor_inputs(test_self)

    def _restore_patch(
        self,
        test_method: Callable,
        test_self: Any,
        patch_state: PatchState[MakeBackend | None],
    ) -> None:
        _restore_make_backend(test_self, patch_state)


class EnforceIntegerValuedFloatsAndIntsInAssertNestedClose(_PatchedTestClassDecorator):
    """
    Wraps a test class and ensures that all calls to assert_nested_close
    from the class's test methods are patched in a way which enforces
    the use of integer-valued floats or non-bool ints.
    """

    def _apply_patch(
        self, test_method: Callable, test_self: Any
    ) -> AssertNestedClosePatchState:
        return _patch_assert_nested_close_to_enforce_integer_valued_floats(test_method)

    def _restore_patch(
        self,
        test_method: Callable,
        test_self: Any,
        patch_state: AssertNestedClosePatchState,
    ) -> None:
        _restore_assert_nested_close(test_method, *patch_state)


class EnforceZeroTolerancesInAssertNestedClose(_PatchedTestClassDecorator):
    """
    Wraps a test class and ensures that all calls to assert_nested_close
    from the class's test methods are made with the tolerance parameters
    set to zero.
    """

    def _apply_patch(
        self, test_method: Callable, test_self: Any
    ) -> AssertNestedClosePatchState:
        return _patch_assert_nested_close_to_enforce_zero_tolerances(test_method)

    def _restore_patch(
        self,
        test_method: Callable,
        test_self: Any,
        patch_state: AssertNestedClosePatchState,
    ) -> None:
        _restore_assert_nested_close(test_method, *patch_state)


class EnforceSharedNumericFixtures:
    """
    A composite decorator class, comprising the three enforcement
    decorators.

    The decorators are applied in an order which keeps failures as clear and
    local as possible. Input checks on to_tensor happen first, value checks on
    assert_nested_close happen next, and the zero-tolerance check is applied
    last so that, when assert_nested_close is called, the most specific shared-
    test requirements are enforced in a predictable sequence.
    """

    def __init__(self) -> None:
        self._decorators: list[Callable[[type], type]] = [
            EnforceIntegerValuedFloatsAndIntsInToTensorInputs(),
            EnforceIntegerValuedFloatsAndIntsInAssertNestedClose(),
            EnforceZeroTolerancesInAssertNestedClose(),
        ]

    def __call__(self, cls: type) -> type:
        for decorator in self._decorators:
            cls = decorator(cls)
        return cls
