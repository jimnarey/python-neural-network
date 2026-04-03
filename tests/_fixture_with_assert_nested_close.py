"""Fixture module with assert_nested_close

The sole purpose of this module is to declare a method which has an
assert_nested_close object in its __globals__ attribute. It is used
in the tests for the decorators/helpers which enforce the use of
integer-valued floats and non-bool ints in tensor backend contract
tests which need to be kept flexible enough to work with backend
implementations which may not be float-based (i.e. tests relating
to tensor structure, shape etc).

assert_nested_close does not necessarily have to be a method here
for the decorator/helper tests to work but it may as well be since
we're declaring it anyway.
"""


def assert_nested_close():
    pass


def method_with_assert_nested_close_in_globals():
    pass
