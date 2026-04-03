"""Fixture module without assert_nested_close

The sole purpose of this module is to declare a method which DOES
NOT have an assert_nested_close object in its __globals__ attribute.

See the docstring in the counterpart fixture for more information.
"""


def method_without_assert_nested_close_in_globals():
    pass
