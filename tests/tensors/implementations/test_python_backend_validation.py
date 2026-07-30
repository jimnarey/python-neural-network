import unittest


class TestRequireNonEmptyTensorSequence(unittest.TestCase):

    def test_returns_tuple_containing_tensors(self):
        pass

    def test_accepts_non_list_sequence(self):
        pass

    def test_raises_when_sequence_is_empty(self):
        pass


class TestValidateStackShapes(unittest.TestCase):

    def test_accepts_tensors_with_same_shape(self):
        pass

    def test_accepts_single_tensor(self):
        pass

    def test_accepts_tensors_with_same_zero_length_shape(self):
        pass

    def test_raises_when_tensor_shapes_differ(self):
        pass
