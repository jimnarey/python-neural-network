
Straw man package shape:

```
src/tensors/python_backend/
    __init__.py
    backend.py
    _types.py
    _validation.py
    _shape.py
    _axes.py
    _traversal.py
    _broadcast.py
    conversion.py
    creation.py
    transform.py
    elementwise.py
    reductions.py
    matmul.py
    composition.py
```

Proposed division of labour between `PythonTensor` and `PythonBackend`:

- `PythonTensor` should own representation concerns only: the flat buffer, shape, strides, offset, and low-level access to elements by full index.
- `PythonTensor` should provide primitive representation-level helpers such as:
  - copying
  - contiguous copying
  - view construction
  - conversion to nested Python lists.
- `PythonBackend` methods should own:
  - validation
  - broadcasting
  - axis handling
  - all operations, including reshape and transpose, even if their implementations can sometimes be metadata-only.
- `ndim` and `size` should be derived rather than stored as independent attributes (profile this).

Methods to add:

is_contiguous, linear_index, view, copy, contiguous_copy, to_list, iter_indices, iter_index_value_pairs

Consider

subscripting and slicing
