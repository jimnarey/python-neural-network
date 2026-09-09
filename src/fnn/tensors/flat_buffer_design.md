# Flat-buffer tensor and C backend design

This document records a possible route from the current `PythonTensor` and
`PythonBackend` to a native C backend. It is a design note, not a description
of interfaces that have already been implemented.

The main objective is to model important traversal and operation logic in
Python before translating it into C. The initial C implementation should be
simple and should closely resemble the logic-optimised Python implementation.
More specialised native optimisations, including use of BLAS, can be considered
later.

## Layers of interface

There are two separate kinds of protocol.

`TensorBackend[T]` is the application-facing interface. Networks, layers,
activations, losses, and optimisers should continue to use this protocol and
should treat `T` as opaque. The addition of a C backend should not require
these parts of the application to know about buffers, offsets, or strides.

`FlatBufferTensor` describes a representation shared by tensor types such as
`PythonTensor` and the proposed `CTensor`. It is intended for low-level
operation implementations and representation tests. It does not replace
`TensorBackend` and should not be used by network code.

The intended dependency direction is:

```text
network code -> TensorBackend[T] -> concrete backend -> concrete tensor
                                                       |
                                                       +-> FlatBufferTensor
```

## Protocol changes

Introduce a small structural protocol for a one-dimensional tensor buffer.
Its initial surface should describe only the behaviour that Python operation
implementations and shared representation tests require:

```python
class FlatBuffer(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Scalar: ...
    def __setitem__(self, index: int, value: Scalar) -> None: ...
    def __iter__(self) -> Iterator[Scalar]: ...
```

This must not depend on `array.array`. `PythonTensor.data` can contain an
`array.array`, while `CTensor.data` can return a typed, one-dimensional
`memoryview` over native storage. Both can conform structurally.

Introduce a structural `FlatBufferTensor` protocol based initially on the
existing public `PythonTensor` surface:

```python
class FlatBufferTensor(Protocol):
    data: FlatBuffer
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int
    writable: bool

    def ndim(self) -> int: ...
    def size(self) -> int: ...
    def is_contiguous(self) -> bool: ...
    def get_scalar(self, indices: tuple[int, ...]) -> Scalar: ...
    def set_scalar(self, indices: tuple[int, ...], value: Scalar) -> None: ...
    def indices(self) -> Iterator[tuple[int, ...]]: ...
    def items(self) -> Iterator[tuple[tuple[int, ...], Scalar]]: ...
    def to_list(self) -> list: ...
    def view(...): ...
    def copy(...): ...
```

The exact annotations for `view` and `copy` should preserve the concrete
tensor type, using a type parameter or `Self` as appropriate when the protocol
is implemented.

The existing name `data` can be retained. Renaming it is not necessary for
test reuse or direct-buffer operation implementations.

The protocol should describe element-based addressing: `offset` and every
entry in `strides` are measured in buffer elements rather than bytes.

The protocols are structural. `PythonTensor` and `CTensor` do not need to
inherit from them. A stub file for the C extension can describe the Python
surface of `CTensor` to the type checker.

## Contiguity

Add `is_contiguous()` to `PythonTensor` when a concrete operation needs it.
Using a method is consistent with the current `ndim()` and `size()` surface.

A tensor is C-contiguous when its logical elements occupy consecutive buffer
positions in row-major order beginning at its own offset. Offset zero is not
required: a consecutive view into part of a larger allocation can be
contiguous.

Contiguity can be determined by walking shape and strides from right to left,
starting with an expected stride of one. Axes with length one do not constrain
the stride because they contain only one position. Every axis with length
greater than one must have the expected stride, after which the expected
stride is multiplied by the axis length.

An empty tensor should be considered contiguous. No element is traversed, so
any otherwise valid empty layout is safe for a contiguous loop.

A broadcast axis with length greater than one and stride zero is not
contiguous. A typical transpose is also not contiguous.

Contiguity is useful for selecting simple traversal algorithms, particularly
for unary and elementwise operations. It is not a prerequisite for all direct
buffer access. Matmul can support non-contiguous inputs directly by using
their offsets and strides.

## Operation logic

Performance-sensitive operations should read the tensor representation once
and access the buffer directly. This is intentional use of the
`FlatBufferTensor` interface, not a bypass of the tensor abstraction.

The logical convenience methods remain useful for validation, debugging,
conversion, simple implementations, and generic fallbacks. Hot loops should
not be forced through `indices()`, `items()`, `get_scalar()`, `set_scalar()`,
or a callback-based common traversal helper.

Different operations should be allowed to use different traversal structures:

- contiguous unary operations can walk one consecutive buffer range;
- elementwise operations can use a contiguous path or aligned strided cursors;
- reductions need loops organised around source and reduction axes;
- concatenation and stack need source- and destination-specific traversal;
- matmul needs loops organised around leading dimensions, rows, columns, and
  the contracted dimension.

Operation implementations may continue to be specific to `PythonBackend`.
There is no requirement for a C backend to invoke a shared Python operation
function. Reuse should come from a common representation model, common
behavioural tests, and deliberately similar algorithms.

Direct writes to `data` bypass `set_scalar()` validation. Internal kernels
must therefore observe these invariants:

- public backend validation happens before a kernel is called;
- input buffers are not modified;
- an output has the correct shape and element type before it is passed to a
  kernel;
- a kernel writes only within the output layout;
- a kernel that accepts a caller-provided output checks writability.

Concrete Python operations necessarily allocate concrete Python tensors.
Output construction does not need to become part of `FlatBufferTensor` merely
to enable representation test reuse. Allocation responsibilities can be
revisited independently if operation code is later made generic across tensor
implementations.

## First matmul experiment

The first direct-buffer optimisation should preserve all of the existing
matmul generality. Rank-2 operands and contiguous layouts should not be given
a separate algorithm. The first step should replace the tuple construction
and scalar helper calls inside `get_matmul_value()` while retaining the
existing general output traversal and shape and broadcasting logic.

For one result position, the existing index-resolution functions identify the
leading indices, row, and column needed from the operands. Those values can be
converted once into physical buffer positions which include each tensor's
offset and strides:

```python
a_leading_index, b_leading_index, row, column = (
    get_matmul_result_index_parts(result_index, a.shape, b.shape)
)

a_index = a.offset
for index, stride in zip(a_leading_index, a.strides):
    a_index += index * stride
if a.ndim() > 1:
    a_index += row * a.strides[-2]

b_index = b.offset
for index, stride in zip(b_leading_index, b.strides):
    b_index += index * stride
if b.ndim() > 1:
    b_index += column * b.strides[-1]

a_inner_stride = a.strides[-1]
b_inner_stride = b.strides[0] if b.ndim() == 1 else b.strides[-2]

total = 0.0
for _ in range(a.shape[-1]):
    total += a.data[a_index] * b.data[b_index]
    a_index += a_inner_stride
    b_index += b_inner_stride
```

This calculation applies to vector-vector, vector-matrix, matrix-vector,
matrix-matrix, and higher-rank matmul. The existing leading-index resolution
continues to handle broadcast dimensions, including dimensions represented by
stride-zero views. Nonzero offsets and arbitrary valid positive strides are
part of the initial implementation rather than later extensions.

The contracted loop advances integer cursors and therefore avoids constructing
two complete operand index tuples and calling `get_scalar()` twice for every
multiplication. One set of result and leading indices remains per output value.
A later optimisation can replace that outer tuple-based traversal with
integer cursors while preserving the same general semantics.

The output should likewise continue to be addressed through its layout. The
current `set_scalar()` call already does this correctly. If it is replaced by
a direct buffer write, the result's offset and strides must be included rather
than assuming an offset-zero contiguous output.

Tests for the first kernel should cover:

- vector-vector, vector-matrix, matrix-vector, and matrix-matrix operations;
- higher-rank operands with matching leading dimensions;
- higher-rank operands with broadcast leading dimensions;
- nonzero offsets on either operand;
- non-contiguous and transposed operands;
- stride-zero broadcast layouts;
- a zero-length contracted dimension;
- zero-length leading and result dimensions.

The existing matmul contract, reference, shared-helper, and implementation
tests should remain applicable to the optimised path. Tests should be added
only where the current suite does not already exercise a representation case
listed above.

An isolated matmul benchmark should accompany the existing dense-forward
benchmark so that changes to matmul are measured separately from addition and
activation.

## Reusing representation tests

Most existing `PythonTensor` tests can become shared flat-buffer
representation tests without changing the `PythonTensor` surface. The shared
tests should not construct `PythonTensor` or `array.array` directly. Instead,
an implementation-specific test subclass should supply factory methods.

A representative factory surface is:

```python
def make_tensor(
    self,
    shape: tuple[int, ...],
    values: Sequence[Scalar],
    *,
    offset: int = 0,
    strides: tuple[int, ...] | None = None,
    writable: bool = True,
    element_type: ElementType = ElementType.FLOAT,
) -> FlatBufferTensor:
    raise NotImplementedError
```

Additional hooks may be useful for allocating default storage and for
constructing views through the public interface. The factory contract should
describe outcomes rather than require all implementations to expose identical
constructors.

The `PythonTensor` test subclass constructs an `array.array` and a
`PythonTensor`. The `CTensor` subclass uses the C extension's construction
function. The shared test methods then exercise only the common tensor
surface.

Reusable tests can cover:

- shape, strides, offset, size, rank, and writability;
- default contiguous layout;
- scalar reads and writes;
- layout-aware reads and writes with offsets and non-default strides;
- multidimensional index and item iteration;
- nested-list conversion;
- view layout and storage aliasing;
- copy values and storage independence;
- copying non-contiguous tensors into logical order;
- float- and integer-valued tensors where both implementations support them;
- zero-length dimensions and valid empty layouts;
- contiguity classification.

Storage aliasing should be tested behaviourally. For example, changing a
writable view should be observable through its parent. A shared test should
not require `view.data is parent.data`: two distinct `memoryview` objects may
safely expose the same native allocation.

Generic buffer assertions should use operations promised by `FlatBuffer`,
such as `list(tensor.data)`, rather than `array.array` methods such as
`tensor.data.tolist()`.

Python-specific tests should remain for:

- accepted and rejected `array.array` typecodes;
- the concrete `typecode` constructor argument;
- `_default_data()` and `_validated_data()`;
- exact `array.array` behaviour;
- private layout and indexing helpers where those helpers remain part of the
  Python implementation rather than the shared representation contract.

Backend-neutral contract and reference tests remain separate. They verify the
observable semantics of `TensorBackend`; the flat-buffer tests verify the
representation claimed by particular tensor types.

## C backend representation

The native implementation should separate shared storage from individual
tensor layouts:

```c
typedef enum {
    FNN_FLOAT64,
    FNN_INT64
} FnnDType;

typedef struct {
    void *data;
    int64_t length;
    FnnDType dtype;
    bool writable;
    size_t refcount;
} FnnStorage;

typedef struct {
    FnnStorage *storage;
    int64_t *shape;
    int64_t *strides;
    int64_t offset;
    int64_t ndim;
    int64_t size;
    bool writable;
} FnnTensor;
```

Each tensor view has its own descriptor and shape and stride arrays. Views
retain the same `FnnStorage`, incrementing its reference count. Destroying a
parent tensor therefore cannot invalidate a surviving view. The allocation is
freed when the final tensor or exposed buffer object releases the storage.

The eventual details may need additional ownership information, including a
destructor callback or a strong reference to a Python owner when storage is
borrowed rather than allocated by the C backend.

## Python `CTensor` wrapper

A plain C struct is not a Python object and cannot itself conform to a Python
protocol. A CPython extension type wraps the native descriptor:

```c
typedef struct {
    PyObject_HEAD
    FnnTensor *tensor;
    PyObject *data_object;
} PyCTensor;
```

`CTensor` is the Python class exposed by the extension. Its C getters expose
`shape`, `strides`, `offset`, `writable`, and `data`; its C methods implement
the remainder of the `FlatBufferTensor` surface. A `.pyi` stub describes that
surface to the Python type checker, allowing structural conformance without
runtime inheritance.

Python-level metadata access may initially construct shape and stride tuples
on demand. Caching can be considered only if profiling later justifies it.

Native backend operations receive `CTensor` extension objects, extract their
`FnnTensor` pointers once, and run complete operations directly against the
native buffers. They do not call Python getters or buffer indexing methods for
each scalar.

## Exposing native storage through `memoryview`

`CTensor.data` should return a typed, one-dimensional `memoryview` over the
complete native storage allocation. It represents the equivalent of
`PythonTensor.data`, not the tensor's logical multidimensional layout:

```python
tensor.shape       # The logical tensor shape
tensor.strides     # Logical strides measured in elements
tensor.data.shape  # A one-dimensional view of the underlying allocation
```

The memoryview must be typed so that integer indexing addresses one tensor
element and returns a Python float or integer. It must not expose an untyped
byte view. Float64 and int64 storage therefore need the appropriate native
buffer format and item size.

The native tensor representation continues to measure `offset` and `strides`
in elements. Python's native buffer export interface describes lengths and
strides in bytes, so the exporter must perform this conversion explicitly.
For example, an element stride of three over float64 storage corresponds to a
byte stride of `3 * sizeof(double)`.

A small private extension object can export the allocation through Python's
native buffer interface:

```c
typedef struct {
    PyObject_HEAD
    FnnStorage *storage;
    bool writable;
} PyFnnStorageExporter;
```

The exporter retains `FnnStorage` and supplies the address, total byte length,
element format, item size, one-dimensional shape, byte stride, and read-only
state requested by `memoryview`. A memoryview retains its exporter, so the
native allocation remains alive even if the originating tensor is deleted:

```python
data = tensor.data
del tensor
value = data[0]
```

The effective writability of an export is the combination of storage
writability and the particular tensor view's writability. A read-only tensor
must return a read-only memoryview even when another tensor sharing the same
storage is writable.

`CTensor` may cache its memoryview so repeated access does not repeatedly
allocate Python wrapper objects. Different tensor views do not need to return
the same memoryview; shared storage is a behavioural guarantee rather than a
Python object-identity guarantee.

Using `memoryview` delegates length, indexing, assignment, iteration, negative
index handling, bounds checks, and standard Python buffer behaviour to an
existing Python type. It also provides zero-copy interoperability with other
Python code that accepts the native buffer interface.

The C backend's operation kernels bypass both the memoryview and its exporter.
They access `FnnStorage.data` directly after selecting the native element type.
The distinction therefore matters only for representation tests, debugging,
inspection, and deliberate manipulation of storage from Python; it does not
affect normal tensor operations.

A dedicated public `CFlatBuffer` extension type is an alternative. It would
make element indexing and assignment rules explicit and give the project full
control over its small surface. It has some merit, particularly as a learning
exercise, but it would duplicate standard Python container behaviour solely
for tests and occasional Python-level storage access. The design therefore
uses the simpler public interface of a standard `memoryview` instead.

## Suggested implementation sequence

1. Continue developing backend-neutral network code to expose real operation
   requirements.
2. Define `FlatBuffer` and `FlatBufferTensor` without changing existing
   `PythonTensor` names.
3. Add `is_contiguous()` and its focused tests when needed by an operation.
4. Extract the reusable representation test suite behind construction hooks.
5. Replace matmul's inner tuple construction and scalar access with a
   generalised direct-buffer calculation that respects offsets and strides for
   every supported operand rank and layout.
6. Replace the remaining tuple-based output and leading-dimension traversal
   with cursor advancement while preserving the full matmul semantics.
7. Extend direct traversal to other operations, preserving operation-specific
   loop structures and general layout handling.
8. Implement native storage, its private buffer exporter, and `CTensor`; expose
   storage through typed memoryviews and run the shared representation tests
   against `CTensor`.
9. Implement a thin `CBackend` and translate the tested Python kernels into
   naive C one operation at a time.
10. Consider specialised kernels, parallelism, or BLAS only after the simple
    implementation is correct and understood.
