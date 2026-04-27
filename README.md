# python-neural-network

A work-in-progress proof of concept neural network written in Python. This follows the book 'Neural Networks from Scratch in Python' (NNfSiP), adopting a more robust design and with the ultimate aim of extending the resulting network. In particular, the aim is to expose more of the underlying mathematical operations and where possible improve performance.

The final code from the NNfSiP book, which encompasses all the concepts taught in the book, can be found [here](https://github.com/Sentdex/nnfs_book/blob/main/Chapter_22/Ch22_Final.py).

## Python compatibility

This project requires Python 3.13 or newer as it is designed with the the option of using free-threaded Python in mind. That aside, it uses typing features only available in 3.11+, e.g. `*tuple[int, ...]` to require a tuple with one `int` as a minimum, and uses 3.12+ `type` declarations.

> NB this project currently uses poetry which doesn't yet support free-threaded Python, so will be transitioned to uv in due course.

## Tensor Backends

NNfSiP and a lot of other Python-based resources on neural networks make heavy use of NumPy for tensor operations. I wanted to fully understand how tensors work and therefore write my own implementations, starting with one in pure Python (NNfSiP does a little work in Python at the beginning to explain weights before moving on to NumPy).

The NumPy tensor backend is a very thin wrapper around the relevant NumPy functions. Producing this early meant it was possible to write a suite of backend contract tests to tightly pin down NumPy's behaviour (e.g. what you get when you multiply two 2D tensors), which could then be run against my own implementations. So although the objective was to wean myself off the convenience provided by NumPy, NumPy was essential to doing so.

Ultimately, my aim was to build a neural network which, for classifying numerical data at least, does not require any imports. It is designed to be runnable, using a pure-Python tensor backend, without the the dependencies required for the other backends (and their tests) installed. This means that, in places, there is some slightly complex import logic, e.g. in `__init__.py` in the `tensors` package. The NumPy-specific backend tests check whether NumPy is available and skip the tests if it isn't but this logic is much more simple.

### Backend Contract

The contract for the tensor backends is expressed using a Protocol class (TensorBackend). When a layer is instantiated it must be passed a backend class which conforms to the TensorBackend contract. The Protocol can't do everything. A lot of the contract is defined through the backend contract tests which were built using NumPy's array methods as a reference implementation.

The backend contract does (or will) enforce the following:

#### Done

##### Types

- Methods which return an index return an `int` and methods which take an index as an argument may be passed an `int`. TODO - should this be 'must'? Probably
- Backend methods only receive and return standard Python types for scalar values (`float`, `int`) and never implementation-specific types (`np.float64`).
- No arrays of rank 0 are returned or can be passed as arguments. Methods will return either at least a 1D array (including an empty array) or a scalar.
- Backend methods only guarantee support for tensors in the native tensor representation used by that backend.
- Each backend must provide a method for converting a rectangular Python nested/un-nested `list` or `tuple` representing at least a tensor of rank 1 or greater into its native tensor type. This method must reject plain scalar values.
- This method must raise `TypeError` or `ValueError` if passed non-numeric values within the `list` or `tuple` (nested `list`s and `tuple`s are fine, as long as the resulting object conforms to the rules on shape).
- The conversion method must accept only Python's built in `int` and `float` types for values.
- Each backend must provide a method for converting an instance of its native tensor type to a nested/un-nested `list`. It cannot return a rank 0 tensor or a scalar because the contract does not allow tensors to represent these.
- The values returned by this method (within the `list`) must be `float`s or `int`s.

##### Numeric operations

- The backend contract does not currently prescribe the behaviour of conventionally forbidden floating-point operations such as division by zero, taking the logarithm of zero or a negative value, or taking the square root of a negative value.
- When `sum` is called on an empty tensor it returns `0.0`, so code which totals values can continue without special handling.
- Other reductions such as `mean`, `max`, `min` and `std` must raise `ValueError` on an empty tensor, because there is no single, obvious value these might sensibly return.

##### Shape

- Tensors must be rectangular: along each axis, every nested sub-array must have the same length. Non-rectangular tensors must raise `ValueError` when passed to any method which accepts a tensor as an input.
- `reshape` must preserve the total number of elements. If the target shape would require a different number of elements, it must raise `ValueError`.
- Zero-length dimensions are allowed in the target shape when calling `reshape`, provided the total number of elements is unchanged.
- `reshape` must reject any negative value in the target shape.
> This is a feature of NumPy which causes the method to infer the size of a single dimension from the number of elements and the size of the other dimensions if `-1` is passed as the size of that dimension. It is not needed and makes an already complex method more difficult to implement.
- Where a method requires tensors to have the same shape, or shapes which are compatible under the relevant broadcasting rules, any mismatch must raise `ValueError`.

##### Axes Rules

- `transpose` must accept either `None` or a full axes `tuple`. When an axes `tuple` is provided, it gives the new order of the axes.
- `sum`, `mean`, `max`, `min` and `std` must accept `None`, a single `int`, or a `tuple` of `int`s for the axis argument. For these reduction methods, `axis=1` and `axis=(1,)` are equivalent.
- When `keepdims=False`, the reduced axes are removed. If this removes all axes, the method returns a plain Python scalar rather than a rank 0 tensor.
- When `keepdims=True`, the reduced axes are kept with length `1`, so the result remains a tensor.

##### Broadcasting

- The elementwise binary methods `add`, `subtract`, `multiply`, `divide`, `maximum` and `minimum` must all follow the same broadcasting rules.
- Where a method signature allows a scalar argument, backends must support applying that scalar elementwise across the tensor.
- The order of arguments is part of the contract: backends are only required to support scalar arguments in the positions explicitly allowed by the method signature.
- For tensor-with-tensor operations, shapes are compared from the end. Two dimensions are compatible if they are equal, or if one of them is `1`.
- If one tensor has fewer dimensions, it is treated as if dimensions of length `1` had been added on the left before comparison.
- The result shape is built one axis at a time. Where two compatible dimensions differ because one of them is `1`, the result takes the other dimension.
- If any pair of aligned dimensions is neither equal nor `1`, the operation must raise `ValueError`.

##### Aliasing/Views

- The backend contract does not guarantee whether a method returns a copy of a tensor or a view onto existing data.
- Backends are free to avoid copying internally where they can, but code using the backend must not rely on mutating one tensor to affect another.

##### Other

#### To do

##### Axes Rules
- Where a method takes an axes `tuple` to reorder axes, the order of the axes in the `tuple` is part of the contract and must be followed exactly.
- Where a method takes an axes `tuple` to identify which axes to operate on, the `tuple` identifies the set of axes to use; the order of those axes is not part of the contract.
- Negative axes are allowed wherever a method accepts an axis or axes `tuple`, and are interpreted by counting back from the end of the tensor shape.
- Duplicate axes are invalid and must raise `ValueError`.
- Any axis outside the valid range for the tensor shape is invalid and must raise `ValueError`.
- `keepdims` is accepted only by the reduction methods `sum`, `mean`, `max`, `min` and `std`.
- `argmax` does not support the `keepdims` parameter.
- `argmax` must accept `None` or a single `int` for the axis argument.

##### Aliasing/Views

- `copy` is the only method which guarantees an independent tensor with the same values.

##### Other
- For `empty` and `empty_like`, only the shape is part of the contract; the values returned are not.
- When input violates the contract for a method, the method must raise `ValueError` rather than guessing or silently adjusting the input. Where the violation is fundamentally about Python input type rather than value or shape, `TypeError` is also acceptable.

#### No decision(s) made
- No decisions have yet been made about the exact contract for stack, concatenate, vstack and hstack. This includes which input shapes and ranks they must accept, how the axis argument should work where relevant, and what should happen when the inputs are empty or their shapes do not match. These decisions have been deferred until the methods themselves are being tested and implemented.

#### Interoperability

- The project is designed such that only one backend will be used in the same instance of a network. The purpose of having multiple backends is to learn about tensor operations and provide a choice when a new network is instantiated.
- The shared backend contract tests exist to pin down the behaviour of the reference implementation closely enough that other backends can be built against it with confidence. This means that where floating-point arithmetic is tested, the tests are designed to allow small differences between actual and expected values, within clearly defined tolerances.

### Reference Design

The backend contract defines what every backend must do. The reference design is narrower: it describes the first family of backends this project is actually building towards.

The reference design is float-based. It is intended for the NumPy backend, the first pure-Python backend, and any later full-fat CPU/GPU backend used for ordinary training or inference. Quantised backends may deliberately diverge from this design while still sharing as much of the backend contract and structural test coverage as possible.

The reference design does (or will) enforce the following:

#### Done

##### Types And Values

- Tensor values are represented as `float`s internally.
- Python `int` values may be accepted at the contract boundary where this is convenient, but they are normalised to `float` values inside tensors.
- Methods which create tensors with values, such as `zeros`, `ones`, `full`, `eye` and `randn`, return float-valued tensors.
- Methods which return scalar numeric results, other than index-returning methods, return plain Python `float`s.
- Methods which return indices return plain Python `int`s.

##### Conversion

- `to_tensor` converts acceptable Python numeric values to the backend's native `float`-valued tensor representation.
- `to_python` converts native tensors back to Python lists containing `float`s.

##### Arithmetic

- Arithmetic follows ordinary floating-point behaviour, subject to the tolerances used in the backend contract tests.
- Tests using non-integer `float` values are reference-design tests. They are not expected to be reusable unchanged for quantised or integer-valued backends.
- Conventionally forbidden floating-point operations complete rather than raising in reference-design backends, which must surface the resulting special values at the Python boundary using ordinary Python `float` values, namely `float("nan")`, `float("inf")` and `float("-inf")`.
- Reduction methods return float-valued tensors when the result is not scalar.

##### Random Tensor Creation

- `randn` returns float-valued tensors.
- Backends should support seeding so tests and examples can be reproduced, but different backends do not have to generate the same random values from the same seed.

#### To do

##### Numeric Operations

- Elementwise arithmetic methods such as `add`, `subtract`, `multiply`, `divide`, `maximum` and `minimum` must return float-valued tensors.
- Unary methods such as `exp`, `log`, `sqrt`, `absolute`, `sign` and `clip` must return float-valued tensors.
- Where conventionally forbidden floating-point operations produce special values, scalar-returning methods may return those values and `to_python` may include them within returned Python lists.

##### Creation Methods

- `zeros_like`, `ones_like` and `full_like` must return float-valued tensors with the same shape as the input tensor.
- `full_like` must normalise an `int` fill value to `float` values in the returned tensor.
- `copy` must return an independent tensor with the same shape and values as the input tensor.
- `empty` and `empty_like` must return tensors with the requested shape, but their values are not part of the contract.
- Any dtype or native-representation expectations for `empty` and `empty_like` belong to reference-design or implementation-level tests, not the universal backend contract.

#### No decision(s) made

- Whether reference-design backends should provide additional shared operations for detecting or replacing non-finite values within tensors, such as `isfinite`, `isnan`, `isinf` or `nan_to_num`.
- `isfinite` and `nan_to_num` seem to be the common handlers in other tensor implementations, so these are probably the priority.
- How much shared behaviour should be required for later operations on tensors which already contain such special values, beyond surfacing them consistently at the Python boundary.
- Whether to implement methods for masking, clipping, replacement or other approaches to dealing with special values.

#### Relationship With The Backend Contract

- Every reference-design backend must satisfy the backend contract.
- Not every backend which satisfies the backend contract has to satisfy the reference design.
- Quantised or integer-valued backends may use different internal numeric representations and different arithmetic tests.
- The test suite therefore separates reusable contract tests from reference-design tests wherever practical.

## Design And Architecture

- The network will always be orchestrated and described in Python.
- Tensor representations will be native to their respective backend.
- Only scalar values and other narrow interface points should normally cross the boundary.
- The tensor backends are concerned only with tensors and operations on them.
- Other concepts present in the NNfSiP book, such as input handling, batching orchestration, sample loading, image preprocessing, serialisation, and labelling, are not considered part of the backend. These should be handled separately and remain pluggable.
- No assumptions are made about the eventual use of the network beyond the need for the core tensor and network logic to remain adaptable.

### Tensor backends

The tensor-backend design is intended to leave room for future backends with different internal representations and numeric types. It has three layers:

1. The backend contract.
   This defines what every backend must do and the behaviour the rest of the network may rely on.

2. The reference design.
   This is a narrower design used for the first implementations. It is float-based and is the design currently being followed by the Python and planned C backends.

3. Individual implementations.
   Each backend has its own native tensor representation and may require a small amount of implementation-specific testing. NumPy is the reference implementation used to pin down the shared contract tests, but other backends are not required to reproduce every NumPy convenience or edge case.

- Backends may be written in any language with good Python integration, though in practice only Python (including Cython) and C are planned.
- The goal is that all backends which follow the reference design can run against the same unittest suite apart from a small number of implementation-specific tests.
- Backends which diverge from the reference design, for example by using `int` values internally, should still be able to use most of the shared backend contract tests.

#### Python

- The first custom backend will be written in Python. Its initial tensor representation will be nested lists, and its initial scalar representation will be Python’s native float type.
- Neither choice is intended to be permanent. If a later bespoke tensor type or bespoke numeric type becomes useful, those should be introduced in a way that can be plugged in or removed cleanly rather than becoming entangled with the rest of the backend.
- The Python backend should remain compatible with Cython where possible.
- If a Cython-oriented variant later rules out a bespoke tensor representation or bespoke number representation, that is acceptable, provided those features remain optional rather than forcing a redesign of the backend. Compatibility with the MicroPython Viper emitter would also be desirable, but is currently an aspiration rather than a hard requirement. It may turn out to be incompatible with the simultaneous use of ordinary Python type annotations and Cython-friendly structure.

#### C

- A C backend is also planned, and is the only non-Python backend the project is likely to pursue in practice.
- The C implementation should favour a flat contiguous buffer plus shape metadata over nested arrays.
- Maximising code reuse across float and quantized arithmetic, x86 CPU execution, possible later CUDA support, CPython extension integration, and MicroPython extension integration is a central design goal. This means, in particular, separating shape, indexing, axis, and other structural logic from the arithmetic core wherever practical, so that later work on quantized inference can reuse as much non-arithmetic code as possible.
- Low-level arithmetic code must be as pure and dependency-light as possible
- It is important to avoid tensors crossing the Python or MicroPython boundary any more than strictly necessary. This has already informed the Protocol-based design.

#### Testing

- Testing will be comprehensive, but the design aims to minimise duplicated effort.
- Tests must cover the functionality needed by NNfSiP as a minimum
- [To Do!] The backend contract tests are, as far as possible split into shape/axes/other structural tests and arithmetic tests. This means that should the project evolve to accommodate quantised neural networks, many of the tests can be shared. It means all 'full fat' tensor implementations can share exactly the same set of float-based tests.
- Where backend contract tests are intended to be shared with future non-float or quantised backends, any input values and expected output values used in those tests must be integer-valued `float`s (e.g. `1.0`) or `int`s.
> When requiring values to be `int`s it is important to account for the fact that in Python `bool` is a subclass of `int` so will pass a naive `isinstance()` check.
- This applies to values passed into `to_tensor` and to values checked with `assert_nested_close`.
- Tests which use non-integer `float` values are treated as part of the `float`-based reference design and are not intended to be shared unchanged with non-`float` backends.
- Shared-test decorators are used to enforce these rules in the test suite, including the requirement that shared `assert_nested_close` tests use `rel_tol=0` and `abs_tol=0`.
- Behaviour for conventionally forbidden floating-point operations is not pinned down in the universal backend contract tests. Where shared behaviour is required across the float-based reference backends, it belongs in the reference-design tests instead.
- Some overlap between arithmetic and structural or semantic testing is accepted where separating them further would make the tests less clear or less useful.
- The tests are written to cover at least some higher-rank work, including 4D tensors, not because every immediate use case requires them, but to ensure that the network and tensor backends are adaptable to a range of use cases.

##### to_python and to_tensor

The backend contract tests are designed to be implementation-agnostic, so their inputs, expected values and observed results need to be expressed using plain Python types. For this reason each backend provides `to_tensor` and `to_python` methods for, respectively, converting Python `list`/`tuple` structures to the backend's tensor representation and from those to `list`s.

This creates a tension in the test design. We want the tests to remain independent of backend-specific dependencies such as NumPy, but we also want to avoid implementing and maintaining separate conversion logic inside the test suite.

The chosen design is:

- Backend contract tests for tensor operations such as `matmul` use each backend's own `to_tensor` and `to_python` methods to complete the round-trip between Python values and backend-native tensors.
- The risk of relying on those methods within other tests is accepted in order to keep each conversion path implemented only once, in the backend itself.
- `to_tensor` and `to_python` are therefore exceptions to the general rule that backend method tests are universal.
- Each method has thorough implementation-specific tests, because the shared tests for those methods cannot by themselves catch all classes of implementation defect.
> These tests are therefor critical. Problems here could result in false successes elsewhere in the test suite
- Each method also has shared contract tests, which define the universal behaviour expected of all backends.


#### Quantised backends

- Quantization will be a possible, later phase, not part of the baseline design.
- Only after a proof-of-concept network with Python and C tensor backends is successfully running on an x86 CPU, will this be attempted. However, the possible implementation of a quantised inference network has informed the testing strategy.
> It actually informed the testing strategy a little too late, requiring some splitting out of structural and arithmetic tests in some modules.
- Each backend capable of supporting quantization will need a different internal tensor representation and different arithmetic implementations.
- Quantised implementations will share as much structural code and as much of the existing structural test coverage as possible, while adding only the extra representations, arithmetic logic, and tests needed for both.


## Docstrings

There are a lot. In some cases these represent my description of what a function/class is doing, i.e. like a docstring. In the cases of the more complex operations (e.g. the matmul backend tests), writing them was an important part of my learning process. In some of these cases I went over them again and again until I was sure I understood the operations concerned and they accurately reflected the code. I also worked hard to avoid needless mathematical notation or terminology. If you find they differ from explanations in authoritative sources trust the latter (and please raise an issue or PR!).

## Use of AI to complete this project

A chief purpose of this project was to learn how neural networks really work. This would be completely undermined by letting coding agents/LLMs write much code. AI coding tools, specifically GitHub Co-pilot (using Sonnet 4.x) and OpenAI Codex (GPT 5.x), were used to:

- Fix small, tedious problems e.g. headaches with the pre-commit hooks
- Answer many hundreds of questions, especially about:
     - the operations implemented by the tensor backends, especially the more complex operations and how they work with higher-rank tensors
     - whether my instincts about what to allow and not in the backend contract (e.g. forbidding zero rank arrays) were sensible
     - whether design decisions were likely to cause headaches later when I came to implementing concepts I didn't yet understand
     - whether my comments and docstrings were accurate
     - what needed to be tested, especially with more complex calculations, e.g. matmul
- Lay down boilerpate code, e.g. test method stubs which I could then work through one-by-one
- Propose text for docstrings but barely ever simply write them
> Codex was great at reading test methods and producing text which stepped through the operations using actual values. It was less good at explanations and there was a lot of back and forth to get the docstrings both accurate and easy to follow. Claude was noticibly better at striking a balance. None of this was helped by the fact that I was learning as I went.
- Quickly produce things like arrays for tests and make close copies of existing tests, especially in the backend contract tests where the reference implementation (NumPy) was certain and the purpose of the tests was to fully describe its behaviour
> Codex did remarkably well at producing both input and output arrays for tests. I had assumed I would have to ask it to quickly knock up some input arrays then run (e.g.) `np.matmul` in REPL to get the expected results but it was often able to produce the results as well. That said, I can't stress enough how much of a bad idea this would have been if I had not been writing tests around a known, good implementation (see the section on how I built the tensor backends). Interestingly, when the tests turned to higher-rank tensors, Codex started running Python commands to get the tensors without being asked. It also became clear, as I started work on the backend tests, that it's trivial to generate input tensors manually with numpy with something like `np.random.randn(2, 3)`.
> As the project grew I became more relaxed about Codex writing whole test methods in the contract tests (stress: I knew the implementation worked because it was effectively testing NumPy). It was good but weaker on organising the tests and suggesting appropriate test coverage for individual operations.
- Carry out simple but laborious refactoring, e.g. splitting code out into different modules as the project grew and I became more sure of the design.
> I stopped using Codex for this once the project reached approx 15+ modules as it became quicker to do it than explain it.
- Parse the source from the NNfSiP book to help answer questions about the code needed to be adapted. E.g. NNfSiP makes a lot of use of dot product which I decided not to implement in the backend, in favour of the more generic `matmul`.
- Write new logic or edit existing logic very sparingly, in very small steps, and only when I was sure what it was doing.

## References

[What is a tensor in deep learning](https://medium.com/data-science/what-is-a-tensor-in-deep-learning-6dedd95d6507)

The final code from the NNfSiP book, which encompasses all the concepts taught in the book, can be found [here](https://github.com/Sentdex/nnfs_book/blob/main/Chapter_22/Ch22_Final.py).

## Handling special values

Writing tests for the elementwise operations in the backend contract and reference tests required some careful thinking about how to handle 'special values' such as `inf` and `nan`. These are produced by NumPy in cases where a conventionally forbidden operation takes place. E.g. `1 / 0 = inf` and `0 / 0 = nan`. This section summarises the considerations which went into determining how these are handled in the tensor backends and the network code.

Where a tensor operation produces `nan`, `inf` or `-inf` the usual objectives are:

- prevent special values appearing where possible by using safe versions of the operations concerned;
- detect them quickly if they appear unexpectedly;
- decide, at the level of the network or application, whether the right response is to stop, mask, clip, replace, or otherwise handle them.

> 'mask' means treating certain positions in a tensor as special, usually by keeping a separate true/false structure saying which values are valid and which are not. This lets later code ignore, replace or handle only those positions while keeping the tensor’s shape unchanged.
> 'clip' means forcing values into a chosen range. For example, clipping probabilities to stay between `1e-7` and `1 - 1e-7` avoids values like `0.0` or `1.0` which would otherwise cause problems for operations such as log(...).

This matters because a small number of `nan`/`inf` values can spread quickly. One bad activation can make a loss `nan`; one bad gradient can make a weight `nan`; and one bad weight can then affect many later outputs through `matmul`.

The NNfSiP code already handles this mainly by prevention rather than cleanup. For example, the softmax activation avoids overflow by subtracting the maximum value before exponentiation:

```python
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
```

The categorical and binary cross-entropy losses clip predicted probabilities before taking logarithms, so they do not evaluate `log(0)`:

```python
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
negative_log_likelihoods = -np.log(correct_confidences)
```

```python
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
sample_losses = -(
    y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
)
```

The adaptive optimizers add a small epsilon before division, so they do not divide by zero when normalising updates:

> 'epsilon' means a very small positive constant added for numerical safety. It is not intended to change the calculation in any meaningful way when the denominator is already a sensible size; its job is to stop the denominator from becoming exactly zero, or so close to zero that the change to a parameter such as a weight or bias becomes extremely large.

```python
layer.weights += (
    -self.current_learning_rate
    * layer.dweights
    / (np.sqrt(layer.weight_cache) + self.epsilon)
)
```

These examples show the normal pattern: if there is a well-understood safe version of an operation, that is usually preferred to allowing special values to appear and then trying to recover afterwards.

If special values do appear unexpectedly, the right response depends on what the tensor represents. In training code, an unexpected `nan` or `inf` is often a sign that something has gone wrong in optimisation and the best response is to stop and diagnose the problem. In inference code, a fallback such as clipping or replacement may be acceptable. In such cases, handling such values should be the responsibility of the network code rather than the tensor backend.

## Notes

Some important things I learned as part of this project which are not obviously expressed by the code/comments:

- All neural networks, whatever their purpose, learn and (once trained) represent a function. If a neural network has been trained to classify pictures which contain a cat, it is still just learning a function.

- Accordingly, we avoid linear activation functions. These limit the network to learning linear functions.

- Don't trust physicists. Tensors have a different meaning in physics vs computer science (they have a different definition in maths vs computer science, but it's close enough to not matter much in the context of this project). In the context of neural networks, high-dimensional space just means arrays with more than three dimensions. Nobody is going to try to squeeze four dimensions into three-dimensional geometry as is common in explanations of e.g. special relativity.

- Some people get very cross when it comes to what does and does not count as a tensor. None of the following descriptions could be said to constitute a definition but, for the purposes of this project, all of them hold:
    - 'A tensor is an object which can be represented as an array' (NNfSiP)
    - 'The biggest difference between a numpy array and a PyTorch Tensor is that a PyTorch Tensor can run on either CPU or GPU' (PyTorch docs)

- If you take just two neurons, you can start to see how the network as a whole represents a function. One neuron might look like this:

```
    x
    |
    \* w
    |
    +--( b )
    |
   ( Σ )
    |
  [ReLU]
    |
    y
```

- The most complex function this neuron can represent looks *something* like this. It has a lower bound but no upper bound.
```
  y
|          /
|         /
|        /
|______./________________ x
```

- But with two neurons:

```
   x
    |
    \* w1 + b1
    |
   ( Σ )
    |
  [ReLU]
    |
    h
    |
    \* w2 + b2
    |
   ( Σ )
    |
  [ReLU]
    |
    y
```

- We can represent a function which looks *something* like this. It has both a lower bound and an upper bound.

```
   y
  |          ____________
  |         /
  |        /
  |______./________________ x
```

- If we keep adding neurons we increase the complexity of the function (i.e. curve) we can describe with the network.

- 'Area of effect' refers to the region of input space which, given the neuron's weights and bias, results in the neuron activating (outputs something other than zero). Different neurons have different areas of effect. When these are combined, we get the function described by the network.
