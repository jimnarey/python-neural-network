# python-neural-network

A work-in-progress proof of concept neural network written in Python. This follows the book 'Neural Networks from Scratch in Python' (NNfSiP), adopting a more robust design and with the ultimate aim of extending the resulting model. In particular, the aim is to expose more of the underlying mathematical operations and where possible improve performance.

The final code from the NNfSiP book, which encompasses all the concepts taught in the book, can be found [here](https://github.com/Sentdex/nnfs_book/blob/main/Chapter_22/Ch22_Final.py).

## Python compatibility

This project requires Python 3.13 or newer. It is designed with the at least the option of using free-threaded Python in mind. That aside, it uses typing features only available in 3.11+, e.g. `*tuple[int, ...]` to require a tuple with one `int` as a minimum.

> NB this project currently uses poetry which doesn't yet support free-threaded Python, so will be transitioned to uv in due course.

## Tensor Backends

NNfSiP and a lot of other Python-based resources on neural networks make heavy use of NumPy for tensor operations. I wanted to fully understand how tensors work and therefore write my own implementations, starting with one in pure Python (NNfSiP does a little of this at the beginning to explain weights before moving on to NumPy).

The NumPy tensor backend is a very thin wrapper around the relevant NumPy functions. Producing this early meant it was possible to write a suite of backend contract tests to tightly pin down NumPy's behaviour (e.g. what you get when you multiply two 2D arrays), which could then be run against my own implementations. So although the objective was to wean myself off the convenience provided by NumPy, NumPy was essential to doing so. I can't overstate how essential it was to have a solid reference implementation to work from.

Ultimately, my aim was to build a neural network which, for classifying numerical data at least, did not require any imports.

### Backend Contract

The contract for the tensor backends is expressed using a Protocol class (TensorBackend). When a layer is instantiated it must be passed a backend class which conforms to the TensorBackend contract. The Protocol can't do everything. A lot of the contract is defined through the backend contract tests which were built using NumPy's array methods as a reference implementation.

The backend contract does (or will) enforce the following:

#### Done

- No arrays of rank 0 are returned or can be passed as arguments. Methods will return either at least a 1D array (including an empty array) or a scalar.
- Methods which return an index return an `int` and methods which take an index as an argument may be passed an `int`.
- Backend methods only receive and return standard Python types for scalar values (`float`, `int`) and never implementation-specific types (`np.float64`).

#### To do

##### Types

- Values within arrays are always `float`s, including internally (not just when values are exposed to the wider application). Methods which return a single scalar value always return a `float` unless that value is an index.
> There is a potential performance hit in places with this but it keeps things much simpler and, for the most part, tensor operations require `float`s anyway. Where this principle differs from NumPy's implementation and interface is that some array operations, e.g. `ones`, would normally return an array of `int`s.
- Backend methods only guarantee support for tensors in the native tensor representation used by that backend.
- Each backend must provide a method for converting a rectangular Python nested `list` or `tuple` representing at least a 1D tensor into its native tensor type. This method must reject plain scalar values.

##### Numeric operations
- Conventionally forbidden numeric operations, such as division by zero or taking the logarithm of a non-positive value, must raise an exception rather than returning special values.
- When `sum` is called on an empty array it returns 0.0, so code which totals values can continue without special handling.
- Other reductions such as `mean`, `max`, `min` and `std` must raise an exception on an empty array, because there is no single, obvious value these might sensibly return.

##### Shape
- Arrays must be rectangular: along each axis, every nested sub-array must have the same length. Non-rectangular arrays must raise an exception when passed as inputs.
- `reshape` must preserve the total number of elements. If the target shape would require a different number of elements, it must raise an exception.
- Zero-length dimensions are allowed in the target shape when calling `reshape`, provided the total number of elements is unchanged.
- Where a method requires tensors to have the same shape, or shapes which are compatible under the relevant broadcasting rules, any mismatch must raise an exception.

##### Axes Rules
- Where a method takes an axes `tuple` to reorder axes, the order of the axes in the `tuple` is part of the contract and must be followed exactly.
- Where a method takes an axes `tuple` to identify which axes to operate on, the `tuple` identifies the set of axes to use; the order of those axes is not part of the contract.
- Negative axes are allowed wherever a method accepts an axis or axes `tuple`, and are interpreted by counting back from the end of the tensor shape.
- Duplicate axes are invalid and must raise an exception.
- Any axis outside the valid range for the tensor shape is invalid and must raise an exception.
- `transpose` must accept either `None` or a full axes `tuple`. When an axes `tuple` is provided, it gives the new order of the axes.
- `sum`, `mean`, `max`, `min` and `std` must accept `None`, a single `int`, or a `tuple` of `int`s for the axis argument. For these reduction methods, axis=1 and axis=(1,) are equivalent.
- `keepdims` is accepted only by the reduction methods `sum`, `mean`, `max`, `min` and `std`.
- When `keepdims=False`, the reduced axes are removed. If this removes all axes, the method returns a plain Python scalar rather than a rank `0` array.
- When `keepdims=True`, the reduced axes are kept with length `1`, so the result remains an array.
- `argmax` does not support the `keepdims` parameter.
- `argmax` must accept `None` or a single `int` for the axis argument.

##### Broadcasting
- The elementwise binary methods `add`, `subtract`, `multiply`, `divide`, `maximum` and `minimum` must all follow the same broadcasting rules.
- Where a method signature allows a scalar argument, backends must support applying that scalar elementwise across the tensor.
- The order of arguments is part of the contract: backends are only required to support scalar arguments in the positions explicitly allowed by the method signature.
- For tensor-with-tensor operations, shapes are compared from the end. Two dimensions are compatible if they are equal, or if one of them is `1`.
- If one tensor has fewer dimensions, it is treated as if dimensions of length `1` had been added on the left before comparison.
- The result shape is built one axis at a time. Where two compatible dimensions differ because one of them is 1, the result takes the other dimension.
- If any pair of aligned dimensions is neither equal nor `1`, the operation must raise an exception.

##### Aliasing/Views
- The backend contract does not guarantee whether a method returns a copy of a tensor or a view onto existing data.
- Backends are free to avoid copying internally where they can, but code using the backend must not rely on mutating one tensor to affect another.
- `copy` is the method which guarantees an independent tensor with the same values.

##### Other
- For `empty` and `empty_like`, only the shape is part of the contract; the values returned are not.
- When input violates the contract for a method, the method must raise an exception rather than guessing or silently adjusting the input

##### No decision(s) made
- No decisions have yet been made about the exact contract for stack, concatenate, vstack and hstack. This includes which input shapes and ranks they must accept, how the axis argument should work where relevant, and what should happen when the inputs are empty or their shapes do not match. These decisions have been deferred until the methods themselves are being tested and implemented.

#### Interoperability

- The project is designed such that only one backend will be used in the same instance of a network. The purpose of having multiple backends is to learn about tensor operations and provide a choice when a new network is instantiated.
- The shared backend contract tests exist to pin down the behaviour of the reference implementation closely enough that other backends can be built against it with confidence, while still allowing small floating-point differences, within tolerances.

### Implementation planning

I haven't got to writing the custom backends yet but the following outlines some design choices I have made (at least for the time being) to act as a reference as I progress.

### Test Coverage

The backends will use NumPy as a reference implementation. The various backend tests were written against the NumPy implementation and were used to describe the behaviour of NumPy's tensor (array) operations so - with some changes noted in the Backend Contract section - this behaviour could be replicated. NumPy's operations often have a huge amount of functionality not required by the network built in the NNfSiP book and a lot of convenience features which are really just a matter of preference.

- Tests cover the functionality needed by NNfSiP as a minimum
- Beyond that, they include selected NumPy behavior only where doing so helps push implementations toward a general and extensible design
> For example, the matmul tests include some higher-dimensional cases not used directly in NNfSiP, because they encourage a more general implementation which should be easier to extend later.
- They do not try to reproduce every NumPy convenience or edge case

### Python

- Use nested lists as the main tensor representation
- Use plain Python scalars only for scalar-returning operations, not tensor values (i.e. no rank 0 tensors)
- Do not introduce a custom Python array/tensor type unless a later need clearly justifies it.

### C

- Do not represent tensors as nested C arrays in the general case.
- Use a flat contiguous block of memory plus shape metadata as the core tensor representation.
- The flat-buffer-plus-shape-metadata design leaves room to avoid copying internally where that is useful, without making aliasing part of the contract
- Keep the scalar type configurable in one place so the same core code can be built around `float` or `double` as needed.
- Keep the low-level maths code as pure and dependency-light as possible so it can be reused from a CPython wrapper, a MicroPython wrapper, and later backends.
- Treat the C backend as the performance-oriented implementation; the Python backend does not need to mirror its internal representation exactly.


## Docstrings

There are a lot. In some cases these represent my description of what a function/class is doing, i.e. like a docstring. In the cases of the more complex operations (e.g. the matmul backend tests), writing them was an important part of my learning process. In some of these cases I went over them again and again until I was sure I understood the operations concerned and they accurately reflected the code. I also worked hard to avoid needless mathematical notation or terminology. If you find they differ from explanations in authoritative sources trust the latter (and please raise an issue or PR!).

## Use of AI to complete this project

A chief purpose of this project was to learn how neural networks really work. This would be completely undermined by letting coding agents/LLMs to write much code. AI, specifically GitHub Co-pilot and OpenAI Codex, were used to:

- Fix small, tedious problems e.g. headaches with the pre-commit hooks
- Answer many hundreds of questions, especially about:
     - the operations implemented by the tensor backends
     - whether design decisions were likely to cause headaches later when I came to implementing concepts I didn't yet understand
     - whether my comments and docstrings were accurate
     - what needed to be tested, especially with more complex matmul calculations
- Lay down boilerpate code, e.g. method stubs which I could then work through one-by-one
- Quickly produce things like arrays for tests and make close copies of existing tests, especially in the backend contract tests where the reference implementation (NumPy) was certain and the purpose of the tests was to fully describe its behaviour
> Codex did remarkably well at producing both input and output arrays for these tests. I had assumed I would have to ask it to quickly knock up some input arrays then run (e.g.) `np.matmul` in REPL to get the expected results but it was often able to produce the results as well. That said, I can't stress enough how much of a bad idea this would have been if I had not been writing tests around a known, good implementation (see the section on how I built the tensor backends).
- Carry out simple but laborious refactoring, e.g. splitting code out into different modules as the project grew and I became more sure of the design.
- Parse the source from the NNfSiP book to help answer questions about the code needed to be adapted. E.g. NNfSiP makes a lot of use of dot product which I decided not to implement in the backend, in favour of the more generic `matmul`.
- Write new logic or edit existing logic very sparingly, in very small steps, and only when I was sure what it was doing.

## References

[What is a tensor in deep learning](https://medium.com/data-science/what-is-a-tensor-in-deep-learning-6dedd95d6507)

The final code from the NNfSiP book, which encompasses all the concepts taught in the book, can be found [here](https://github.com/Sentdex/nnfs_book/blob/main/Chapter_22/Ch22_Final.py).

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
