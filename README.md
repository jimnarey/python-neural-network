# python-neural-network

A work-in-progress proof of concept neural network written in Python. This follows the book 'Neural Networks from Scratch in Python' (NNfSiP), adopting a more robust design and with the ultimate aim of extending the resulting model. In particular, the aim is to expose more of the underlying mathematical operations and where possible improve performance.

The final code from the NNfSiP book, which encompasses all the concepts taught in the book, can be found [here](https://github.com/Sentdex/nnfs_book/blob/main/Chapter_22/Ch22_Final.py).

## Python compatibility

This project requires Python 3.13 or newer. It is designed with the at least the option of using free-threaded Python in mind. That aside, it uses typing features only available in 3.11+, e.g. `*tuple[int, ...]` to require a tuple with one `int` as a minimum.

> NB this project currently uses poetry which doesn't yet support free-threaded Python, so will be transitioned to uv in due course.

## Approach to developing the tensor backends

NNfSiP and a lot of other Python-based resources on neural networks make heavy use of NumPy for tensor operations. I wanted to fully understand how tensors work and therefore write my own implementations, starting with one in pure Python (NNfSiP does a little of this at the beginning to explain weights before moving on to NumPy).

The NumPy tensor backend is a very thin wrapper around the relevant NumPy functions. Producing this early meant it was possible to write a suite of backend contract tests to tightly pin down NumPy's behaviour (e.g. what you get when you multiply two 2D arrays), which could then be run against my own implementations. So although the objective was to wean myself off the convenience provided by NumPy, NumPy was essential to doing so. I can't overstate how essential it was to have a solid reference implementation to work from.

Ultimately, my aim was to build a neural network which, for classifying numerical data at least, did not require any imports.

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

## Backend planning

I haven't got to writing the custom backends yet but the following outlines some design choices I have made (at least for the time being) to act as a reference as I progress.

### Python

- Use nested lists as the main tensor representation and plain Python scalars for rank 0 values.
- Do not introduce a custom Python array/tensor type unless a later need clearly justifies it.

### C

- Do not represent tensors as nested C arrays in the general case.
- Use a flat contiguous block of memory plus shape metadata as the core tensor representation.
- Keep the scalar type configurable in one place so the same core code can be built around `float` or `double` as needed.
- Keep the low-level maths code as pure and dependency-light as possible so it can be reused from a CPython wrapper, a MicroPython wrapper, and later backends.
- Treat the C backend as the performance-oriented implementation; the Python backend does not need to mirror its internal representation exactly.
