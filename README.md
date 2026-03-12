# python-neural-network

A work-in-progress proof of concept neural network written in Python. This follows the book 'Neural Networks from Scratch in Python', adopting a more robust design and with the ultimate aim of extending the resulting model. In particular, the aim is to expose more of the underlying
formulae and where possible improve performance.

## Notes

Some important things I learned as part of this project which are not obviously expressed by the code/comments:

- All neural networks, whatever their purpose, learn and (once trained) represent a function. If a neural network has been trained to classify pictures which contain a cat, it is still just learning a function.

- Accordingly, we avoid linear activation functions. These limit the network to learning linear functions.

- Don't trust physicists. Tensors have a different meaning in physics vs computer science (they have a different definition in maths vs computer science, but it's close enough to not matter much in the context of this project). In the context of neural networks, high-dimensional space just means arrays with more than three dimensions. Nobody is going to try to squeeze four dimensions into three-dimensional geometry as is common in explanations of e.g. special relativity.

- If you take just two neurons, you can start to see how the network as a whole represents a function. One neuron might look like this:

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

- The most complex function this neuron can represent looks *something* like this. It has a lower bound but no upper bound.

   y
  |          /
  |         /
  |        /
  |______./________________ x


- But with two neurons:

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

- We can represent a function which looks *something* like this. It has both a lower bound and an upper bound.

   y
  |          ____________
  |         /
  |        /
  |______./________________ x

- If we keep adding neurons we increase the complexity of the function (i.e. curve) we can describe with the network.

- 'Area of effect' refers to the region of input space which, given the neuron's weights and bias, results in the neuron activating (outputs something other than zero). Different neurons have different areas of effect. When these are combined, we get the function described by the network.
