# Euler's number. Unique in that if e^x is plotted on a graph the
# slope of the curve equals x at every point (d/dx * e^x = e^x).

e = 2.718281828459045

# It is used in the softmax activation function. Other bases could
# be used but they result in more complex equations.

# The curve produced by e^x is differentiable at all points, i.e.
# you can draw a tangent which precisely describes the slope
# (rate of change) at that point. Curves which have corners or
# cusps are not.

# Run the following (with venv enabled) to see the curve produced by e^x:
# python -c "import numpy as np, matplotlib.pyplot as plt; x=np.linspace(-2,2,400); plt.plot(x,np.exp(x)); plt.title('y = e^x'); plt.axhline(0); plt.axvline(0); plt.show()"

# The curve produced by |x| is not differentiable everywhere (at 0):
# python -c "import numpy as np, matplotlib.pyplot as plt; x=np.linspace(-2,2,400); plt.plot(x,np.abs(x)); plt.title('y = |x| (not differentiable at x=0)'); plt.axhline(0); plt.axvline(0); plt.show()"
