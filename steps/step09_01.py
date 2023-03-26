import numpy as np
from step08 import Exp, Square, Variable


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def main():
    # Method 1
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    # Method 2
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


if __name__ == "__main__":
    main()
