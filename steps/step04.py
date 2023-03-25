from step01 import Variable
from step02 import Square
from step03 import Exp


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def main():
    import numpy as np

    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

    # 합성 함수 미분
    def f(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy = numerical_diff(f, x)
    print(dy)


if __name__ == "__main__":
    main()
