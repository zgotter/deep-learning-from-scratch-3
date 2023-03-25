class Variable:
    def __init__(self, data) -> None:
        self.data = data


def main():
    import numpy as np

    data = np.array(1.0)
    x = Variable(data)
    print(x.data)


if __name__ == "__main__":
    main()
