import numpy as np
import matplotlib.pyplot as plt


def main():
    _, original_fig = plt.subplots()
    _, target_fig = plt.subplots()

    xs = np.arange(0, 2 * np.pi, 0.1)
    ys = np.sin(xs)

    original_fig.set_title("Sine Wave Road")
    original_fig.plot(xs, ys)

    target_fig.set_title("MPC Road")

    plt.show()


if __name__ == '__main__':
    main()