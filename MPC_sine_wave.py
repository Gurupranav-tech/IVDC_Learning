import numpy as np
import matplotlib.pyplot as plt
from MPC import MPC


def main():
    _, original_fig = plt.subplots()
    _, target_fig = plt.subplots()

    descritization = 0.1
    xs = np.arange(0, 2 * np.pi, descritization)
    print(len(xs))
    ys = np.sin(xs)

    desired_trajectory = np.zeros(shape=(xs.shape[0], 2))
    for i, (x, y) in enumerate(zip(xs, ys)):
        desired_trajectory[i] = [x, y]

    model = MPC(1, descritization, 10, ys, ys[0])
    pys = model.complete()

    original_fig.set_title("Sine Wave Road")
    original_fig.plot(xs, ys)

    target_fig.set_title("MPC Road")
    target_fig.plot(xs, pys, color='orange')

    plt.show()


if __name__ == '__main__':
    main()