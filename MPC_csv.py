import matplotlib.pyplot as plt

from MPC import MPC


def main():
    with open('pathdiv.csv', 'r') as file:
        content = file.read().splitlines()[1:]
        xs = []
        ys = []
        for line in content:
            cords = line.split(',')
            xs.append(float(cords[0]))
            ys.append(float(cords[1]))

    model = MPC(1, 0.1, 10, ys, ys[0])
    print(len(xs))
    pys = model.complete()

    _, original_fig = plt.subplots()
    _, target_fig = plt.subplots()

    original_fig.set_title("Original Map")
    original_fig.plot(xs, ys)

    target_fig.set_title("MPC Map")
    target_fig.plot(xs, pys)

    plt.show()


if __name__ == '__main__':
    main()