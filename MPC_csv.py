import matplotlib.pyplot as plt


def main():
    with open('pathdiv.csv', 'r') as file:
        content = file.read().splitlines()[1:]
        xs = []
        ys = []
        for line in content:
            cords = line.split(',')
            xs.append(float(cords[0]))
            ys.append(float(cords[1]))

    _, original_fig = plt.subplots()

    original_fig.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    main()