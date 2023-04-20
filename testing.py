import numpy as np
import matplotlib.pyplot as plt


def generate_points(k, size):
    s = size // k
    loc = [x for x in range(0, (5 * k + 1), 5)]
    xs = []
    ys = []
    # s = num_datapoints // 2 because this is the size of each cluster
    for i in range(k):
        x = np.random.normal(loc=loc[i], size=s)
        y = np.random.normal(loc=loc[i], size=s)

        xs = np.concatenate((xs, x), axis=0)
        ys = np.concatenate((ys, y), axis=0)

    return np.array([list(pair) for pair in zip(xs, ys)])

data = generate_points(3, 16)

plt.scatter(*data[:].T)
plt.show()


