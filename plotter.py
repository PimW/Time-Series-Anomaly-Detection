import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (32, 6)

file = 'data/tweakers.net-post-day.csv'


def load_file(filename):
    file = open(filename)
    timeseries = [x.strip() for x in file.readlines()]
    return np.array(timeseries[-200:-1], dtype=float)


def normalize(x):
        return (x - np.mean(x)) / np.std(x)


timeseries = load_file(file)

for i in range(150, 155):
    timeseries[i] = 0

plt.plot(timeseries)
plt.savefig('%s-flat.eps' % file.split('/')[1].split('.')[0], bbox_inches='tight')
plt.show()