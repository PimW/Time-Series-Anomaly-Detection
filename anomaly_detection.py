import matplotlib.pyplot as plt
import numpy as np

import pysax
import pysequitur

plt.rcParams["figure.figsize"] = (16, 8)

file = 'data/tweakers.net-post-day.csv'


class AnomalyDectector(object):
    def __init__(self, filename):
        self.window_size = 60
        self.at = 0

        self.sax = pysax.SAXModel(window=3, stride=1, nbins=3)
        self.sequitur = pysequitur.SequiturModel("sequitur/sequitur")

        self.timeseries = self.load_file(filename)[-240:]
        self.timeseries = self.normalize(self.timeseries)
        self.density = np.zeros(len(self.timeseries))
        self.window_density = np.zeros(self.window_size)
        self.anomalies = np.zeros(len(self.timeseries))

        self.window_anomalies = np.zeros(len(self.timeseries))
        self.window_anomalies_delta = np.zeros(len(self.timeseries))

    def load_file(self, filename):
        file = open(filename)
        timeseries = [x.strip() for x in file.readlines()]
        timeseries = timeseries[:-1]
        timeseries.extend([0, 0])
        return np.array(timeseries, dtype=float)

    def normalize(self, x):
        return (x - np.mean(x)) / np.std(x)

    def detect(self):
        symbol_string = self.sax.symbolize_signal(self.timeseries)

        for i in range(3, len(symbol_string) + 3):
            start = max(0, i - self.window_size)
            end = min(len(self.timeseries) - 1, i)
            self.at = i
            sequence = symbol_string[start:end]
            self.sequitur.fit(sequence)
            self.grammar_rule_density()
            self.add_window_density(self.window_density, start)
            self.window_anomalies[end] = self.sequitur.size

        self.grammar_rule_density()
        self.anomaly_rating()
        self.anomaly_delta()

    def grammar_rule_density(self):
        self.window_density = np.zeros(self.window_size)
        rules = self.sequitur.rule0.split()
        index = 0
        for rule in rules:
            if not rule.isdigit():
                index += 1
            else:
                index = self.sub_rule_density(self.sequitur.rules[int(rule)], index, 1)

    def sub_rule_density(self, rule, start, depth):
        new_index = start

        for sub_rule in rule['body']:
            if not sub_rule.isdigit():
                self.window_density[new_index] = depth
                new_index += 1
            else:
                new_index = self.sub_rule_density(self.sequitur.rules[int(sub_rule)], new_index, depth + 1)
        return new_index

    def add_window_density(self, sequence, start):
        for i in range(len(sequence)):
            self.density[i + start] += sequence[i] / len(sequence)

    def anomaly_rating(self):
        avg_rule_density = np.mean(self.density)
        anomaly_rating = 0
        for i in range(len(self.anomalies)):
            anomaly_rating = max(0, anomaly_rating + (avg_rule_density - 1.5 * self.density[i]))
            self.anomalies[i] = anomaly_rating

    def window_end_anomaly(self, sequence_density):
        avg_rule_density = np.mean(self.density[:self.at])
        anomaly_rating = 0
        for i in range(len(sequence_density)):
            anomaly_rating = max(0, anomaly_rating + (avg_rule_density - 1.5 * sequence_density[i]))
        return anomaly_rating

    def anomaly_delta(self):
        self.window_anomalies_delta[0] = 0
        for i in range(1, len(self.timeseries)):
            self.window_anomalies_delta[i] = self.window_anomalies[i] - self.window_anomalies[i - 1]  # compression loss

    def plot(self):
        ax1 = plt.subplot(311)
        ax1.set_title("Time Series")
        plt.plot(self.timeseries[self.window_size:])

        # ax2 = plt.subplot(412)
        # ax2.set_title("Grammar Density")
        # plt.plot(self.density[self.window_size:])

        # ax3 = plt.subplot(313)
        # ax3.set_title("Anomaly Rating")
        # plt.plot(self.anomalies)

        ax4 = plt.subplot(312)
        ax4.set_title("Window Compressed Size")
        plt.plot(self.window_anomalies[self.window_size:])

        ax5 = plt.subplot(313)
        ax5.set_title("Anomaly Delta")
        plt.plot(self.window_anomalies_delta[self.window_size:])

        plt.tight_layout()
        plt.savefig('figures/window-tweakers-day-compressed-flatline-final.eps')
        plt.show()


detector = AnomalyDectector(file)
detector.detect()
detector.plot()
