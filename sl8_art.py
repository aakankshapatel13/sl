import numpy as np
class ART1:
    def __init__(self, input_size, vigilance=0.75):
        self.input_size = input_size
        self.vigilance = vigilance
        self.weights = []

    def _match_score(self, input_pattern, weight):

        intersection = np.minimum(input_pattern, weight)
        return np.sum(intersection) / np.sum(input_pattern)

    def _choose_cluster(self, input_pattern):
        for i, w in enumerate(self.weights):
            if self._match_score(input_pattern, w) >= self.vigilance:
                return i
        return -1

    def _update_weights(self, input_pattern, cluster_index):
        self.weights[cluster_index] = np.minimum(self.weights[cluster_index], input_pattern)

    def train(self, patterns):
        clusters = []
        for pattern in patterns:
            pattern = np.array(pattern)
            idx = self._choose_cluster(pattern)
            if idx == -1:
                self.weights.append(pattern.copy())
                clusters.append(len(self.weights) - 1)
            else:
                self._update_weights(pattern, idx)
                clusters.append(idx)
        return clusters

if __name__ == "__main__":
    patterns = [
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1]
    ]

    art = ART1(input_size=5, vigilance=0.75)
    clusters = art.train(patterns)

    for i, (pattern, cluster) in enumerate(zip(patterns, clusters)):
        print(f"Pattern {i+1}: {pattern} → Cluster {cluster}")
