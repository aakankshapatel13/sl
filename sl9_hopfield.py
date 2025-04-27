import numpy as np
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.weights += np.dot(p, p.T)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def recall(self, pattern, steps=5):
        pattern = pattern.copy().astype(float)
        for _ in range(steps):
            for i in range(self.size):
                raw = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw >= 0 else -1
        return pattern.astype(int)

patterns = np.array([
    [1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, 1, -1],
    [-1, 1, -1, 1, -1, 1],
    [1, 1, 1, -1, -1, -1]
])

hopfield_net = HopfieldNetwork(size=patterns.shape[1])
hopfield_net.train(patterns)
test_pattern = np.array([1, -1, 1, -1, 1, -1])
output = hopfield_net.recall(test_pattern)
print("Input pattern: ", test_pattern)
print("Recalled pattern:", output)
