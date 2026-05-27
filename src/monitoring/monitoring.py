import numpy as np


class DriftMonitor:

    def __init__(self):

        self.reference_distribution = None

    def fit(self, data):

        self.reference_distribution = np.mean(data, axis=0)

    def detect_drift(self, new_data, threshold=0.1):

        new_mean = np.mean(new_data, axis=0)

        drift = np.linalg.norm(
            new_mean - self.reference_distribution
        )

        return drift > threshold